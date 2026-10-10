"""Independent CPU/CUDA panel-d kernel with precision chosen at construction.

The frozen production kernels are not modified.  All grids and spherical
Woods--Saxon caches are constructed in the requested real precision, before any
device transfer.  Float32 retains the legacy grid arithmetic.  Only coherent
amplitudes are evaluated, and Stokes bilinears precede interpolation and all
orientation, impact-parameter and rapidity averages.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from .decay import b_quadrature, stokes_maps
from .forward import UPCForwardModel
from .observables import case_physics
from .vendor.upc_forward import _interp2d_batched


def _real_dtype(value):
    choices = {"float32": torch.float32, "float64": torch.float64,
               torch.float32: torch.float32, torch.float64: torch.float64}
    if value not in choices:
        raise ValueError("dtype must be float32 or float64")
    return choices[value]


class PrecisionUPCForwardModel(UPCForwardModel):
    """Build native grids and static spherical caches directly in FP32/FP64."""

    def __init__(self, config, longitudinal_units="physical", dtype="float64"):
        self._construction_dtype = _real_dtype(dtype)
        # The inherited constructor calls our grid builder, then calculates
        # spherical density and complex form-factor caches from those grids.
        super().__init__(config, longitudinal_units)

    def _setup_grids(self):
        dtype = self._construction_dtype
        for name in ("WS", "T_prim", "output"):
            params = self.grid_params[name]
            n, length = int(params["N"]), float(params["L"])
            if n < 2 or n % 2 or not math.isfinite(length) or length <= 0:
                raise ValueError("spatial grids require even N >= 2 and finite positive L")
            dx = length/n
            x = torch.arange(n, dtype=dtype)*dx-length/2
            raw = torch.fft.fftfreq(n, d=dx, dtype=dtype)*(2*math.pi)
            ordered = raw.sort().values
            for suffix, values in (("x_grid", x), ("k_raw", raw), ("k_sorted", ordered)):
                self.register_buffer(f"_{name}_{suffix}", values)
            for suffix, value in (("N", n), ("L", length), ("dx", dx),
                                  ("k_min_float", float(ordered[0])),
                                  ("k_max_float", float(ordered[-1]))):
                setattr(self, f"_{name}_{suffix}", value)
        if self._WS_N != self._T_prim_N or self._WS_L != self._T_prim_L:
            raise ValueError("WS and thickness must use the same spatial grid")
        for names, values in ((("_output_X", "_output_Y"), self._output_x_grid),
                              (("_output_kx", "_output_ky"), self._output_k_raw)):
            for name, field in zip(names, torch.meshgrid(values, values, indexing="ij")):
                self.register_buffer(name, field)
        if self.safe_padding_factor < 1 or self.target_size_q < 3 or self.p_limit_q <= 0:
            raise ValueError("positive padding, momentum limit and target_size_q >= 3 required")
        self._target_points = self._output_N*self.safe_padding_factor
        full = torch.fft.fftshift(torch.fft.fftfreq(
            self._target_points, d=self._output_dx, dtype=dtype))*(2*math.pi*self.hbar_c)
        self.register_buffer("_p_full_q", full)
        self._p_full_q_min_float, self._p_full_q_max_float = float(full[0]), float(full[-1])
        axis = torch.linspace(-self.p_limit_q, self.p_limit_q, self.target_size_q, dtype=dtype)
        self.register_buffer("_p_axis_q", axis)
        x, y = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("_p_axis_q_interp_coords", torch.stack((x, y), -1).reshape(-1, 2))
        self.register_buffer("_p_axis_q_radius2", x*x+y*y)

    def _compute_amplitude_at_grid(self, nuclear_ID, T_prim_target, T_thick_source,
                                   Vx_source, Vy_source, Egamma_target, Ep_target,
                                   A_source=None):
        # The legacy primitive creates shift tensors in the global default
        # dtype.  Explicit dtype is necessary for non-binary b values in FP64.
        batch, n = T_prim_target.shape[0], self._output_N
        if nuclear_ID not in (1, 2):
            raise ValueError("nuclear_ID must be 1 or 2")
        if A_source is None:
            A_source = self.A_2 if nuclear_ID == 1 else self.A_1
        coords = torch.stack((self._output_X, self._output_Y), -1).reshape(1, -1, 2).expand(batch, -1, -1)
        sign = -1 if nuclear_ID == 1 else 1
        shift = coords.new_tensor((sign*self.b_imp/2, 0.)).reshape(1, 1, 2)
        target, source = coords+shift, coords-shift
        tmin, tmax = -self._T_prim_L/2, self._T_prim_L/2-self._T_prim_dx
        omin, omax = -self._output_L/2, self._output_L/2-self._output_dx
        t = _interp2d_batched(T_prim_target, target, tmin, tmax, tmin, tmax).reshape(batch, n, n)
        gamma = self._gamma_amp_at_points(t, Egamma_target, Ep_target)
        thick = _interp2d_batched(T_thick_source, source, tmin, tmax, tmin, tmax)
        probability = self._prob(thick, A_nuc=A_source).reshape(batch, n, n)
        vx = _interp2d_batched(Vx_source, source, omin, omax, omin, omax).reshape(batch, n, n)
        vy = _interp2d_batched(Vy_source, source, omin, omax, omin, omax).reshape(batch, n, n)
        scalar = gamma*probability.to(gamma.dtype)
        return vx*scalar, vy*scalar


class PrecisionPanelDRefiner:
    """One-process CPU/CUDA kernel; each call preserves individual orientations.

    The instance mutates ``model.b_imp`` temporarily and is not thread-safe.
    ``study.y_batch_size`` bounds rapidity-expanded fields.  Returned arrays are
    NumPy float64 except axes, which preserve the selected grid precision.
    ``b_level_coefficients`` are ordered finest, twice, four times the b step,
    omitting coarse steps that do not reach the exact same endpoints.
    """

    def __init__(self, config, case, device="cpu", dtype="float64", fft_mode="combined"):
        self.dtype, self.device = _real_dtype(dtype), torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("only cpu and cuda devices are supported")
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable; CPU fallback is not allowed")
            index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            if index >= torch.cuda.device_count():
                raise ValueError("requested CUDA device does not exist")
            self.device = torch.device("cuda", index)
        if fft_mode not in ("combined", "separate"):
            raise ValueError("fft_mode must be combined or separate")
        self.fft_mode, self.case = fft_mode, dict(case)
        cfg = config["study"]
        if cfg.get("longitudinal_units", "physical") != "physical":
            raise ValueError("precision refinement requires physical longitudinal units")
        self.n_phi, self.n_y = int(cfg["n_phi"]), int(cfg["n_y"])
        nr, ymax = int(cfg["n_radial"]), float(cfg["ymax"])
        self.y_batch_size = int(cfg.get("y_batch_size", self.n_y))
        if (self.n_phi < 8 or self.n_phi % 4 or self.n_y < 1 or nr < 2
                or not math.isfinite(ymax) or ymax <= 0 or self.y_batch_size < 1):
            raise ValueError("invalid radial, angular or rapidity quadrature")
        lo, hi = float(cfg["b_min_fm"]), float(cfg["b_max_fm"])
        if not math.isfinite(lo+hi) or not 0 < lo < hi:
            raise ValueError("impact-parameter endpoints must satisfy 0 < b_min < b_max")
        if "b_step_fm" in cfg:
            step = float(cfg["b_step_fm"])
            if not math.isfinite(step) or step <= 0:
                raise ValueError("b_step_fm must be finite and positive")
            intervals = int(round((hi-lo)/step))
            if intervals < 1 or not math.isclose(intervals*step, hi-lo, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("b_step_fm must divide the complete impact-parameter window")
        else:
            intervals = int(cfg["b_nodes"])-1
            if intervals < 1:
                raise ValueError("at least two impact-parameter nodes are required")
            step = (hi-lo)/intervals
        self.b_values = torch.linspace(lo, hi, intervals+1, dtype=torch.float64)
        self.b_host = self.b_values.tolist()
        strides = [s for s in (1, 2, 4) if intervals % s == 0]
        self.b_steps_fm = np.asarray([step*s for s in strides], dtype=np.float64)
        weights = torch.zeros((len(strides), intervals+1), dtype=torch.float64)
        for i, stride in enumerate(strides):
            weights[i, ::stride] = b_quadrature(self.b_values[::stride])
        self.b_level_weights = weights.to(self.device)
        self.y_values, self.y_weights = np.polynomial.legendre.leggauss(self.n_y)
        self.y_values *= ymax
        self.y_weights *= ymax
        self.y_tensor = torch.as_tensor(self.y_values, dtype=self.dtype, device=self.device)
        self.y_weight_tensor = torch.as_tensor(self.y_weights, dtype=torch.float64, device=self.device)
        self.physics, self.density = case_physics(config, case)
        self.model = PrecisionUPCForwardModel(self.physics, "physical", self.dtype).to(self.device).eval()
        if any(self.model._nuclear_configs[i]["density_model"] != "woods_saxon" for i in (1, 2)):
            raise ValueError("precision panel-d kernel currently supports Woods--Saxon nuclei only")
        if self.model._output_L/2 <= hi+16:
            raise ValueError("field domain does not cover the impact-parameter window plus nuclear support")
        self.axis = self.model._p_axis_q
        self.radii = torch.linspace(0, float(self.axis.abs().max())*math.sqrt(2)*(1+1e-6),
                                    nr, dtype=self.dtype, device=self.device)
        if float(self.radii[-1]) > min(-self.model._p_full_q_min_float, self.model._p_full_q_max_float):
            raise ValueError("radial projection exceeds the complete native FFT circles")
        # Match polar_stokes exactly, but build these b/y-independent queries
        # once. Cached Python bounds avoid CUDA scalar synchronizations at every
        # b node. Stokes are still formed on the native grid before interpolation.
        phi = torch.arange(self.n_phi, dtype=self.dtype, device=self.device)*(2*math.pi/self.n_phi)
        coordinates = torch.stack((self.radii[:, None]*phi.cos(),
                                   self.radii[:, None]*phi.sin()), -1)
        self._projection_queries = coordinates.reshape(1, -1, 2)
        self._projection_cos2, self._projection_sin2 = (2*phi).cos(), (2*phi).sin()
        self._projection_bounds = (self.model._p_full_q_min_float, self.model._p_full_q_max_float)

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _rotation(self, value):
        value = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        if value.shape == (3, 3):
            value = value[None]
        if value.ndim != 3 or value.shape[1:] != (3, 3) or not len(value) or not bool(torch.isfinite(value).all()):
            raise ValueError("rotations must be a nonempty finite (B,3,3) array")
        identity = torch.eye(3, dtype=self.dtype, device=self.device).expand(len(value), -1, -1)
        if (not torch.allclose(value@value.transpose(-1, -2), identity, rtol=0, atol=2e-5)
                or not torch.allclose(torch.linalg.det(value), torch.ones(len(value), dtype=self.dtype,
                                     device=self.device), rtol=0, atol=2e-5)):
            raise ValueError("rotations must be proper orthogonal matrices")
        return value

    def _prepare(self, r1, r2):
        """Density and form factors are reused for every rapidity and b node."""
        prepared = {}
        for nid, rotation in ((1, r1), (2, r2)):
            proton, total = self.model._compute_density_ws(nid, rotation, batch_size=len(rotation))
            proton, total = self.model._validate_external_density_pair(
                proton, total, nuclear_ID=nid, normalization_rtol=5e-4, negativity_atol=1e-8)
            prepared[f"f{nid}"] = self.model._compute_F_3d(proton)
            prepared[f"rho{nid}"] = total
        return prepared

    def _prepare_y(self, base, batch, start, stop):
        m, ny = self.model, stop-start
        ys = self.y_tensor[start:stop][None].expand(batch, -1).reshape(-1)
        result = {}
        for nid, sign in ((1, 1), (2, -1)):
            gamma, charge = getattr(m, f"gamma_{nid}"), getattr(m, f"Z_{nid}")
            energy = .5*m.Mjpsi*torch.exp(sign*ys)
            q = m.Mjpsi*.5/gamma*torch.exp(sign*ys)
            total = m._expand_to_BVY(base[f"rho{nid}"], ny)
            form = m._expand_to_BVY(base[f"f{nid}"], ny)
            result[f"t{nid}"], result[f"thick{nid}"] = m._compute_T_prim_and_thickness(total, q, Z_nuc=charge)
            result[f"vx{nid}"], result[f"vy{nid}"] = m._compute_photon_field(form, gamma, q, Z_nuc=charge)
            result[f"energy{nid}"] = energy
        return result

    def _node(self, p):
        m = self.model
        fields = []
        for target, source in ((1, 2), (2, 1)):
            fields.append(m._compute_amplitude_at_grid(target,
                T_prim_target=p[f"t{target}"], T_thick_source=p[f"thick{source}"],
                Vx_source=p[f"vx{source}"], Vy_source=p[f"vy{source}"],
                Egamma_target=p[f"energy{source}"], Ep_target=getattr(m, f"Ep_{target}"),
                A_source=getattr(m, f"A_{source}")))
        if self.fft_mode == "combined":
            mx, my, axis = m._coherent_amplitude_q(fields[0][0]+fields[1][0], fields[0][1]+fields[1][1])
        else:
            mx1, my1, axis = m._coherent_amplitude_q(*fields[0])
            mx2, my2, _ = m._coherent_amplitude_q(*fields[1])
            mx, my = mx1+mx2, my1+my2
        return self._project_stokes(stokes_maps(mx, my))

    def _project_stokes(self, stokes):
        """Original polar_stokes/decay_coefficients operations with cached grids."""
        flat = stokes.reshape(-1, *stokes.shape[-2:])
        queries = self._projection_queries.expand(len(flat), -1, -1)
        lower, upper = self._projection_bounds
        sampled = _interp2d_batched(flat, queries, lower, upper, lower, upper)
        s0, s1, s2 = sampled.reshape(*stokes.shape[:-3], 3, len(self.radii), self.n_phi).unbind(-3)
        return torch.stack((s0.mean(-1),
            (s1*self._projection_cos2+s2*self._projection_sin2).mean(-1),
            (s1*self._projection_sin2-s2*self._projection_cos2).mean(-1)), -2)

    def calculate(self, R1, R2, *, return_nodes=False):
        """Return per-orientation raw coefficients; do not normalize orientations."""
        self._sync()
        started = time.monotonic()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        r1, r2 = self._rotation(R1), self._rotation(R2)
        if len(r1) != len(r2):
            raise ValueError("both nuclei must have the same number of paired rotations")
        batch, nr, nb = len(r1), len(self.radii), len(self.b_host)
        old_b = self.model.b_imp
        preparation_seconds = 0.
        with torch.inference_mode():
            base = self._prepare(r1, r2)
            self._sync()
            preparation_seconds += time.monotonic()-started
            integrated_y = torch.zeros((nb, batch, 3, nr), dtype=torch.float64, device=self.device)
            nodes = (torch.empty((nb, batch, self.n_y, 3, nr), dtype=torch.float64,
                                device=self.device) if return_nodes else None)
            try:
                for start in range(0, self.n_y, self.y_batch_size):
                    stop = min(start+self.y_batch_size, self.n_y)
                    self._sync()
                    preparing = time.monotonic()
                    prepared = self._prepare_y(base, batch, start, stop)
                    self._sync()
                    preparation_seconds += time.monotonic()-preparing
                    for bi, b in enumerate(self.b_host):
                        self.model.b_imp = b
                        value = self._node(prepared).reshape(batch, stop-start, 3, nr).double()
                        integrated_y[bi] += torch.einsum("y,oycr->ocr", self.y_weight_tensor[start:stop], value)
                        if nodes is not None:
                            nodes[bi, :, start:stop] = value
                    del prepared
            finally:
                self.model.b_imp = old_b
            levels = torch.einsum("lb,bocr->locr", self.b_level_weights, integrated_y)
            if not bool(torch.isfinite(levels).all()) or bool(torch.any(levels[:, :, 0] < 0)):
                raise ValueError("nonfinite coefficients or negative intensity after integration")
        self._sync()
        allocated = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0
        reserved = torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else 0
        def array(value):
            return value.detach().cpu().numpy().copy()
        output = {"coefficients": array(levels[0]), "b_level_coefficients": array(levels),
                  "b_steps_fm": self.b_steps_fm.copy(), "axis": array(self.axis),
                  "radii": array(self.radii), "p_full_axis": array(self.model._p_full_q),
                  "b_values": self.b_values.numpy().copy(), "b_weights": array(self.b_level_weights[0]),
                  "b_level_weights": array(self.b_level_weights), "y_values": self.y_values.copy(),
                  "y_weights": self.y_weights.copy(), "n_orientations": batch,
                  "fft_mode": self.fft_mode, "device": str(self.device),
                  "dtype": str(self.dtype).removeprefix("torch."), "y_batch_size": self.y_batch_size,
                  "preparation_seconds": preparation_seconds,
                  "peak_memory_allocated_bytes": allocated, "peak_memory_reserved_bytes": reserved}
        if nodes is not None:
            output["b_node_coefficients"] = array(nodes)
        self._sync()
        output["elapsed_seconds"] = time.monotonic()-started
        return output
