"""Exact-support A0 evaluation with cached geometry and deferred projection.

Only output-grid points whose *discrete bilinear interpolation stencil* reaches
the finite T-prime array can contribute. This is zero-padding support, never a
density-tail threshold. Each source retains its own original-grid translation
phase. Native full/self intensities are integrated before the linear radial
projection; nonlinear Gamma evaluation remains after T-prime interpolation.
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from .shape_scan_crop_czt import NativeFFTCropSubsetCZT
from .shape_scan_fast_forward import ShapeScanFastRefiner


class ShapeScanFastV2Refiner(ShapeScanFastRefiner):
    """FP64 implementation preserving the v1 saved-A0 and metadata contract."""

    engine = "shape_scan_support_czt_a0_v2"

    def __init__(self, config, case, device="cpu", dtype="float64", fft_mode="separate"):
        super().__init__(config, case, device=device, dtype=dtype, fft_mode=fft_mode)
        self._supports = [[self._support(b, target) for target in (1, 2)] for b in self.b_host]
        shapes = sorted({support["shape"] for row in self._supports for support in row})
        self._support_shapes = [list(shape) for shape in shapes]
        self._support_points_max = max(h*w for h, w in shapes)

    @staticmethod
    def _normalized(coordinates, lower, upper):
        # Same arithmetic/order as legacy _interp2d_batched; grid_sample puts
        # physical y on its W axis and physical x on its H axis.
        nx = 2.0*(coordinates[..., 0]-lower)/(upper-lower)-1.0
        ny = 2.0*(coordinates[..., 1]-lower)/(upper-lower)-1.0
        return torch.stack((ny, nx), dim=-1)[None]

    def _support(self, b, target):
        model, n = self.model, self.model._output_N
        coords = torch.stack((model._output_X, model._output_Y), -1)
        sign = -1 if target == 1 else 1
        shift = coords.new_tensor((sign*b/2, 0.))
        target_coords, source_coords = coords+shift, coords-shift
        tlo, thi = -model._T_prim_L/2, model._T_prim_L/2-model._T_prim_dx
        olo, ohi = -model._output_L/2, model._output_L/2-model._output_dx
        target_grid = self._normalized(target_coords, tlo, thi)
        # Probe the actual device operator rather than reproducing its pixel
        # arithmetic: CUDA grid_sample may fuse rounding at a zero-padding
        # boundary. Positive interpolation of an all-ones array identifies
        # every stencil with a nonzero weight, including the one-cell halo.
        ones = torch.ones((1, 1, model._T_prim_N, model._T_prim_N),
                          dtype=self.dtype, device=self.device)
        valid = F.grid_sample(ones, target_grid, mode="bilinear", padding_mode="zeros",
                              align_corners=True)[0, 0] > 0
        rows, columns = torch.nonzero(valid.any(1)).flatten(), torch.nonzero(valid.any(0)).flatten()
        if not len(rows) or not len(columns):
            raise ValueError("registered output grid does not intersect the target interpolation support")
        x0, x1, y0, y1 = int(rows[0]), int(rows[-1])+1, int(columns[0]), int(columns[-1])+1
        selection = (slice(x0, x1), slice(y0, y1))
        # A rectangular product grid must have a rectangular support. Failing
        # this check means a new geometry needs its own support proof.
        if not bool(valid[selection].all()) or int(valid.sum()) != (x1-x0)*(y1-y0):
            raise ValueError("nonrectangular interpolation support")
        source_t = self._normalized(source_coords[selection], tlo, thi)
        overlap = bool((F.grid_sample(ones, source_t, mode="bilinear", padding_mode="zeros",
                                     align_corners=True) > 0).any())
        shape, origin = (x1-x0, y1-y0), (x0, y0)
        return {"target": target, "source": 3-target, "b": b, "shape": shape,
            "origin": origin, "selection": selection,
            "target_grid": target_grid[:, x0:x1, y0:y1].contiguous(),
            "source_thickness_grid": source_t,
            "source_field_grid": self._normalized(source_coords[selection], olo, ohi),
            "source_overlap": overlap,
            "fourier": NativeFFTCropSubsetCZT(n, model._target_points, self._projection_vertices,
                shape, origin, device=self.device, dtype=self.dtype)}

    @staticmethod
    def _sample_packed(combined, grid, counts):
        """Sample already packed real/imaginary channels at one location."""
        sampled = F.grid_sample(combined, grid.expand(combined.shape[0], -1, -1, -1),
                                mode="bilinear", padding_mode="zeros", align_corners=True)
        results, start = [], 0
        for count in counts:
            results.append(torch.complex(sampled[:, start], sampled[:, start+1])
                           if count == 2 else sampled[:, start])
            start += count
        return tuple(results)

    @classmethod
    def _sample_channels(cls, values, grid):
        channels, counts = [], []
        for value in values:
            channels.append(value.real)
            if value.is_complex():
                channels.append(value.imag)
            counts.append(2 if value.is_complex() else 1)
        return cls._sample_packed(torch.stack(channels, dim=1), grid, counts)

    def _prepare_v2(self, r1, r2):
        base = self._prepare(r1, r2)
        model = self.model
        for nid in (1, 2):
            rho = base[f"rho{nid}"]
            thick = rho.sum(dim=-1)*model._WS_dx
            integral = thick.sum(dim=(1, 2), keepdim=True)*model._T_prim_dx*model._T_prim_dx
            scale = getattr(model, f"Z_{nid}")/integral.clamp_min(1e-20)
            base[f"thick{nid}"] = thick*scale
            base[f"scale{nid}"] = scale
            base[f"rho_complex{nid}"] = rho.to(torch.complex128)
        for name, value in base.items():
            if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
                raise ValueError(f"nonfinite base field: {name}")
        base["probabilities"] = {}
        return base

    def _prepare_y_v2(self, base, batch, start, stop):
        model, ny = self.model, stop-start
        ys = self.y_tensor[start:stop][None].expand(batch, -1).reshape(-1)
        result = {"batch": batch, "ny": ny, "base": base}
        z = model._WS_x_grid.to(device=self.device, dtype=self.dtype)
        for nid, sign in ((1, 1), (2, -1)):
            gamma, charge = getattr(model, f"gamma_{nid}"), getattr(model, f"Z_{nid}")
            energy = .5*model.Mjpsi*torch.exp(sign*ys)
            q = model.Mjpsi*.5/gamma*torch.exp(sign*ys)
            # UPCForwardModel converts GeV to inverse fm at this interface.
            # Photon-field normalization still receives the original GeV q.
            inverse_length = q/model.hbar_c
            phase = torch.exp(1j*inverse_length.reshape(batch, ny, 1, 1, 1).to(torch.complex128)
                              *z.reshape(1, 1, 1, 1, -1).to(torch.complex128))
            t = (base[f"rho_complex{nid}"][:, None]*phase).sum(-1)*model._WS_dx
            t = t*base[f"scale{nid}"][:, None].to(torch.complex128)
            result[f"t{nid}"] = t.reshape(batch*ny, model._T_prim_N, model._T_prim_N)
            form = model._expand_to_BVY(base[f"f{nid}"], ny)
            result[f"vx{nid}"], result[f"vy{nid}"] = model._compute_photon_field(form, gamma, q, Z_nuc=charge)
            result[f"energy{nid}"] = energy
        for target, source in ((1, 2), (2, 1)):
            energy = result[f"energy{source}"]
            result[f"sigma{target}"] = model._sigma_total_VN(energy, getattr(model, f"Ep_{target}")).to(
                torch.complex128).reshape(batch*ny, 1, 1)
            result[f"kin{target}"] = torch.sqrt((model.alpha/(16.0*math.pi*max(
                model.vmphotoncoupling_Jpsi, 1e-20)))*4.0*math.pi*energy.clamp_min(0.0)).to(
                torch.complex128).reshape(batch*ny, 1, 1)
        # Pack full spatial fields once per y batch, not once at every b node.
        # Drop the old arrays so channel fusion does not double resident field
        # memory throughout the b loop.
        for nid in (1, 2):
            t = result.pop(f"t{nid}")
            vx, vy = result.pop(f"vx{nid}"), result.pop(f"vy{nid}")
            result[f"t_channels{nid}"] = torch.stack((t.real, t.imag), dim=1)
            result[f"field_channels{nid}"] = torch.stack((vx.real, vx.imag, vy.real, vy.imag), dim=1)
        # Check complete fields *before* support selection. A NaN outside the
        # selected support must not silently disappear through exact-zero logic.
        for name, value in result.items():
            if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
                raise ValueError(f"nonfinite prepared field: {name}")
        if not math.isfinite(model.NQ) or model.NQ == 0:
            raise ValueError("finite nonzero NQ required for Gamma")
        return result

    def _probability(self, prepared, bi, support):
        base = prepared["base"]
        key = (bi, support["target"])
        if key not in base["probabilities"]:
            if not support["source_overlap"]:
                base["probabilities"][key] = None  # exactly _prob(0) == 1
            else:
                thick, = self._sample_channels((base[f"thick{support['source']}"],),
                                                support["source_thickness_grid"])
                base["probabilities"][key] = self.model._prob(thick,
                    A_nuc=getattr(self.model, f"A_{support['source']}"))
        probability = base["probabilities"][key]
        return (None if probability is None else probability[:, None].expand(
            -1, prepared["ny"], -1, -1).reshape(prepared["batch"]*prepared["ny"], *support["shape"]))

    def _amplitude_support(self, prepared, bi, support):
        target, source = support["target"], support["source"]
        t, = self._sample_packed(prepared[f"t_channels{target}"], support["target_grid"], (2,))
        # Preserve Gamma(interpolate(T)); exp must not move before sampling.
        prof = -2.0*(torch.exp(-0.5*prepared[f"sigma{target}"]*t)-1.0)/self.model.NQ
        gamma = prepared[f"kin{target}"]*prof
        probability = self._probability(prepared, bi, support)
        scalar = gamma if probability is None else gamma*probability.to(gamma.dtype)
        vx, vy = self._sample_packed(prepared[f"field_channels{source}"], support["source_field_grid"], (2, 2))
        return torch.stack((vx*scalar, vy*scalar), dim=1)

    def _node_intensity(self, prepared, bi):
        amplitudes = []
        for support in self._supports[bi]:
            fields = self._amplitude_support(prepared, bi, support)
            value = support["fourier"].transform(fields)*(self.model._output_dx**2)/self.model.hbar_c
            amplitudes.append(value.unbind(1))
        (x1, y1), (x2, y2) = amplitudes

        def s0(x, y):
            return (x.abs().square()+y.abs().square())/(2*math.pi)

        return torch.stack((s0(x1+x2, y1+y2), s0(x1, y1)+s0(x2, y2)), dim=-2)

    def calculate(self, R1, R2, *, return_nodes=False):
        self._sync()
        started = time.monotonic()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        r1, r2 = self._rotation(R1), self._rotation(R2)
        if len(r1) != len(r2):
            raise ValueError("both nuclei must have the same number of paired rotations")
        batch, nr, nb, nv = len(r1), len(self.stored_q), len(self.b_host), len(self._projection_vertices)
        old_b, preparation_seconds = self.model.b_imp, 0.
        with torch.inference_mode():
            base = self._prepare_v2(r1, r2)
            self._sync()
            preparation_seconds += time.monotonic()-started
            integrated_y = torch.zeros((nb, batch, 2, nv), dtype=torch.float64, device=self.device)
            nodes = (torch.empty((nb, batch, self.n_y, 2, nr), dtype=torch.float64,
                                 device=self.device) if return_nodes else None)
            try:
                for start in range(0, self.n_y, self.y_batch_size):
                    stop = min(start+self.y_batch_size, self.n_y)
                    self._sync()
                    preparing = time.monotonic()
                    prepared = self._prepare_y_v2(base, batch, start, stop)
                    self._sync()
                    preparation_seconds += time.monotonic()-preparing
                    for bi, b in enumerate(self.b_host):
                        self.model.b_imp = b
                        value = self._node_intensity(prepared, bi).reshape(batch, stop-start, 2, nv)
                        integrated_y[bi] += torch.einsum("y,oysv->osv", self.y_weight_tensor[start:stop], value)
                        if nodes is not None:
                            nodes[bi, :, start:stop] = self._project_intensity_vertices(value)
                    del prepared
            finally:
                self.model.b_imp = old_b
            if not bool(torch.isfinite(integrated_y).all()) or bool(torch.any(integrated_y < 0)):
                raise ValueError("nonfinite or negative native intensity after y integration")
            levels = torch.einsum("lb,bosv->losv", self.b_level_weights, integrated_y)
            levels = self._project_intensity_vertices(levels)
            if not bool(torch.isfinite(levels).all()) or bool(torch.any(levels < 0)):
                raise ValueError("nonfinite or negative A0 after integration")
        self._sync()
        array = lambda value: value.detach().cpu().numpy().copy()
        output = {key: value.copy() for key, value in self._grid.items()}
        output.update(A0=array(levels.permute(1, 0, 2, 3)), branch_names=list(self.branch_names),
            n_orientations=batch, engine=self.engine, fft_mode=self.fft_mode,
            device=str(self.device), dtype=str(self.dtype).removeprefix("torch."),
            y_batch_size=self.y_batch_size, preparation_seconds=preparation_seconds,
            fourier_backend="exact_spatial_support_native_frequency_czt",
            fourier_crop_shape=list(self._fourier.crop_shape), native_dft_side=self.model._target_points,
            czt_crop_side=max(self._fourier.crop_shape), projection_vertices=nv,
            native_fft_vertices=int(self.model._target_points**2), evaluated_intensity_vertices=nv,
            spatial_support_shapes=self._support_shapes, spatial_support_points_max=self._support_points_max,
            original_spatial_points=self.model._output_N**2, projection_after_integration=True,
            thickness_reused_across_y=True, probability_reused_across_y=True,
            sigma_kin_reused_across_b=True, normalized_grids_cached=True,
            nonoverlap_source_nodes=sum(not item["source_overlap"] for row in self._supports for item in row),
            peak_memory_allocated_bytes=(torch.cuda.max_memory_allocated(self.device)
                                         if self.device.type == "cuda" else 0),
            peak_memory_reserved_bytes=(torch.cuda.max_memory_reserved(self.device)
                                        if self.device.type == "cuda" else 0))
        if nodes is not None:
            output["b_node_A0"] = array(nodes)
        self._sync()
        output["elapsed_seconds"] = time.monotonic()-started
        return output
