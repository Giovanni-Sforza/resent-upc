"""A0-only shape-scan kernel on the unchanged production quadrature.

The reference :class:`DiffractionRefiner` stays untouched.  This implementation
evaluates exactly the native Fourier vertices needed by the stored radial
queries, forms intensities at those vertices, then performs the same bilinear
interpolation.  It never interpolates amplitudes before squaring.  The native
Fourier grid, padding phase, radial/angular nodes, and b/y integration are
unchanged; a separable CZT evaluates only the required original frequencies.
"""
from __future__ import annotations

import math
import time

import torch

from .diffraction_forward import DiffractionRefiner
from .shape_scan_czt import NativeFFTSubsetCZT


class ShapeScanFastRefiner(DiffractionRefiner):
    """Return only saved full/self A0; one instance is not thread-safe.

    ``A0`` is shaped ``(orientation, b_level, branch, saved_radius)``.
    ``b_node_A0``, when requested, is ``(b, orientation, y, branch, radius)``.
    Metadata retains the reference's *full* radial axis, alongside the exact
    stored index subset.  Reduced interpolation does not change that axis.
    """

    engine = "shape_scan_czt_a0_v1"

    def __init__(self, config, case, device="cpu", dtype="float64", fft_mode="separate"):
        if dtype not in ("float64", torch.float64):
            raise ValueError("optimized shape scans require float64/complex128")
        super().__init__(config, case, device=device, dtype=dtype, fft_mode=fft_mode)
        qmax = float(config["study"]["qmax"])
        if not math.isfinite(qmax) or qmax <= 0 or qmax > float(self.radii[-1]):
            raise ValueError("qmax must lie inside the registered radial grid")
        self.stored_indices = torch.nonzero(self.radii <= qmax, as_tuple=False).flatten()
        if len(self.stored_indices) < 3:
            raise ValueError("at least three stored radial nodes are required")
        self.stored_q = self.radii[self.stored_indices]
        queries = self._projection_queries.reshape(len(self.radii), self.n_phi, 2)
        self._saved_queries = queries[self.stored_indices].reshape(-1, 2)
        self._build_sparse_projection()
        self._fourier = NativeFFTSubsetCZT(self.model._output_N, self.model._target_points,
            self._projection_vertices, device=self.device, dtype=self.dtype)
        # Immutable host metadata is copied into each result; no per-batch
        # device synchronization is needed to download these constant grids.
        array = lambda value: value.detach().cpu().numpy().copy()
        self._grid = {
            "b_steps_fm": self.b_steps_fm.copy(), "axis": array(self.axis),
            "radii": array(self.radii), "p_full_axis": array(self.model._p_full_q),
            "b_values": self.b_values.numpy().copy(), "b_weights": array(self.b_level_weights[0]),
            "b_level_weights": array(self.b_level_weights), "y_values": self.y_values.copy(),
            "y_weights": self.y_weights.copy(), "stored_indices": array(self.stored_indices),
            "stored_q": array(self.stored_q),
        }

    def _build_sparse_projection(self):
        """Cache align_corners=True bilinear vertices and weights.

        Coordinate normalization follows ``_interp2d_batched`` and PyTorch's
        align-corners unnormalization.  Cached zero-padding masks also cover
        boundary queries.  Unique vertices remove duplicate samples at q=0 and
        at neighboring radial/angular queries without changing their weights.
        """
        n = self.model._target_points
        lower, upper = self._projection_bounds
        normalized = 2.0 * (self._saved_queries - lower) / (upper - lower) - 1.0
        pixel = ((normalized + 1.0) / 2.0) * (n - 1)
        x, y = pixel.unbind(-1)
        x0, y0 = x.floor(), y.floor()
        x1, y1 = x0 + 1, y0 + 1
        vertices = ((x0, y0), (x0, y1), (x1, y0), (x1, y1))
        weights = ((x1-x)*(y1-y), (x1-x)*(y-y0),
                   (x-x0)*(y1-y), (x-x0)*(y-y0))
        indices, masked_weights = [], []
        for (xx, yy), weight in zip(vertices, weights):
            valid = (xx >= 0) & (xx < n) & (yy >= 0) & (yy < n)
            indices.append(xx.long().clamp(0, n-1)*n + yy.long().clamp(0, n-1))
            masked_weights.append(weight * valid)
        unique, inverse = torch.unique(torch.stack(indices), sorted=True, return_inverse=True)
        self._projection_vertices = unique
        self._projection_inverse = tuple(inverse.unbind(0))
        self._projection_weights = tuple(masked_weights)

    def _fft_at_vertices(self, field):
        """Selected samples of the original padded DFT, before scaling."""
        model = self.model
        selected = self._fourier.transform(field)
        # Preserve the original two operations and their rounding order.
        return selected * (model._output_dx ** 2) / model.hbar_c

    def _project_intensity_vertices(self, intensity):
        """Bilinear interpolation of native intensities, then angular mean."""
        inverse, weights = self._projection_inverse, self._projection_weights
        sampled = intensity.index_select(-1, inverse[0]) * weights[0]
        for index in range(1, 4):
            sampled = sampled + intensity.index_select(-1, inverse[index]) * weights[index]
        return sampled.reshape(*intensity.shape[:-1], len(self.stored_q), self.n_phi).mean(-1)

    def _node_A0(self, prepared):
        model = self.model
        amplitudes = []
        for target, source in ((1, 2), (2, 1)):
            fields = model._compute_amplitude_at_grid(
                target, T_prim_target=prepared[f"t{target}"],
                T_thick_source=prepared[f"thick{source}"],
                Vx_source=prepared[f"vx{source}"], Vy_source=prepared[f"vy{source}"],
                Egamma_target=prepared[f"energy{source}"],
                Ep_target=getattr(model, f"Ep_{target}"), A_source=getattr(model, f"A_{source}"))
            amplitudes.append(tuple(self._fft_at_vertices(field) for field in fields))
        (x1, y1), (x2, y2) = amplitudes

        def s0(x, y):
            return (x.abs().square() + y.abs().square()) / (2*math.pi)

        full = self._project_intensity_vertices(s0(x1+x2, y1+y2))
        self_terms = self._project_intensity_vertices(s0(x1, y1) + s0(x2, y2))
        return torch.stack((full, self_terms), dim=-2)

    def calculate(self, R1, R2, *, return_nodes=False):
        self._sync()
        started = time.monotonic()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        r1, r2 = self._rotation(R1), self._rotation(R2)
        if len(r1) != len(r2):
            raise ValueError("both nuclei must have the same number of paired rotations")
        batch, nr, nb = len(r1), len(self.stored_q), len(self.b_host)
        old_b, preparation_seconds = self.model.b_imp, 0.
        with torch.inference_mode():
            base = self._prepare(r1, r2)
            self._sync()
            preparation_seconds += time.monotonic() - started
            integrated_y = torch.zeros((nb, batch, 2, nr), dtype=torch.float64, device=self.device)
            nodes = (torch.empty((nb, batch, self.n_y, 2, nr), dtype=torch.float64,
                                 device=self.device) if return_nodes else None)
            try:
                for start in range(0, self.n_y, self.y_batch_size):
                    stop = min(start+self.y_batch_size, self.n_y)
                    self._sync()
                    preparing = time.monotonic()
                    prepared = self._prepare_y(base, batch, start, stop)
                    self._sync()
                    preparation_seconds += time.monotonic() - preparing
                    for bi, b in enumerate(self.b_host):
                        self.model.b_imp = b
                        value = self._node_A0(prepared).reshape(batch, stop-start, 2, nr).double()
                        integrated_y[bi] += torch.einsum("y,oysr->osr", self.y_weight_tensor[start:stop], value)
                        if nodes is not None:
                            nodes[bi, :, start:stop] = value
                    del prepared
            finally:
                self.model.b_imp = old_b
            levels = torch.einsum("lb,bosr->losr", self.b_level_weights, integrated_y)
            if not bool(torch.isfinite(levels).all()) or bool(torch.any(levels < 0)):
                raise ValueError("nonfinite or negative A0 after integration")
        self._sync()
        array = lambda value: value.detach().cpu().numpy().copy()
        output = {key: value.copy() for key, value in self._grid.items()}
        output.update(
            A0=array(levels.permute(1, 0, 2, 3)), branch_names=list(self.branch_names),
            n_orientations=batch, engine=self.engine, fft_mode=self.fft_mode,
            device=str(self.device), dtype=str(self.dtype).removeprefix("torch."),
            y_batch_size=self.y_batch_size, preparation_seconds=preparation_seconds,
            fourier_backend="native_frequency_subset_czt",
            fourier_crop_shape=list(self._fourier.crop_shape),
            native_dft_side=self.model._target_points,
            czt_crop_side=max(self._fourier.crop_shape),
            projection_vertices=len(self._projection_vertices),
            native_fft_vertices=int(self.model._target_points**2),
            evaluated_intensity_vertices=len(self._projection_vertices),
            peak_memory_allocated_bytes=(torch.cuda.max_memory_allocated(self.device)
                                         if self.device.type == "cuda" else 0),
            peak_memory_reserved_bytes=(torch.cuda.max_memory_reserved(self.device)
                                        if self.device.type == "cuda" else 0))
        if nodes is not None:
            output["b_node_A0"] = array(nodes)
        self._sync()
        output["elapsed_seconds"] = time.monotonic()-started
        return output
