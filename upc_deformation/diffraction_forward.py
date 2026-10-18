"""Source-resolved panel-d coefficients on the precision production grids.

``full`` retains the coherent sum of the two production amplitudes; ``self``
adds their individual Stokes bilinears.  Both use identical nuclei, rotations,
photon fields, b/y weights and projection.  Bilinears are formed on the native
FFT grid, before interpolation or averaging.  No branch is normalized: the
signed cross contribution is the linear difference ``full - self``.
"""
from __future__ import annotations

import time

import torch

from .decay import stokes_maps
from .precision_forward import PrecisionPanelDRefiner


class DiffractionRefiner(PrecisionPanelDRefiner):
    """Evaluate full and self coefficients together, retaining each orientation.

    The branch axis is ordered ``('full', 'self')``.  All inherited precision,
    geometry and rotation validation applies.  Separate source FFTs are
    required; the coherent result reuses their sum.  As with the parent, one
    instance is not thread-safe because ``model.b_imp`` is changed temporarily.
    """

    branch_names = ("full", "self")

    def __init__(self, config, case, device="cpu", dtype="float64", fft_mode="separate"):
        if fft_mode != "separate":
            raise ValueError("diffraction refinement requires fft_mode='separate'")
        super().__init__(config, case, device=device, dtype=dtype, fft_mode=fft_mode)

    def _node(self, prepared):
        """Return (orientation * rapidity, branch, coefficient, radius)."""
        model = self.model
        amplitudes = []
        for target, source in ((1, 2), (2, 1)):
            fields = model._compute_amplitude_at_grid(
                target,
                T_prim_target=prepared[f"t{target}"],
                T_thick_source=prepared[f"thick{source}"],
                Vx_source=prepared[f"vx{source}"],
                Vy_source=prepared[f"vy{source}"],
                Egamma_target=prepared[f"energy{source}"],
                Ep_target=getattr(model, f"Ep_{target}"),
                A_source=getattr(model, f"A_{source}"),
            )
            mx, my, _ = model._coherent_amplitude_q(*fields)
            amplitudes.append((mx, my))
        (mx1, my1), (mx2, my2) = amplitudes
        # Project each native bilinear map before stacking the small radial
        # coefficients, avoiding an additional full/self native-grid copy.
        full = self._project_stokes(stokes_maps(mx1 + mx2, my1 + my2))
        self_terms = self._project_stokes(stokes_maps(mx1, my1) + stokes_maps(mx2, my2))
        return torch.stack((full, self_terms), dim=-3)

    def calculate(self, R1, R2, *, return_nodes=False):
        """Return unnormalized per-orientation full/self decay coefficients.

        Shapes are ``coefficients: (n, 2, 3, nr)`` and
        ``b_level_coefficients: (levels, n, 2, 3, nr)``.  With ``return_nodes``,
        ``b_node_coefficients`` has shape ``(nb, n, ny, 2, 3, nr)`` before b/y
        integration.  The coefficient axis is the inherited ``(A0, C, D)``;
        these are linear decay coefficients, not normalized angular moments.
        Grid, precision, timing and memory metadata match the parent API.
        """
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
            preparation_seconds += time.monotonic() - started
            integrated_y = torch.zeros((nb, batch, 2, 3, nr), dtype=torch.float64,
                                       device=self.device)
            nodes = (torch.empty((nb, batch, self.n_y, 2, 3, nr), dtype=torch.float64,
                                 device=self.device) if return_nodes else None)
            try:
                for start in range(0, self.n_y, self.y_batch_size):
                    stop = min(start + self.y_batch_size, self.n_y)
                    self._sync()
                    preparing = time.monotonic()
                    prepared = self._prepare_y(base, batch, start, stop)
                    self._sync()
                    preparation_seconds += time.monotonic() - preparing
                    for bi, b in enumerate(self.b_host):
                        self.model.b_imp = b
                        value = self._node(prepared).reshape(batch, stop-start, 2, 3, nr).double()
                        integrated_y[bi] += torch.einsum(
                            "y,oyscr->oscr", self.y_weight_tensor[start:stop], value)
                        if nodes is not None:
                            nodes[bi, :, start:stop] = value
                    del prepared
            finally:
                self.model.b_imp = old_b
            levels = torch.einsum("lb,boscr->loscr", self.b_level_weights, integrated_y)
            if not bool(torch.isfinite(levels).all()) or bool(torch.any(levels[:, :, :, 0] < 0)):
                raise ValueError("nonfinite coefficients or negative branch intensity after integration")
        self._sync()
        allocated = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0
        reserved = torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else 0

        def array(value):
            return value.detach().cpu().numpy().copy()

        output = {
            "coefficients": array(levels[0]), "b_level_coefficients": array(levels),
            "branch_names": list(self.branch_names),
            "b_steps_fm": self.b_steps_fm.copy(), "axis": array(self.axis),
            "radii": array(self.radii), "p_full_axis": array(self.model._p_full_q),
            "b_values": self.b_values.numpy().copy(), "b_weights": array(self.b_level_weights[0]),
            "b_level_weights": array(self.b_level_weights), "y_values": self.y_values.copy(),
            "y_weights": self.y_weights.copy(), "n_orientations": batch,
            "fft_mode": self.fft_mode, "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."), "y_batch_size": self.y_batch_size,
            "preparation_seconds": preparation_seconds,
            "peak_memory_allocated_bytes": allocated, "peak_memory_reserved_bytes": reserved,
        }
        if nodes is not None:
            output["b_node_coefficients"] = array(nodes)
        self._sync()
        output["elapsed_seconds"] = time.monotonic() - started
        return output
