"""Cached, full-coherent panel-d calculation for orientation refinement.

Only b-independent density, form-factor, thickness and photon-field work is
reused. The inherited float32 physics, physical longitudinal-unit bridge,
impact-parameter nodes, native FFT grid and Stokes-to-decay projection are
unchanged. The optional combined FFT uses linearity before Stokes bilinears;
``separate`` retains the archived branchwise FFT order for regression checks.
"""
from __future__ import annotations

import math
import time
import numpy as np
import torch

from .decay import b_quadrature, decay_coefficients, stokes_maps
from .forward import UPCForwardModel
from .observables import case_physics


class PanelDOrientationRefiner:
    """Compute one raw orientation pair without redundant b-independent work.

    Instances mutate their model's temporary b value and are not thread-safe.
    Use one independent instance per process. Thread count is controlled by the
    caller, not this class. No orientation averaging or image normalization is
    performed here.
    """
    def __init__(self, config, case, fft_mode="combined"):
        if fft_mode not in ("combined", "separate"):
            raise ValueError("fft_mode must be combined or separate")
        self.fft_mode = fft_mode
        cfg = config["study"]
        if cfg["longitudinal_units"] != "physical":
            raise ValueError("orientation refinement requires the physical longitudinal-unit bridge")
        if (cfg["b_min_fm"] != 12 or cfg["b_max_fm"] != 60 or cfg["b_nodes"] != 25
                or cfg["n_y"] != 5 or cfg["ymax"] != 1
                or cfg["n_radial"] != 757 or cfg["n_phi"] != 256):
            raise ValueError("refinement must preserve original b12:2:60, y5, radial757, phi256 scope")
        self.physics, self.density = case_physics(config, case)
        self.model = UPCForwardModel(self.physics, cfg["longitudinal_units"]).eval()
        self.case = dict(case)
        if self.model._output_L/2 <= cfg["b_max_fm"]+16:
            raise ValueError("field domain does not cover the b window and nuclear support")
        self.axis = self.model._p_axis_q
        self.radii = torch.linspace(0, float(self.axis.abs().max())*math.sqrt(2)*(1+1e-6), cfg["n_radial"])
        self.b_values = torch.linspace(cfg["b_min_fm"], cfg["b_max_fm"], cfg["b_nodes"], dtype=torch.float64)
        self.b_weights = b_quadrature(self.b_values)
        self.y_values, self.y_weights = np.polynomial.legendre.leggauss(cfg["n_y"])
        self.y_values *= cfg["ymax"]
        self.y_weights *= cfg["ymax"]
        self.y_tensor = torch.tensor(self.y_values[None], dtype=torch.float32)
        self.n_phi = cfg["n_phi"]

    @staticmethod
    def _rotation(value):
        value = torch.as_tensor(value, dtype=torch.float32, device="cpu")
        if value.shape == (3, 3):
            value = value[None]
        if value.shape != (1, 3, 3) or not bool(torch.isfinite(value).all()):
            raise ValueError("one finite 3-by-3 orientation matrix per nucleus is required")
        eye = torch.eye(3, dtype=value.dtype)[None]
        if (not torch.allclose(value@value.transpose(-1, -2), eye, rtol=0, atol=2e-5)
                or not torch.allclose(torch.linalg.det(value), torch.ones(1), rtol=0, atol=2e-5)):
            raise ValueError("orientation must be a proper orthogonal rotation")
        return value

    def _prepare(self, R1, R2):
        model = self.model
        rp1, rt1 = model._compute_density_ws(nuclear_ID=1, R=R1, batch_size=1)
        rp2, rt2 = model._compute_density_ws(nuclear_ID=2, R=R2, batch_size=1)
        rp1, rt1 = model._validate_external_density_pair(rp1, rt1, nuclear_ID=1,
            normalization_rtol=5e-4, negativity_atol=1e-8)
        rp2, rt2 = model._validate_external_density_pair(rp2, rt2, nuclear_ID=2,
            normalization_rtol=5e-4, negativity_atol=1e-8)
        ny = len(self.y_values)
        f1 = model._expand_to_BVY(model._compute_F_3d(rp1), ny)
        f2 = model._expand_to_BVY(model._compute_F_3d(rp2), ny)
        rt1 = model._expand_to_BVY(rt1, ny)
        rt2 = model._expand_to_BVY(rt2, ny)
        ys = self.y_tensor.to(device=rp1.device, dtype=rp1.dtype).reshape(ny)
        q1 = model.Mjpsi*.5/model.gamma_1*torch.exp(ys)
        q2 = model.Mjpsi*.5/model.gamma_2*torch.exp(-ys)
        t1, thick1 = model._compute_T_prim_and_thickness(rt1, q1, Z_nuc=model.Z_1)
        t2, thick2 = model._compute_T_prim_and_thickness(rt2, q2, Z_nuc=model.Z_2)
        vx1, vy1 = model._compute_photon_field(f1, model.gamma_1, q1, Z_nuc=model.Z_1)
        vx2, vy2 = model._compute_photon_field(f2, model.gamma_2, q2, Z_nuc=model.Z_2)
        return {"t1": t1, "t2": t2, "thick1": thick1, "thick2": thick2,
                "vx1": vx1, "vy1": vy1, "vx2": vx2, "vy2": vy2,
                "energy1": .5*model.Mjpsi*torch.exp(ys),
                "energy2": .5*model.Mjpsi*torch.exp(-ys)}

    def _node(self, prepared):
        m, p = self.model, prepared
        ax1, ay1 = m._compute_amplitude_at_grid(nuclear_ID=1,
            T_prim_target=p["t1"], T_thick_source=p["thick2"],
            Vx_source=p["vx2"], Vy_source=p["vy2"], Egamma_target=p["energy2"],
            Ep_target=m.Ep_1, A_source=m.A_2)
        ax2, ay2 = m._compute_amplitude_at_grid(nuclear_ID=2,
            T_prim_target=p["t2"], T_thick_source=p["thick1"],
            Vx_source=p["vx1"], Vy_source=p["vy1"], Egamma_target=p["energy1"],
            Ep_target=m.Ep_2, A_source=m.A_1)
        if self.fft_mode == "combined":
            mx, my, pfull = m._coherent_amplitude_q(ax1+ax2, ay1+ay2)
        else:
            mx1, my1, pfull = m._coherent_amplitude_q(ax1, ay1)
            mx2, my2, _ = m._coherent_amplitude_q(ax2, ay2)
            mx, my = mx1+mx2, my1+my2
        return decay_coefficients(stokes_maps(mx, my), pfull, self.radii, self.n_phi)

    def calculate(self, R1, R2, *, return_nodes=True):
        """Return raw b,y-integrated coefficients and optional node diagnostics."""
        started = time.monotonic()
        R1, R2 = self._rotation(R1), self._rotation(R2)
        old_b = self.model.b_imp
        with torch.inference_mode():
            prepared = self._prepare(R1, R2)
            preparation_seconds = time.monotonic()-started
            nodes = np.empty((len(self.b_values), len(self.y_values), 3, len(self.radii)), dtype=np.float64)
            try:
                for i, b in enumerate(self.b_values):
                    self.model.b_imp = float(b)
                    nodes[i] = self._node(prepared).double().numpy()
            finally:
                self.model.b_imp = old_b
        bw = self.b_weights.numpy()
        coefficients = np.einsum("b,y,bycr->cr", bw, self.y_weights, nodes)
        coarse = np.arange(0, len(self.b_values), 2)
        coarse_weights = b_quadrature(self.b_values[coarse]).numpy()
        output = {"coefficients": coefficients,
                  "coarse_b_coefficients": np.einsum("b,y,bycr->cr", coarse_weights, self.y_weights, nodes[coarse]),
                  "axis": self.axis.numpy().copy(), "radii": self.radii.numpy().copy(),
                  "p_full_axis": self.model._p_full_q.numpy().copy(),
                  "b_values": self.b_values.numpy().copy(), "b_weights": bw.copy(),
                  "y_values": self.y_values.copy(), "y_weights": self.y_weights.copy(),
                  "fft_mode": self.fft_mode, "preparation_seconds": preparation_seconds,
                  "elapsed_seconds": time.monotonic()-started}
        if return_nodes:
            output["b_node_coefficients"] = nodes
        return output
