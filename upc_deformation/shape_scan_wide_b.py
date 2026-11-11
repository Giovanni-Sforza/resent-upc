"""Coherent J/psi shape-scan spectra integrated on a wide impact-parameter window.

Both sources use a finite Woods–Saxon proton density and an isolated transverse
photon field. Sample each reaction on its target-centred patch. Apply the
relative translation phase to the cross term at the actual polar momentum;
interpolate only the smooth, local bilinears. All b cutoffs integrate the same
unnormalized intensities using 2*pi*b*db and the configured rapidity weights.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .decay import b_quadrature
from .shape_scan_crop_czt import NativeFFTCropSubsetCZT
from .shape_scan_fast_forward import ShapeScanFastRefiner
from .shape_scan_free_field import FreePhotonStrip
from .shape_scan_physics import resolve_case
from .vendor.upc_forward import _interp2d_batched


DEFAULT_CUTOFFS = (60., 120., 240., 480., 1000., 10000.)


def impact_grid(lower, cutoffs, near_step, step_scale):
    """Nested trapezoids through requested cutoffs, with fixed refinement breaks."""
    cutoffs = np.asarray(cutoffs, dtype=np.float64)
    if (cutoffs.ndim != 1 or not cutoffs.size or not np.isfinite(cutoffs).all()
            or not np.isfinite([lower, near_step, step_scale]).all()
            or not 0 < lower < cutoffs[0] or not np.all(np.diff(cutoffs) > 0)
            or near_step <= 0 or step_scale <= 0):
        raise ValueError('require positive grid steps and increasing cutoffs above b_min')
    # Use existing fine spacing through 60 fm. Above 1000 fm use a log grid.
    # The spacing changes at these same physical radii even for a single cutoff.
    breaks = np.array([60., 120., 240., 480., 1000.])
    ends = np.union1d(cutoffs, breaks[(lower < breaks) & (breaks < cutoffs[-1])])
    parts = [np.array([float(lower)])]
    for end in ends:
        start = parts[-1][-1]
        band = np.searchsorted(breaks, end)
        if band < len(breaks):
            step = near_step * (1, 2, 4, 8, 20)[band] * step_scale
            n = max(1, math.ceil((end-start)/step))
            part = np.linspace(start, end, n+1)
        else:
            n = max(1, math.ceil(math.log(end/start)/(0.12*step_scale)))
            part = np.geomspace(start, end, n+1)
        parts.append(part[1:])
    nodes = np.concatenate(parts)
    weights = np.zeros((len(cutoffs), len(nodes)))
    for i, upper in enumerate(cutoffs):
        last = np.searchsorted(nodes, upper)
        if nodes[last] != upper:
            raise ValueError('b grid missed a requested integration endpoint')
        weights[i, :last+1] = b_quadrature(torch.from_numpy(nodes[:last+1])).numpy()
    return nodes, weights


class WideBRefiner:
    def __init__(self, cfg, case, *, device='cpu', step_scale=1., cutoffs=DEFAULT_CUTOFFS):
        self.cutoffs = np.asarray(cutoffs, dtype=np.float64)
        study = cfg['study']
        self.b_nodes, weights = impact_grid(study['b_min_fm'], cutoffs, study['b_step_fm'], step_scale)
        kernel_cfg, kernel_case, self.density = resolve_case(cfg, case)
        # Reuse the baseline momentum grid, polar queries and native Fourier
        # vertices. The baseline lens size is used only for this local transform.
        self.local = ShapeScanFastRefiner(kernel_cfg, kernel_case, device=device)
        self.model = m = self.local.model
        self.device = self.local.device
        self.stored_q = self.local.stored_q.cpu().numpy().copy()
        self._weights = torch.as_tensor(weights, device=self.device)
        self._b = torch.as_tensor(self.b_nodes, device=self.device)
        self._qx = self.local._saved_queries[:, 0]
        lo, hi = m._T_prim_x_grid[0]-m._T_prim_dx, m._T_prim_x_grid[-1]+m._T_prim_dx
        axis = m._output_x_grid
        indices = torch.nonzero((lo <= axis) & (axis <= hi)).flatten()
        if not len(indices) or int(indices[0]) == 0 or int(indices[-1]) == len(axis)-1:
            raise ValueError('target-centred patch must fit within the local Fourier grid')
        self.patch = axis[indices]
        self.coords = torch.stack(torch.meshgrid(self.patch, self.patch, indexing='ij'), dim=-1)
        self.fourier = NativeFFTCropSubsetCZT(m._output_N, m._target_points,
            self.local._projection_vertices, (len(indices),)*2, (int(indices[0]),)*2, device=self.device)
        self.strips = {}

    def _on_patch(self, data, coords):
        m = self.model
        lo = -m._T_prim_L/2
        hi = m._T_prim_L/2 - m._T_prim_dx
        points = coords.reshape(1, -1, 2).expand(len(data), -1, -1)
        return _interp2d_batched(data, points, lo, hi, lo, hi).reshape(len(data), *coords.shape[:-1])

    def _source_and_target(self, nid, proton, total, y):
        m = self.model
        direction = 1 if nid == 1 else -1
        energy = .5*m.Mjpsi*math.exp(direction*y)
        gamma, charge = getattr(m, f'gamma_{nid}'), getattr(m, f'Z_{nid}')
        q_gev = energy/gamma
        kappa = q_gev/m.hbar_c
        q = proton.new_full((len(proton),), q_gev)
        t, thick = m._compute_T_prim_and_thickness(total, q, Z_nuc=charge)
        phase_z = torch.exp(1j*kappa*m._WS_x_grid)
        weights = (proton*phase_z).sum(-1)/proton.sum(dim=(1, 2, 3))[:, None, None]
        if kappa not in self.strips:
            self.strips[kappa] = FreePhotonStrip(m._WS_x_grid, kappa, self.b_nodes, self.patch, self.device)
        # Source 1 is seen from target 2 (at -b/2), and vice versa.
        sign = -direction
        field = self.strips[kappa].sample(weights, sign)
        field *= math.sqrt(4*m.alpha*gamma*charge**2/q_gev)
        return self._on_patch(t, self.coords), thick, field, energy

    def _survival(self, thick, nid, sign, start, stop):
        if self.b_nodes[start] > self.model._T_prim_L + 2*self.model._T_prim_dx:
            return 1.
        shifted = self.coords.expand(stop-start, -1, -1, -1).clone()
        shifted[..., 0] += sign*self._b[start:stop, None, None]
        return self.model._prob(self._on_patch(thick, shifted), A_nuc=getattr(self.model, f'A_{nid}'))

    def _polar(self, data):
        weights = self.local._projection_weights
        indices = self.local._projection_inverse
        samples = data.index_select(-1, indices[0])*weights[0]
        for i in range(1, 4):
            samples = samples + data.index_select(-1, indices[i])*weights[i]
        return samples

    def _radial_intensities(self, a, c, b):
        # a and c carry the same local Fourier origin. Retain their signed
        # complex correlation before translating by the centre separation.
        own = self._polar((a.abs().square()+c.abs().square()).sum(dim=2))
        cross = self._polar((a*c.conj()).sum(dim=2))
        phase = torch.exp((-1j/self.model.hbar_c)*b[:, None]*self._qx)
        full = own + 2*(phase*cross).real
        # Positive bilinear weights guarantee full >= 0 analytically. Permit
        # only roundoff from cancellation at a destructive-interference zero.
        # At the farthest b a squared amplitude can be subnormal. Several
        # rounded products can then differ by a few ulps of zero.
        roundoff = 2e-13*own + 16*float(np.nextafter(0., 1.))
        if torch.any(full < -roundoff):
            raise ArithmeticError('negative full intensity exceeds roundoff')
        shape = (*own.shape[:2], len(self.stored_q), self.local.n_phi)
        return torch.stack((full.clamp_min(0).reshape(shape).mean(-1),
                            own.reshape(shape).mean(-1)), dim=2)/(2*math.pi)

    @torch.inference_mode()
    def calculate(self, R1, R2):
        rotations = [self.local._rotation(r) for r in (R1, R2)]
        n = len(rotations[0])
        if n != len(rotations[1]):
            raise ValueError('both nuclei must have the same number of rotations')
        m = self.model
        densities = []
        for nid, r in enumerate(rotations, start=1):
            density = m._compute_density_ws(nuclear_ID=nid, R=r, batch_size=n)
            densities.append(m._validate_external_density_pair(*density, nuclear_ID=nid,
                normalization_rtol=5e-4, negativity_atol=1e-8))
        levels = self._weights.new_zeros((n, len(self.cutoffs), 2, len(self.stored_q)))
        block = max(1, min(12, 12//n))  # bound temporaries on the polar grid
        for y, wy in zip(self.local.y_values, self.local.y_weights):
            branches = [self._source_and_target(nid, *densities[nid-1], y) for nid in (1, 2)]
            profiles = []
            for target, source in ((1, 2), (2, 1)):
                t = branches[target-1][0]
                energies = t.real.new_full((n,), branches[source-1][3])
                profiles.append(m._gamma_amplitude(t, energies, getattr(m, f'Ep_{target}')))
            for start in range(0, len(self.b_nodes), block):
                stop = min(start+block, len(self.b_nodes))
                amplitudes = []
                for target, source in ((1, 2), (2, 1)):
                    _, thick, field, _ = branches[source-1]
                    sign = 1 if target == 1 else -1
                    scalar = profiles[target-1][:, None]*self._survival(thick, source, sign, start, stop)
                    local = field[:, start:stop]*scalar[:, :, None]
                    amplitudes.append(self.fourier.transform(local)*(m._output_dx**2/m.hbar_c))
                radial = self._radial_intensities(*amplitudes, self._b[start:stop])
                levels += wy*torch.einsum('kb,nbsr->nksr', self._weights[:, start:stop], radial)
        if not bool(torch.isfinite(levels).all()) or not bool((levels >= 0).all()):
            raise ArithmeticError('invalid wide-window radial intensity')
        return levels.cpu().numpy()
