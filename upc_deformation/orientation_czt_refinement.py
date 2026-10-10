"""Native-frequency CZT implementation of the unchanged panel-d projection.

Only native FFT samples needed by the existing bilinear circle projection are
computed. Spatial amplitudes, Fourier frequencies, Stokes bilinears, original
global interpolation weights and b/y quadratures remain unchanged. SciPy CZT
uses complex128 internally, then amplitudes are cast to inherited complex64
before Stokes formation. This is a numerical implementation with explicit
regression tolerances, not a reduced grid or an amplitude-interpolation model.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from scipy.signal import CZT

from .decay import stokes_maps
from .orientation_refinement import PanelDOrientationRefiner


class NativeGridCZTProjector:
    """Compute the required native DFT crop and its original Stokes projection.

    The centered spatial zero pad contributes one common unit phase to both
    transverse amplitudes. Every S0/S1/S2 bilinear cancels that phase, so it need
    not be evaluated. Stokes are formed at DFT grid corners before interpolation.
    """
    def __init__(self, spatial_size, native_axis, radii, n_phi, normalization):
        self.spatial_size = int(spatial_size)
        self.native_axis = torch.as_tensor(native_axis, dtype=torch.float32, device='cpu')
        self.radii = torch.as_tensor(radii, dtype=torch.float32, device='cpu')
        self.n_phi, self.normalization = int(n_phi), float(normalization)
        n = len(self.native_axis)
        if (n < self.spatial_size or n % 2 or self.spatial_size < 2
                or self.native_axis.ndim != 1 or not bool(torch.isfinite(self.native_axis).all())
                or not bool(torch.all(self.native_axis[1:] > self.native_axis[:-1]))):
            raise ValueError('a finite increasing even native grid covering the spatial grid is required')
        if (self.radii.ndim != 1 or len(self.radii) < 2 or self.radii[0] < 0
                or not bool(torch.isfinite(self.radii).all())
                or not bool(torch.all(self.radii[1:] > self.radii[:-1]))
                or n_phi < 8 or n_phi % 4 or not math.isfinite(self.normalization) or self.normalization <= 0):
            raise ValueError('finite increasing radii, n_phi divisible by four, and positive normalization are required')
        phi = torch.arange(n_phi, dtype=torch.float32)*(2*math.pi/n_phi)
        coords = torch.stack((self.radii[:, None]*phi.cos(), self.radii[:, None]*phi.sin()), -1).reshape(-1, 2)
        # Preserve the full-grid float32 coordinate arithmetic in _interp2d_batched
        # and grid_sample(align_corners=True), rather than renormalizing a crop.
        lower, upper = float(self.native_axis[0]), float(self.native_axis[-1])
        gx = 2*(coords[:, 0]-lower)/(upper-lower)-1
        gy = 2*(coords[:, 1]-lower)/(upper-lower)-1
        ix, iy = ((gx+1)/2)*(n-1), ((gy+1)/2)*(n-1)
        ix0, iy0 = torch.floor(ix).long(), torch.floor(iy).long()
        if min(int(ix0.min()), int(iy0.min())) < 0 or max(int(ix0.max()), int(iy0.max()))+1 >= n:
            raise ValueError('all bilinear query corners must lie strictly inside the native DFT grid')
        self.lo = max(0, int(torch.floor(torch.minimum(ix.min(), iy.min())))-1)
        self.hi = min(n, int(torch.ceil(torch.maximum(ix.max(), iy.max())))+2)
        self.crop_size = self.hi-self.lo
        dx, dy = ix-ix0, iy-iy0
        side = self.crop_size
        self.indices = ((ix0-self.lo)*side+iy0-self.lo,
                        (ix0-self.lo)*side+iy0+1-self.lo,
                        (ix0+1-self.lo)*side+iy0-self.lo,
                        (ix0+1-self.lo)*side+iy0+1-self.lo)
        self.weights = ((1-dx)*(1-dy), (1-dx)*dy, dx*(1-dy), dx*dy)
        self.sin2, self.cos2 = (2*phi).sin(), (2*phi).cos()
        self.transform = CZT(n=self.spatial_size, m=side,
            w=np.exp(-2j*np.pi/n), a=np.exp(2j*np.pi*(self.lo-n//2)/n))

    def project(self, ax, ay):
        if (ax.shape != ay.shape or ax.shape[-2:] != (self.spatial_size, self.spatial_size)
                or ax.dtype != torch.complex64 or ay.dtype != torch.complex64
                or ax.device.type != 'cpu' or ay.device.type != 'cpu'):
            raise ValueError('matching CPU complex64 spatial amplitudes are required')
        spatial = np.stack((ax.detach().numpy(), ay.detach().numpy()))
        transformed = self.transform(self.transform(spatial, axis=-1), axis=-2)*self.normalization
        mx, my = torch.from_numpy(transformed.astype(np.complex64)).unbind(0)
        stokes = stokes_maps(mx, my).reshape(*ax.shape[:-2], 3, self.crop_size**2)
        i0, i1, i2, i3 = self.indices
        w0, w1, w2, w3 = self.weights
        sampled = stokes[..., i0]*w0+stokes[..., i1]*w1+stokes[..., i2]*w2+stokes[..., i3]*w3
        s0, s1, s2 = sampled.reshape(*ax.shape[:-2], 3, len(self.radii), self.n_phi).unbind(-3)
        return torch.stack((s0.mean(-1),
            (s1*self.cos2+s2*self.sin2).mean(-1),
            (s1*self.sin2-s2*self.cos2).mean(-1)), -2)


class PanelDOrientationCZTRefiner(PanelDOrientationRefiner):
    """Separate validated candidate; leaves the dense production kernel intact."""
    def __init__(self, config, case):
        super().__init__(config, case, fft_mode='combined')
        m = self.model
        self.projector = NativeGridCZTProjector(m._output_N, m._p_full_q,
            self.radii, self.n_phi, m._output_dx**2/m.hbar_c)

    def _node(self, p):
        m = self.model
        ax1, ay1 = m._compute_amplitude_at_grid(nuclear_ID=1,
            T_prim_target=p['t1'], T_thick_source=p['thick2'],
            Vx_source=p['vx2'], Vy_source=p['vy2'], Egamma_target=p['energy2'],
            Ep_target=m.Ep_1, A_source=m.A_2)
        ax2, ay2 = m._compute_amplitude_at_grid(nuclear_ID=2,
            T_prim_target=p['t2'], T_thick_source=p['thick1'],
            Vx_source=p['vx1'], Vy_source=p['vy1'], Egamma_target=p['energy1'],
            Ep_target=m.Ep_2, A_source=m.A_1)
        return self.projector.project(ax1+ax2, ay1+ay2)

    def calculate(self, R1, R2, *, return_nodes=True):
        result = super().calculate(R1, R2, return_nodes=return_nodes)
        result.update(fourier_backend='native_grid_czt',
            native_dft_side=len(self.projector.native_axis),
            czt_crop_bounds=[self.projector.lo, self.projector.hi],
            czt_crop_side=self.projector.crop_size)
        return result
