"""Free-space transverse photon field on patches centered at x=±b.

For the model's positive-sign proton form factor and negative-sign transverse
Fourier transform, a unit point charge at s contributes
    -i/(2 pi) kappa K1(kappa |x-s|) (x-s)/|x-s|.
The input is the transverse array of complex charge *weights* (including the
longitudinal phase exp(+i kappa z) and the voxel volume, normalized by total
proton charge). The caller supplies the model's charge/energy prefactor.

On the source lattice we sum this free-space kernel against every source
voxel. The kernel at the origin vanishes by odd symmetry over the central
cell. Between lattice points we bilinearly interpolate the two components.
These are finite spatial discretizations; varying lattice spacing tests them.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.fft import next_fast_len
import torch


def _axis(values, label):
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    axis = np.asarray(values, dtype=np.float64)
    if axis.ndim != 1 or not len(axis) or not np.isfinite(axis).all():
        raise ValueError(f"{label} must be a finite, nonempty vector")
    return axis


class FreePhotonStrip:
    """Linear convolution on the short strips needed by the requested b nodes.

    Coordinates, source spacing and b are in fm, kappa in inverse fm. Both
    physical axes of a source or a patch use the supplied one-dimensional
    coordinate vector in ij indexing. Each field has shape (batch,b,2,x,y).
    """

    def __init__(self, source_axis, kappa, b_values, patch_axis, device="cpu"):
        source = _axis(source_axis, "source_axis")
        if len(source) < 2:
            raise ValueError("at least two source sites required")
        self.h = float(source[1] - source[0])
        if self.h <= 0 or not np.allclose(np.diff(source), self.h, rtol=1e-12, atol=0):
            raise ValueError("source_axis must increase uniformly")
        self.origin, self.n = float(source[0]), len(source)
        self.b = _axis(b_values, "b_values")
        if np.any(self.b < 0) or np.any(np.diff(self.b) <= 0):
            raise ValueError("b_values must be increasing and nonnegative")
        self.patch = _axis(patch_axis, "patch_axis")
        if np.any(np.diff(self.patch) <= 0):
            raise ValueError("patch_axis must increase")
        self.kappa = float(kappa)
        if not math.isfinite(self.kappa) or self.kappa <= 0:
            raise ValueError("kappa must be finite and positive")
        self.device = torch.device(device)

    def _component_kernels(self, x0, x1, y0, y1):
        # These difference indices suffice for *every* contributing source
        # voxel. The circular FFT equals linear convolution on the retained
        # output rectangle [N-1:N-1+width, N-1:N-1+height].
        x = torch.arange(x0-self.n+1, x1+1, dtype=torch.float64, device=self.device) * self.h
        y = torch.arange(y0-self.n+1, y1+1, dtype=torch.float64, device=self.device) * self.h
        r = torch.hypot(x[:, None], y[None, :])
        r = torch.where(r == 0, 1.0, r)
        radial = self.kappa * torch.special.modified_bessel_k1(self.kappa * r) / r
        return torch.stack((x[:, None]*radial, y[None, :]*radial))

    def _interpolation_indices(self, coordinates):
        fractional = (coordinates-self.origin)/self.h
        base = np.floor(fractional).astype(np.int64)
        indices = torch.as_tensor(base-base.min()+self.n-1, device=self.device)
        weight = torch.as_tensor(fractional-base, device=self.device)
        return int(base.min()), int(base.max())+1, indices, weight

    def iter_samples(self, projected_charge, sign):
        """Yield (b_slice, complex vector field) with bounded working memory."""
        if sign not in (-1, 1):
            raise ValueError("source displacement sign must be +1 or -1")
        charge = torch.as_tensor(projected_charge, dtype=torch.complex128, device=self.device)
        if (charge.ndim != 3 or not charge.shape[0] or charge.shape[-2:] != (self.n, self.n)
                or not bool(torch.isfinite(charge).all())):
            raise ValueError("projected_charge must have finite shape (batch, source_x, source_y)")
        y0, y1, iy, wy = self._interpolation_indices(self.patch)
        # Limit variation within a strip to preserve relative tail accuracy and
        # exclude empty space between distant groups of sampled b values.
        span = min(self.n*self.h*2, 6/self.kappa)
        start = 0
        while start < len(self.b):
            stop = min(int(np.searchsorted(self.b, self.b[start]+span, side="right")), start+32)
            selection = slice(start, stop)
            coordinates = sign*self.b[selection, None]+self.patch[None, :]
            x0, x1, ix, wx = self._interpolation_indices(coordinates)
            kernels = self._component_kernels(x0, x1, y0, y1)
            fft_shape = tuple(next_fast_len(s) for s in kernels.shape[-2:])
            spectra = torch.fft.fft2(kernels, s=fft_shape)
            sources = torch.fft.fft2(charge, s=fft_shape)
            convolved = torch.fft.ifft2(sources[:, None]*spectra)
            # Interpolate y first. Only the requested x sites (two per
            # query) and y sites enter the result of the linear convolution.
            line = convolved.index_select(-1, iy)*(1-wy) + convolved.index_select(-1, iy+1)*wy
            value = line[:, :, ix, :]*(1-wx[..., None]) + line[:, :, ix+1, :]*wx[..., None]
            yield selection, value.transpose(1, 2)*(-1j/(2*math.pi))
            start = stop

    def sample(self, projected_charge, sign):
        """Return all requested patches in b order, keeping the batch axis."""
        return torch.cat([field for _, field in self.iter_samples(projected_charge, sign)], dim=1)
