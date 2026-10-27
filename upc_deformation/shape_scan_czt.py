"""Torch complex128 transforms at selected *original* padded FFT vertices.

The separable Bluestein transforms evaluate a rectangular subset of the same
finite DFT as ``fftshift(fft2(center_pad(field)))``.  They do not change the
spatial samples or momentum grid, interpolate amplitudes, or omit the phase
from centered padding.  Only the requested native vertices are returned.
"""
from __future__ import annotations

import math

import torch


def _unit_phase(numerator, denominator):
    """exp(2 pi i numerator / denominator), reducing integer phases first."""
    angle = numerator.remainder(denominator).to(torch.float64) * (2 * math.pi / denominator)
    return torch.polar(torch.ones_like(angle), angle)


class _NativeAxisCZT:
    def __init__(self, spatial_size, native_size, first_frequency, count, device):
        self.n, self.m = spatial_size, count
        self.nfft = 1 << (spatial_size + count - 2).bit_length()
        index = torch.arange(max(spatial_size, count), dtype=torch.int64, device=device)
        # w^(j^2/2) with w=exp(-2*pi*i/native_size). Integer modular
        # reduction avoids the accumulation of phase error from powers of w.
        chirp = _unit_phase(-index.square(), 2 * native_size)
        input_phase = _unit_phase(-first_frequency * index[:spatial_size], native_size)
        self.input_factor = input_phase * chirp[:spatial_size]
        convolution = torch.cat((chirp[1:spatial_size].flip(0), chirp[:count])).conj()
        self.kernel_fft = torch.fft.fft(convolution, n=self.nfft)
        self.output_factor = chirp[:count]

    def transform_last_axis(self, field):
        transformed = torch.fft.fft(field * self.input_factor, n=self.nfft, dim=-1)
        convolution = torch.fft.ifft(transformed * self.kernel_fft, dim=-1)
        return convolution[..., self.n-1:self.n+self.m-1] * self.output_factor


class NativeFFTSubsetCZT:
    """Return selected unscaled, centered-padded 2D FFT samples in FP64.

    ``shifted_flat_indices`` are flattened indices into the full N-by-N
    ``fftshift`` result. ``transform(field)`` accepts complex128 arrays with
    arbitrary leading dimensions and trailing ``(spatial_size, spatial_size)``
    and returns those leading dimensions followed by the vertex dimension.
    """

    def __init__(self, spatial_size, native_size, shifted_flat_indices,
                 device="cpu", dtype=torch.float64):
        if (isinstance(spatial_size, bool) or isinstance(native_size, bool)
                or not isinstance(spatial_size, int) or not isinstance(native_size, int)
                or spatial_size < 2 or native_size < spatial_size or native_size % 2):
            raise ValueError("require spatial_size >= 2 and an even native_size >= spatial_size")
        if dtype not in (torch.float64, "float64"):
            raise ValueError("the validated subset CZT requires float64/complex128")
        self.device = torch.device(device)
        self.spatial_size, self.native_size = spatial_size, native_size
        indices = torch.as_tensor(shifted_flat_indices, device=self.device)
        if (indices.ndim != 1 or not len(indices)
                or indices.dtype not in (torch.int32, torch.int64)
                or bool(torch.any(indices < 0)) or bool(torch.any(indices >= native_size**2))):
            raise ValueError("nonempty one-dimensional integer native FFT indices required")
        indices = indices.to(torch.int64)
        x, y = torch.div(indices, native_size, rounding_mode="floor"), indices.remainder(native_size)
        self.x_start, self.y_start = int(x.min()), int(y.min())
        self.x_count = int(x.max()) - self.x_start + 1
        self.y_count = int(y.max()) - self.y_start + 1
        self.crop_shape = (self.x_count, self.y_count)
        self._selected = (x-self.x_start)*self.y_count + y-self.y_start
        self._x_transform = _NativeAxisCZT(spatial_size, native_size,
            self.x_start-native_size//2, self.x_count, self.device)
        self._y_transform = (self._x_transform
            if self.x_start == self.y_start and self.x_count == self.y_count
            else _NativeAxisCZT(spatial_size, native_size,
                self.y_start-native_size//2, self.y_count, self.device))
        pad_start = (native_size-spatial_size)//2
        self._padding_phase = _unit_phase(
            -pad_start * (x+y-native_size), native_size)

    def transform(self, field):
        if (field.ndim < 2 or field.shape[-2:] != (self.spatial_size, self.spatial_size)
                or field.dtype != torch.complex128
                or field.device != self._padding_phase.device):
            raise ValueError("matching-device complex128 field on the registered spatial grid required")
        along_y = self._y_transform.transform_last_axis(field)
        rectangle = self._x_transform.transform_last_axis(along_y.transpose(-2, -1)).transpose(-2, -1)
        return rectangle.flatten(-2).index_select(-1, self._selected) * self._padding_phase
