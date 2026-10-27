"""Original native DFT samples from a rectangular spatial support patch.

The patch occupies its registered integer offset in the original output grid;
that grid retains the original centered zero padding.  Both translation phases
are included, independently for x and y.  This is a finite-DFT implementation,
not a spatial-tail cutoff or an interpolation of complex amplitudes.
"""
from __future__ import annotations

import torch

from .shape_scan_czt import _NativeAxisCZT, _unit_phase


def _integer_pair(value, name, minimum):
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(type(item) is not int or item < minimum for item in value)):
        raise ValueError(f"{name} must contain two integers >= {minimum}")
    return tuple(value)


class NativeFFTCropSubsetCZT:
    """Evaluate native ``fftshift`` vertices for an offset complex128 patch.

    ``crop_shape=(height,width)`` follows the physical x/y axes of the square
    ``spatial_size`` output grid. ``crop_start`` is its zero-based integer
    offset in that grid. The result is identical in exact arithmetic to putting
    the patch into an otherwise-zero output grid, center-padding to
    ``native_size``, evaluating the full 2D DFT and selecting
    ``shifted_flat_indices`` in their original order.

    ``transform`` accepts arbitrary leading dimensions followed by crop_shape
    and returns those leading dimensions followed by the vertex dimension.
    """

    def __init__(self, spatial_size, native_size, shifted_flat_indices,
                 crop_shape, crop_start, device="cpu", dtype=torch.float64):
        if (type(spatial_size) is not int or type(native_size) is not int
                or spatial_size < 2 or native_size < spatial_size or native_size % 2):
            raise ValueError("require spatial_size >= 2 and an even native_size >= spatial_size")
        if dtype not in (torch.float64, "float64"):
            raise ValueError("the validated crop CZT requires float64/complex128")
        self.spatial_size, self.native_size = spatial_size, native_size
        self.crop_shape = _integer_pair(crop_shape, "crop_shape", 1)
        self.crop_start = _integer_pair(crop_start, "crop_start", 0)
        if any(start+count > spatial_size for start, count in zip(self.crop_start, self.crop_shape)):
            raise ValueError("crop must fit entirely inside the original spatial grid")
        self.device = torch.device(device)
        indices = torch.as_tensor(shifted_flat_indices, device=self.device)
        if (indices.ndim != 1 or not len(indices)
                or indices.dtype not in (torch.int32, torch.int64)
                or bool(torch.any(indices < 0)) or bool(torch.any(indices >= native_size**2))):
            raise ValueError("nonempty one-dimensional integer native FFT indices required")
        indices = indices.to(torch.int64)
        x, y = torch.div(indices, native_size, rounding_mode="floor"), indices.remainder(native_size)
        self.x_start, self.y_start = int(x.min()), int(y.min())
        self.x_count = int(x.max())-self.x_start+1
        self.y_count = int(y.max())-self.y_start+1
        self.frequency_crop_shape = (self.x_count, self.y_count)
        self._selected = (x-self.x_start)*self.y_count+y-self.y_start
        self._x_transform = _NativeAxisCZT(self.crop_shape[0], native_size,
            self.x_start-native_size//2, self.x_count, self.device)
        self._y_transform = (self._x_transform
            if (self.crop_shape[0] == self.crop_shape[1] and self.x_start == self.y_start
                and self.x_count == self.y_count)
            else _NativeAxisCZT(self.crop_shape[1], native_size,
                self.y_start-native_size//2, self.y_count, self.device))
        output_padding = (native_size-spatial_size)//2
        origin_x, origin_y = (output_padding+start for start in self.crop_start)
        self._translation_phase = _unit_phase(
            -origin_x*(x-native_size//2)-origin_y*(y-native_size//2), native_size)

    def transform(self, field):
        if (field.ndim < 2 or field.shape[-2:] != self.crop_shape
                or field.dtype != torch.complex128
                or field.device != self._translation_phase.device):
            raise ValueError("matching-device complex128 field with the registered crop_shape required")
        along_y = self._y_transform.transform_last_axis(field)
        rectangle = self._x_transform.transform_last_axis(along_y.transpose(-2, -1)).transpose(-2, -1)
        return rectangle.flatten(-2).index_select(-1, self._selected)*self._translation_phase
