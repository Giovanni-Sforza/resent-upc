"""Selected native complex DFT samples agree with a separately padded FFT."""
import math

import pytest
import torch

from upc_deformation.shape_scan_czt import NativeFFTSubsetCZT


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def dense_reference(field, native_size, vertices):
    spatial = field.shape[-1]
    padded = field.new_zeros((*field.shape[:-2], native_size, native_size))
    start = (native_size-spatial)//2
    padded[..., start:start+spatial, start:start+spatial] = field
    transformed = torch.fft.fftshift(torch.fft.fft2(padded), dim=(-2, -1))
    return transformed.flatten(-2).index_select(-1, vertices)


def random_field(shape, device="cpu"):
    rng = torch.Generator().manual_seed(20261027)
    real = torch.randn(shape, generator=rng, dtype=torch.float64)
    imag = torch.randn(shape, generator=rng, dtype=torch.float64)
    return torch.complex(real, imag).to(device)


@pytest.mark.parametrize("spatial,native,leading", [(10, 14, ()), (16, 64, (2,)), (48, 96, (2, 3))])
def test_irregular_repeated_vertices_and_padding_phase(spatial, native, leading):
    # Asymmetric corners include both sides of fftshift and repeat a vertex;
    # this checks order, x/y axes, padding phase and arbitrary leading axes.
    vertices = torch.tensor([0, native**2-1, native*2+7, native//2*native+native//2,
                             native*(native-3)+1, native*2+7], dtype=torch.int64)
    field = random_field((*leading, spatial, spatial))
    subset = NativeFFTSubsetCZT(spatial, native, vertices)
    actual = subset.transform(field)
    expected = dense_reference(field, native, vertices)
    torch.testing.assert_close(actual, expected, rtol=3e-13, atol=3e-13*float(expected.abs().max()))


def test_production_native_frequencies_without_regridding():
    spatial, native = 320, 1280
    axis = torch.arange(native//2-90, native//2+90)
    xy = torch.cartesian_prod(axis, axis)
    vertices = xy[:, 0]*native + xy[:, 1]
    field = random_field((2, spatial, spatial))
    subset = NativeFFTSubsetCZT(spatial, native, vertices)
    actual = subset.transform(field)
    expected = dense_reference(field, native, vertices)
    assert subset.crop_shape == (180, 180)
    torch.testing.assert_close(actual, expected, rtol=3e-13, atol=3e-13*float(expected.abs().max()))


def test_offset_impulse_has_exact_discrete_frequency_phase():
    spatial, native = 16, 64
    vertices = torch.tensor([0, 1, 64*9+12, 64*32+32, 4095], dtype=torch.int64)
    field = torch.zeros((spatial, spatial), dtype=torch.complex128)
    field[3, 11] = 1+2j
    actual = NativeFFTSubsetCZT(spatial, native, vertices).transform(field)
    frequency_x = torch.div(vertices, native, rounding_mode="floor")-native//2
    frequency_y = vertices.remainder(native)-native//2
    start = (native-spatial)//2
    phase_index = (frequency_x*(start+3)+frequency_y*(start+11)).to(torch.float64)
    angle = -2*math.pi*phase_index/native
    expected = (1+2j)*torch.exp(1j*angle)
    torch.testing.assert_close(actual, expected, rtol=5e-13, atol=5e-13)


def test_two_source_destructive_interference_uses_common_phase():
    spatial, native = 16, 64
    axis = torch.arange(22, 43)
    xy = torch.cartesian_prod(axis, axis)
    vertices = xy[:, 0]*native + xy[:, 1]
    first = random_field((2, spatial, spatial))
    second = -first + 1e-8*first.flip(-1)
    subset = NativeFFTSubsetCZT(spatial, native, vertices)
    a, b = subset.transform(first), subset.transform(second)
    x, y = dense_reference(first, native, vertices), dense_reference(second, native, vertices)
    # Absolute error is scaled to the self intensity, so a near-zero full
    # branch is not divided by a cancellation residue.
    self_scale = float((x.abs().square()+y.abs().square()).max())
    torch.testing.assert_close((a+b).abs().square(), (x+y).abs().square(),
                               rtol=1e-10, atol=1e-13*self_scale)
    torch.testing.assert_close(a.abs().square()+b.abs().square(),
                               x.abs().square()+y.abs().square(),
                               rtol=3e-13, atol=3e-13*self_scale)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable on this host")
def test_cuda_complex128_subset_matches_cuda_dense():
    spatial, native = 48, 192
    axis = torch.arange(70, 123, device="cuda")
    xy = torch.cartesian_prod(axis, axis)
    vertices = xy[:, 0]*native+xy[:, 1]
    field = random_field((2, spatial, spatial), "cuda")
    subset = NativeFFTSubsetCZT(spatial, native, vertices, device="cuda")
    actual, expected = subset.transform(field), dense_reference(field, native, vertices)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12*float(expected.abs().max()))


def test_rejects_invalid_geometry_precision_and_fields():
    vertices = torch.tensor([0, 63])
    with pytest.raises(ValueError, match="native_size"):
        NativeFFTSubsetCZT(16, 15, vertices)
    with pytest.raises(ValueError, match="float64"):
        NativeFFTSubsetCZT(16, 64, vertices, dtype=torch.float32)
    with pytest.raises(ValueError, match="integer"):
        NativeFFTSubsetCZT(16, 64, vertices.to(torch.float64))
    with pytest.raises(ValueError, match="integer"):
        NativeFFTSubsetCZT(16, 64, torch.tensor([4096]))
    subset = NativeFFTSubsetCZT(16, 64, vertices)
    with pytest.raises(ValueError, match="complex128"):
        subset.transform(torch.zeros(16, 16, dtype=torch.complex64))
    with pytest.raises(ValueError, match="spatial grid"):
        subset.transform(torch.zeros(17, 17, dtype=torch.complex128))
