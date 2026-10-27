"""Offset rectangular support transforms versus independent full padded FFTs."""
import math

import pytest
import torch

from upc_deformation.shape_scan_crop_czt import NativeFFTCropSubsetCZT


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def random_patch(shape, seed=20261027, device="cpu"):
    rng = torch.Generator().manual_seed(seed)
    real = torch.randn(shape, generator=rng, dtype=torch.float64)
    imag = torch.randn(shape, generator=rng, dtype=torch.float64)
    return torch.complex(real, imag).to(device)


def dense_reference(patch, spatial, native, offset, vertices):
    # Two explicit placements keep this independent of the CZT translation
    # phase implementation: patch -> output grid -> centered padded grid.
    output = patch.new_zeros((*patch.shape[:-2], spatial, spatial))
    height, width = patch.shape[-2:]
    x, y = offset
    output[..., x:x+height, y:y+width] = patch
    padded = patch.new_zeros((*patch.shape[:-2], native, native))
    start = (native-spatial)//2
    padded[..., start:start+spatial, start:start+spatial] = output
    return torch.fft.fftshift(torch.fft.fft2(padded), dim=(-2, -1)).flatten(-2).index_select(-1, vertices)


@pytest.mark.parametrize("shape,offset,leading", [
    ((3, 7), (0, 9), ()), ((7, 3), (9, 0), (2,)),
    ((1, 1), (15, 15), (2, 3)), ((16, 16), (0, 0), (1,)),
])
def test_asymmetric_edge_patches_repeated_vertices_and_leading_axes(shape, offset, leading):
    spatial, native = 16, 64
    vertices = torch.tensor([0, 4095, 64*3+9, 64*32+32, 64*55+1, 64*3+9], dtype=torch.int64)
    patch = random_patch((*leading, *shape))
    transform = NativeFFTCropSubsetCZT(spatial, native, vertices, shape, offset)
    actual = transform.transform(patch)
    expected = dense_reference(patch, spatial, native, offset, vertices)
    torch.testing.assert_close(actual, expected, rtol=3e-13, atol=3e-13*float(expected.abs().max()))


@pytest.mark.parametrize("height,offset", [(48, (104, 136)), (49, (167, 136))])
def test_production_support_and_original_177_frequency_rectangle(height, offset):
    spatial, native = 320, 1280
    frequency_axis = torch.arange(native//2-88, native//2+89)
    xy = torch.cartesian_prod(frequency_axis, frequency_axis)
    vertices = xy[:, 0]*native+xy[:, 1]
    patch = random_patch((2, height, 48))
    transform = NativeFFTCropSubsetCZT(spatial, native, vertices, (height, 48), offset)
    actual = transform.transform(patch)
    expected = dense_reference(patch, spatial, native, offset, vertices)
    assert transform.frequency_crop_shape == (177, 177)
    assert transform._x_transform.nfft == transform._y_transform.nfft == 256
    torch.testing.assert_close(actual, expected, rtol=3e-13, atol=3e-13*float(expected.abs().max()))


def test_distinct_source_offsets_preserve_coherent_interference():
    spatial, native = 32, 128
    frequency_axis = torch.arange(45, 84)
    xy = torch.cartesian_prod(frequency_axis, frequency_axis)
    vertices = xy[:, 0]*native+xy[:, 1]
    offsets = ((3, 12), (21, 8))
    first = random_patch((2, 7, 9), seed=19)
    second = .8*first*torch.exp(torch.tensor(.41j, dtype=torch.complex128))
    sources, expected = [], []
    for patch, offset in zip((first, second), offsets):
        transform = NativeFFTCropSubsetCZT(spatial, native, vertices, (7, 9), offset)
        sources.append(transform.transform(patch))
        expected.append(dense_reference(patch, spatial, native, offset, vertices))
    for actual, reference in zip(sources, expected):
        torch.testing.assert_close(actual, reference, rtol=3e-13,
                                   atol=3e-13*float(reference.abs().max()))
    reference_full = (expected[0]+expected[1]).abs().square()
    actual_full = (sources[0]+sources[1]).abs().square()
    torch.testing.assert_close(actual_full, reference_full, rtol=3e-13,
                               atol=3e-13*float(reference_full.max()))
    # An implementation that discarded each source's distinct translation
    # phase would incorrectly predict a constant coherent enhancement here.
    wrong = ((1+.8*torch.exp(torch.tensor(.41j)))*sources[0]).abs().square()
    assert not torch.allclose(actual_full, wrong, rtol=.01, atol=.01)


def test_offset_impulse_with_uneven_original_padding_has_correct_phase():
    spatial, native = 15, 64
    offset, shape = (8, 3), (5, 8)
    vertices = torch.tensor([0, 1, 64*7+19, 64*32+32, 4095], dtype=torch.int64)
    patch = torch.zeros(shape, dtype=torch.complex128)
    patch[2, 6] = 1+2j
    actual = NativeFFTCropSubsetCZT(spatial, native, vertices, shape, offset).transform(patch)
    frequency_x = torch.div(vertices, native, rounding_mode="floor")-native//2
    frequency_y = vertices.remainder(native)-native//2
    start = (native-spatial)//2
    index = (frequency_x*(start+offset[0]+2)+frequency_y*(start+offset[1]+6)).to(torch.float64)
    expected = (1+2j)*torch.exp(-2j*math.pi*index/native)
    torch.testing.assert_close(actual, expected, rtol=5e-13, atol=5e-13)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable on this host")
def test_cuda_rectangular_patch_matches_cuda_full_fft():
    spatial, native = 64, 256
    xy = torch.cartesian_prod(torch.arange(100, 157, device="cuda"),
                             torch.arange(107, 151, device="cuda"))
    vertices = xy[:, 0]*native+xy[:, 1]
    patch = random_patch((2, 13, 12), device="cuda")
    offset = (31, 20)
    transform = NativeFFTCropSubsetCZT(spatial, native, vertices, (13, 12), offset, device="cuda")
    actual = transform.transform(patch)
    expected = dense_reference(patch, spatial, native, offset, vertices)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12*float(expected.abs().max()))


@pytest.mark.parametrize("shape,offset", [
    ((0, 4), (0, 0)), ((4, 4), (-1, 0)), ((17, 2), (0, 0)),
    ((4, 4), (13, 0)), ((4, 4), (0, 13)), ((4., 4), (0, 0)),
    ((4, 4), (True, 0)), ((4,), (0, 0)),
])
def test_invalid_crop_rejected(shape, offset):
    with pytest.raises(ValueError, match="crop"):
        NativeFFTCropSubsetCZT(16, 64, torch.tensor([0]), shape, offset)


def test_invalid_fft_indices_precision_and_field_rejected():
    vertices = torch.tensor([0, 63])
    with pytest.raises(ValueError, match="native_size"):
        NativeFFTCropSubsetCZT(16, 15, vertices, (4, 4), (1, 2))
    with pytest.raises(ValueError, match="float64"):
        NativeFFTCropSubsetCZT(16, 64, vertices, (4, 4), (1, 2), dtype=torch.float32)
    for bad in (torch.tensor([]), torch.tensor([-1]), torch.tensor([4096]), vertices.double()):
        with pytest.raises(ValueError, match="integer"):
            NativeFFTCropSubsetCZT(16, 64, bad, (4, 4), (1, 2))
    transform = NativeFFTCropSubsetCZT(16, 64, vertices, (4, 5), (1, 2))
    for bad in (torch.zeros((4, 5), dtype=torch.complex64), torch.zeros((5, 4), dtype=torch.complex128)):
        with pytest.raises(ValueError, match="crop_shape"):
            transform.transform(bad)
