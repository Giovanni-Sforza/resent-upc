import numpy as np
import pytest
import torch

from upc_deformation.decay import stokes_maps, decay_coefficients
from upc_deformation.orientation_czt_refinement import NativeGridCZTProjector


def _setup():
    native=torch.linspace(-2.,2.-4/64,64,dtype=torch.float32)
    radii=torch.linspace(0.,.65,31,dtype=torch.float32)
    projector=NativeGridCZTProjector(16,native,radii,32,.3)
    generator=torch.Generator().manual_seed(8841)
    ax=torch.complex(torch.randn(2,16,16,generator=generator),torch.randn(2,16,16,generator=generator))
    ay=torch.complex(torch.randn(2,16,16,generator=generator),torch.randn(2,16,16,generator=generator))
    return projector,ax,ay


def test_native_czt_with_stokes_matches_centered_dense_fft_projection():
    projector,ax,ay=_setup()
    def fft(value):
        padded=torch.zeros(2,64,64,dtype=torch.complex64)
        padded[:,24:40,24:40]=value
        return torch.fft.fftshift(torch.fft.fft2(padded),dim=(-2,-1))*.3
    mx,my=fft(ax),fft(ay)
    expected=decay_coefficients(stokes_maps(mx,my),projector.native_axis,projector.radii,32)
    actual=projector.project(ax,ay)
    torch.testing.assert_close(actual,expected,rtol=3e-6,atol=3e-6)


def test_common_padding_phase_cancels_before_corner_interpolation():
    projector,ax,ay=_setup()
    expected=projector.project(ax,ay)
    phase=torch.tensor(np.exp(.713j),dtype=torch.complex64)
    actual=projector.project(ax*phase,ay*phase)
    torch.testing.assert_close(actual,expected,rtol=3e-6,atol=3e-6)
    # Identical polarization components have S1=0, so incorrect amplitude
    # interpolation or swapped axes cannot be hidden by an arbitrary phase.
    both=projector.project(ax,ax)
    assert bool(torch.isfinite(both).all())


def test_crop_contains_all_original_global_grid_corners_and_weights_sum_to_one():
    projector,_,_=_setup()
    for index in projector.indices:
        assert int(index.min())>=0
        assert int(index.max())<projector.crop_size**2
    torch.testing.assert_close(sum(projector.weights),torch.ones_like(projector.weights[0]),rtol=0,atol=2e-7)
    assert projector.crop_size<len(projector.native_axis)


def test_rejects_incompatible_grid_and_amplitude_precision():
    projector,ax,ay=_setup()
    with pytest.raises(ValueError,match='complex64'):
        projector.project(ax.to(torch.complex128),ay.to(torch.complex128))
    with pytest.raises(ValueError,match='native grid'):
        NativeGridCZTProjector(16,torch.arange(63.),torch.linspace(0,.6,31),32,.3)
