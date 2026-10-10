"""Physical normalization and independent angular-moment checks for kT."""
import numpy as np
import pytest
from upc_deformation.kt import kt_bin_density, matrix_element_weight


@pytest.mark.parametrize('stokes',[[1.,1.,0.],[1.,0.,0.],[2.,.8,1.2]])
def test_pixel_distribution_matches_independent_sphere_integrals(stokes):
    mass=3.097;k0=mass/2
    # Use a finer grid here because moments replace each finite bin by its
    # centre; the displayed 201-bin density itself is already area-integrated.
    axis,edges,density=kt_bin_density(stokes,mass,bins=401)
    prob=density*(edges[1]-edges[0])**2
    x,y=np.meshgrid(axis,axis,indexing='ij')
    # Integrate the original lepton tensor directly on the sphere, without
    # Cartesian Jacobians or the pixel integration code under test.
    u,w=np.polynomial.legendre.leggauss(64)
    phi=np.arange(512)*2*np.pi/512
    nx=np.sqrt(1-u[:,None]**2)*np.cos(phi)
    ny=np.sqrt(1-u[:,None]**2)*np.sin(phi)
    s0,s1,s2=stokes
    tensor=np.array([[s0+s1,s2],[s2,s0-s1]])/(2*s0)
    weight=3/(8*np.pi)*(1-tensor[0,0]*nx*nx-tensor[1,1]*ny*ny-2*tensor[0,1]*nx*ny)
    measure=w[:,None]*(2*np.pi/512)*weight
    np.testing.assert_allclose(prob.sum(),1,atol=1e-10)
    for cartesian,spherical in [(x*x,k0*k0*nx*nx),(y*y,k0*k0*ny*ny),(x*y,k0*k0*nx*ny)]:
        np.testing.assert_allclose((prob*cartesian).sum(),(measure*spherical).sum(),atol=3e-5)
    angle=np.arctan2(y,x)
    np.testing.assert_allclose((prob*np.cos(2*angle)).sum(),-s1/(4*s0),atol=5e-5)
    np.testing.assert_allclose((prob*np.sin(2*angle)).sum(),-s2/(4*s0),atol=5e-5)
    assert prob.min()>=0
    # A bin fully outside the on-shell circle carries no probability.
    assert prob[0,0]==0


def test_old_weight_vertical_bands_do_not_imply_a_sharp_axis():
    axis=np.linspace(-1,1,11)
    weight=matrix_element_weight([1.,1.,0.],axis,2.)
    np.testing.assert_allclose(weight,np.broadcast_to((1-axis**2)[:,None],weight.shape),atol=1e-15)
    # Despite vertical contours, a fully x-polarized state has only -1/4
    # second-harmonic angular correlation after full polar-angle integration.
    from upc_deformation.kt import azimuth_density
    psi=np.arange(1024)*2*np.pi/1024
    p=azimuth_density([1.,1.,0.],psi)
    np.testing.assert_allclose((p*np.cos(2*psi)).sum()*2*np.pi/1024,-.25,atol=1e-14)


def test_rejects_nonphysical_polarization():
    with pytest.raises(ValueError):kt_bin_density([1.,1.1,0.],3.097)
