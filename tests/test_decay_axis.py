"""Independent angular-integral, frame and probability checks."""
import math
import numpy as np
import pytest
import torch
import yaml
from pathlib import Path
from upc_deformation.decay import (stokes_maps, lepton_azimuth_density,
    decay_coefficients, decay_axis_image, b_quadrature, polar_stokes)


def test_polar_integration_matches_spin_one_lepton_tensor():
    # Independent Gauss integration of (delta_ij - n_i n_j), incl. normalization.
    u,w=np.polynomial.legendre.leggauss(32)
    tensor=np.array([[2.,.4],[.4,1.]])
    psi=np.linspace(0,2*np.pi,39)
    numerical=[]
    for p in psi:
        n=np.sqrt(1-u*u)[:,None]*np.array([np.cos(p),np.sin(p)])
        values=3/(8*np.pi)*(np.trace(tensor)-np.einsum('ni,ij,nj->n',n,tensor,n))
        numerical.append(w@values)
    got=lepton_azimuth_density(torch.tensor([3.,1.,.8]),torch.tensor(psi))
    np.testing.assert_allclose(got,numerical,rtol=1e-7,atol=1e-12)


def test_complex_cross_term_and_global_phase():
    m=torch.tensor([[1+2j,2-1j],[3+0j,1j]])
    n=torch.tensor([[2-1j,1+1j],[1j,2+0j]])
    s=stokes_maps(m,n)
    torch.testing.assert_close(s[2],2*(m*n.conj()).real/(2*math.pi))
    phase=torch.exp(torch.tensor(.7j))
    torch.testing.assert_close(stokes_maps(m*phase,n*phase),s)
    assert bool(torch.all(s[0]**2+1e-5 >= s[1]**2+s[2]**2))


def toy():
    axis=torch.linspace(-1,1,401,dtype=torch.float64)
    x,y=torch.meshgrid(axis,axis,indexing='ij')
    intensity=x*x*torch.exp(-4*(x*x+y*y))
    s=torch.stack((intensity,intensity,torch.zeros_like(intensity)))
    radii=torch.linspace(0,.9,101,dtype=torch.float64)
    return axis,s,radii


def test_quadratic_two_lobes_have_exact_one_quarter_modulation():
    axis,s,r=toy()
    a0,c,d=decay_coefficients(s,axis,r,512)
    keep=r>.05
    expected=.5*r*r*torch.exp(-4*r*r)
    torch.testing.assert_close(a0[keep],expected[keep],rtol=.007,atol=1e-7)
    torch.testing.assert_close(c[keep],expected[keep]/2,rtol=.007,atol=1e-7)
    assert float(d.abs().max())<1e-14
    # I=A0[1-.25*cos(2alpha)]; observed v2 = -.125 versus true v2=.5.


def test_direct_azimuth_integral_matches_compressed_coefficients():
    axis,s,r=toy()
    # Nonzero cross term exercises the sin term and frame convention.
    s[1]=.6*s[0];s[2]=.7*s[0]
    samples,phi=polar_stokes(s,axis,r,512)
    coeff=decay_coefficients(s,axis,r,512)
    for alpha in [0.,.3,1.1,2.2]:
        psi=phi-alpha
        direct=(samples[0]-.5*(samples[1]*torch.cos(2*psi)+samples[2]*torch.sin(2*psi))).mean(-1)
        compressed=coeff[0]-.5*(coeff[1]*math.cos(2*alpha)+coeff[2]*math.sin(2*alpha))
        torch.testing.assert_close(direct,compressed,rtol=1e-12,atol=1e-14)


def test_axis_rotation_and_angular_yield_conservation():
    axis,s,r=toy()
    coeff=decay_coefficients(s,axis,r,256)
    out=torch.linspace(-.6,.6,121,dtype=torch.float64)
    lepton=decay_axis_image(coeff,r,out)
    perp=decay_axis_image(coeff,r,out,'perpendicular')
    torch.testing.assert_close(perp,torch.rot90(lepton,1,(-2,-1)),rtol=1e-12,atol=1e-14)
    alpha=torch.arange(128,dtype=torch.float64)*2*math.pi/128
    density=coeff[0,:,None]-.5*(coeff[1,:,None]*torch.cos(2*alpha)+coeff[2,:,None]*torch.sin(2*alpha))
    torch.testing.assert_close(density.mean(-1),coeff[0])
    assert bool(torch.all(density>=0))


def test_b_quadrature_is_incoherent_and_has_correct_area():
    b=torch.tensor([12.,13.,16.,20.],dtype=torch.float64)
    weights=b_quadrature(b)
    torch.testing.assert_close(weights.sum(),torch.tensor(math.pi*(20**2-12**2),dtype=b.dtype))
    with pytest.raises(ValueError):b_quadrature(b.flip(0))
    # Opposite phases on distinct b nodes do not cancel their intensities.
    amplitudes=torch.tensor([1+0j,-1+0j])
    assert float((amplitudes.abs()**2).mean())==1
    assert float(amplitudes.mean().abs()**2)==0


def test_longitudinal_phase_uses_dimensionless_qz():
    from upc_deformation.forward import UPCForwardModel
    path=Path(__file__).resolve().parents[1]/'configs/jpsi_decay_axis.yaml'
    cfg=yaml.safe_load(path.read_text())['physics']
    cfg['grid_parameters']={'WS':{'N':16,'L':8.},'T_prim':{'N':16,'L':8.},'output':{'N':32,'L':32.}}
    cfg['output_settings'].update(safe_padding_factor=2,target_size_q=33)
    model=UPCForwardModel(cfg,'physical')
    density=torch.zeros(1,16,16,16)
    density[0,8,8,6]=density[0,8,8,10]=1 # two sheets at z=+-1 fm
    q_gev=torch.tensor([.1])
    phase,thickness=model._compute_T_prim_and_thickness(density,q_gev)
    ratio=phase[0,8,8].real/thickness[0,8,8]
    torch.testing.assert_close(ratio,torch.cos(q_gev[0]/model.hbar_c))


def test_upstream_replay_is_exact_for_torch_forward():
    from upc_deformation.forward import UPCForwardModel
    from upc_deformation.vendor.upc_forward import UPCForwardModel as Imported
    path=Path(__file__).resolve().parents[1]/'configs/jpsi_decay_axis.yaml'
    cfg=yaml.safe_load(path.read_text())['physics']
    cfg['grid_parameters']={'WS':{'N':16,'L':24.},'T_prim':{'N':16,'L':24.},'output':{'N':32,'L':64.}}
    cfg['output_settings'].update(safe_padding_factor=2,target_size_q=33)
    expected,_=Imported(cfg).forward_multi_y(None,y_views=torch.zeros(1,1))
    actual,_=UPCForwardModel(cfg,'upstream').forward_multi_y(None,y_views=torch.zeros(1,1))
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
