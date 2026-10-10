import copy
from pathlib import Path
import numpy as np
import torch
import yaml
from upc_deformation.forward import UPCForwardModel
from upc_deformation.structure_noise import neutron_radius_for_skin,ws_rms,masked_ratio,summarize_shapes


def test_skin_type_changes_radius_but_keeps_diffuseness():
    rp,ap,skin=5.02,.46,.20
    rn=neutron_radius_for_skin(rp,ap,ap,skin)
    assert rn>rp
    np.testing.assert_allclose(ws_rms(rn,ap)-ws_rms(rp,ap),skin,atol=1e-10)
    assert abs((rn-rp)-skin)>.01 # RMS skin is not the WS parameter difference.


def test_empirical_yield_reproduces_source_incoherent_switch():
    config=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/jpsi_structure_noise.yaml').read_text())['physics']
    config['grid_parameters']={'WS':{'N':24,'L':24.},'T_prim':{'N':24,'L':24.},'output':{'N':64,'L':96.}}
    config['output_settings'].update(safe_padding_factor=2,target_size_q=65)
    config['incoherent_quadrature'].update(n_r=32,n_phi=32)
    config['nuclear_parameters'].update(beta2=.3,beta3=.2,R_ws_n=5.3)
    model=UPCForwardModel(config,'physical')
    initial_b=model.b_imp
    g=torch.Generator().manual_seed(42)
    r1=model._sample_uniform_rotations(1,2,torch.device('cpu'),torch.float32,g)[0]
    r2=model._sample_uniform_rotations(1,2,torch.device('cpu'),torch.float32,g)[0]
    ys=torch.tensor([[0.,.4],[0.,-.4]])
    with torch.inference_mode():
        prob=model.fixed_b_empirical_incoherent_probability_multi_y(ys,[12.,20.],R_1=r1,R_2=r2)
        assert model.b_imp==initial_b
        x,y=torch.meshgrid(model._p_axis_q,model._p_axis_q,indexing='ij')
        envelope=model.b_slope_incoh/np.pi*torch.exp(-model.b_slope_incoh*(x*x+y*y))
        for bi,b in enumerate([12.,20.]):
            model.b_imp=b
            a,_=model.forward_multi_y(None,y_views=ys,R_1=r1,R_2=r2,incoherent=False)
            c,_=model.forward_multi_y(None,y_views=ys,R_1=r1,R_2=r2,incoherent=True)
            target=prob[:,bi,:,None,None]*envelope
            torch.testing.assert_close(c-a,target,rtol=5e-5,atol=float(target.max())*2e-5)


def test_isotropic_noise_fills_hole_and_dilutes_v2():
    axis=torch.linspace(-.17,.17,101,dtype=torch.float64)
    x,y=torch.meshgrid(axis,axis,indexing='ij')
    coherent=(x*x+2*y*y)*torch.exp(-300*(x*x+y*y))
    noise=.002*torch.exp(-4*(x*x+y*y))
    _,yields,v2,hole=summarize_shapes(torch.stack((coherent,coherent+noise)),axis,.17)
    torch.testing.assert_close(v2[1],v2[0]*yields[0]/yields[1],rtol=1e-12,atol=1e-12)
    assert hole[0]==0 and hole[1]>0


def test_ratio_masks_the_dark_reference_instead_of_epsilon_division():
    axis=np.linspace(-.17,.17,11)
    reference=np.ones((11,11));reference[5,5]=0
    ratio,keep=masked_ratio(2*reference,reference,axis)
    assert np.isnan(ratio[5,5]) and not keep[5,5]
    np.testing.assert_array_equal(ratio[keep],2.)
