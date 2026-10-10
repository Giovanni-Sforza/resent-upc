"""Nuclear-structure response in the b-integrated lepton reference frame."""
from __future__ import annotations
import copy
import math
import time
import numpy as np
import torch
from scipy.integrate import simpson
from scipy.optimize import brentq
from .forward import UPCForwardModel
from .decay import stokes_maps, decay_coefficients, decay_axis_image, b_quadrature, normalize_in_disk


def ws_rms(radius, diffuseness):
    r=np.linspace(0, radius+40*diffuseness, 10001)
    density=1/(1+np.exp((r-radius)/diffuseness))
    return float(np.sqrt(simpson(density*r**4,x=r)/simpson(density*r**2,x=r)))


def neutron_radius_for_skin(rp, ap, an, skin):
    target=ws_rms(rp,ap)+skin
    return float(brentq(lambda rn:ws_rms(rn,an)-target,.5*rp,2*rp))


def masked_ratio(image, reference, axis, radius=.17, floor_fraction=.001):
    x,y=np.meshgrid(axis,axis,indexing='ij')
    keep=(x*x+y*y<=radius**2)&(reference>floor_fraction*reference.max())
    ratio=np.full_like(image,np.nan,dtype=float)
    np.divide(image,reference,out=ratio,where=keep)
    return ratio,keep


def summarize_shapes(raw, axis, radius):
    norm,yields=normalize_in_disk(raw,axis,radius)
    x,y=torch.meshgrid(axis,axis,indexing='ij')
    disk=x*x+y*y<=radius**2
    angle=torch.atan2(y,x); mid=len(axis)//2
    harmonic=torch.cos(2*angle)
    harmonic[mid,mid]=0  # angular average of the central square pixel
    v2=(norm*disk*harmonic).sum((-2,-1))/(norm*disk).sum((-2,-1))
    return norm,yields,v2,norm[...,mid,mid]/norm.amax((-2,-1))


def calculate_structure_case(config, case, log=print):
    cfg=config['comparison']; physics=copy.deepcopy(config['physics'])
    nuc=physics['nuclear_parameters'];nuc.update(case['nuclear_overrides'])
    model=UPCForwardModel(physics,cfg['longitudinal_units']).eval()
    device=torch.device(cfg.get('device','cpu'));model=model.to(device)
    support=max(nuc['R_ws_p'],nuc['R_ws_n'])*(1+abs(nuc['beta2'])+.75*abs(nuc['beta3']))+8*max(nuc['d_ws_p'],nuc['d_ws_n'])
    if model._output_L/2<=cfg['b_max_fm']+support:raise ValueError('Increase field domain')
    if model.incoh_r_max<=cfg['b_max_fm']/2+support:raise ValueError('Increase incoherent quadrature domain')
    axis=model._p_axis_q
    radii=torch.linspace(0,float(axis.abs().max())*math.sqrt(2)*(1+1e-6),cfg['n_radial'],device=device)
    b=torch.linspace(cfg['b_min_fm'],cfg['b_max_fm'],cfg['b_nodes'],dtype=torch.float64,device=device)
    weights=b_quadrature(b)
    n=cfg['n_orientations'] if case['orientation']=='independent_average' else 1
    if n>1:
        # Independent reproducible streams: extending n preserves both prefixes.
        banks=[]
        for seed in [cfg['seed'],cfg['seed']+1]:
            gen=torch.Generator(device='cpu').manual_seed(seed)
            banks.append(model._sample_uniform_rotations(1,n,torch.device('cpu'),torch.float32,gen)[0].to(device))
        r1,r2=banks
    else:r1=r2=torch.eye(3,device=device)[None]
    coeff=torch.zeros(n,len(b),3,len(radii),dtype=torch.float64,device=device)
    probabilities=torch.zeros(n,len(b),dtype=torch.float64,device=device)
    started=time.monotonic(); checks={}; density_rms=[]
    with torch.inference_mode():
        for start in range(0,n,cfg['orientation_chunk']):
            stop=min(start+cfg['orientation_chunk'],n);count=stop-start
            ys=torch.full((count,1),cfg['y'],device=device)
            for bi in range(0,len(b),cfg['b_chunk']):
                bj=min(bi+cfg['b_chunk'],len(b))
                mx,my,pfull=model.fixed_b_complex_amplitude_multi_y(None,None,ys,b[bi:bj].tolist(),R_1=r1[start:stop],R_2=r2[start:stop])
                maps=stokes_maps(mx[:,:,0],my[:,:,0])
                # Preserve each orientation. Squaring precedes every average.
                coeff[start:stop,bi:bj]=decay_coefficients(maps,pfull,radii,cfg['n_phi']).double()
                del mx,my,maps
            probabilities[start:stop]=model.fixed_b_empirical_incoherent_probability_multi_y(
                ys,b.tolist(),R_1=r1[start:stop],R_2=r2[start:stop])[:,:,0].double()
            if start==0:
                # Independent source forward closure at the first b node.
                old=model.b_imp;model.b_imp=float(b[0])
                off,_=model.forward_multi_y(None,y_views=ys,R_1=r1[start:stop],R_2=r2[start:stop],incoherent=False)
                on,_=model.forward_multi_y(None,y_views=ys,R_1=r1[start:stop],R_2=r2[start:stop],incoherent=True)
                qx,qy=torch.meshgrid(axis,axis,indexing='ij')
                envelope=model.b_slope_incoh/math.pi*torch.exp(-model.b_slope_incoh*(qx*qx+qy*qy))
                expected=probabilities[start:stop,0,None,None]*envelope
                checks['incoherent_on_off_closure_relative_max']=float(((on-off)[:,0]-expected).abs().max()/expected.abs().max())
                model.b_imp=old
                # Refine the actual source quadrature, without changing physics.
                nr,np_=model.incoh_n_r,model.incoh_n_phi
                model.incoh_n_r=2*nr;model.incoh_n_phi=2*np_
                refined=model.fixed_b_empirical_incoherent_probability_multi_y(ys,b.tolist(),R_1=r1[start:stop],R_2=r2[start:stop])[:,:,0].double()
                model.incoh_n_r=nr;model.incoh_n_phi=np_
                coarse_sum=(probabilities[start:stop]*weights).sum(1)
                refined_sum=(refined*weights).sum(1)
                checks['incoherent_quadrature_refinement_relative_max']=float((refined_sum/coarse_sum-1).abs().max())
                for nid,bank in [(1,r1),(2,r2)]:
                    rp,rt=model._compute_density_ws(nuclear_ID=nid,R=bank[:1])
                    g=model._WS_x_grid;gx,gy,gz=torch.meshgrid(g,g,g,indexing='ij');r2grid=gx*gx+gy*gy+gz*gz
                    rn=rt-rp
                    pr=torch.sqrt((rp*r2grid).sum()/rp.sum());ne=torch.sqrt((rn*r2grid).sum()/rn.sum())
                    density_rms.append({'proton_fm':float(pr),'neutron_fm':float(ne),'skin_fm':float(ne-pr)})
            log(f"{case['name']}: orientations {stop}/{n}; {time.monotonic()-started:.1f}s",flush=True)
            if time.monotonic()-started>cfg.get('timeout_seconds',1800):raise TimeoutError('Case timeout')
        integrated_coeff=(coeff*weights[None,:,None,None]).sum(1)
        images=decay_axis_image(integrated_coeff,radii,axis)
        background_yields=(probabilities*weights).sum(1)
        background=background_yields[:,None,None]*envelope
        coherent=images.mean(0);inc=background.mean(0)
        raw=torch.stack((coherent,coherent+inc))
        normalized,yields,v2,hole=summarize_shapes(raw,axis,cfg['normalization_radius_GeV'])
        inc_yield=yields[1]-yields[0]
        coarse_idx=torch.arange(0,len(b),2,device=device)
        coarse_b=(((coeff[:,coarse_idx]*b_quadrature(b[coarse_idx])[None,:,None,None]).sum(1)).mean(0))
        coarse_img=decay_axis_image(coarse_b,radii,axis)
        coarse_norm,_=normalize_in_disk(coarse_img,axis,cfg['normalization_radius_GeV'])
        qx,qy=torch.meshgrid(axis,axis,indexing='ij');disk=qx*qx+qy*qy<=cfg['normalization_radius_GeV']**2
        checks['b_half_vs_full_coherent_shape_l1']=float(((coarse_norm-normalized[0]).abs()*disk).sum()/(normalized[0]*disk).sum())
        checks['v2_isotropic_dilution_absolute_error']=float((v2[1]-v2[0]*yields[0]/yields[1]).abs())
        checks['minimum_raw_intensity']=float(raw.min())
        if n>1:
            partial=torch.stack((images[:n//2].mean(0),(images[:n//2]+background[:n//2]).mean(0)))
            partial_norm,_=normalize_in_disk(partial,axis,cfg['normalization_radius_GeV'])
            checks['half_vs_full_orientation_shape_l1']=(((partial_norm-normalized).abs()*disk).sum((-2,-1))/(normalized*disk).sum((-2,-1))).tolist()
        if checks['incoherent_on_off_closure_relative_max']>2e-4:raise AssertionError('Source incoherent closure failed')
        if checks['v2_isotropic_dilution_absolute_error']>1e-6:raise AssertionError('Background is not isotropic')
        if not torch.isfinite(raw).all() or raw.min()<0:raise FloatingPointError('Invalid intensities')
    def arr(t):return t.detach().cpu().numpy()
    return {'case':case,'physics':physics,'checks':checks,'elapsed_seconds':time.monotonic()-started,
            'density_rms_first_orientation':density_rms,'orientations':n,
            'axis':arr(axis),'radii':arr(radii),'b_values':arr(b),'b_weights':arr(weights),
            'coefficients_per_b':arr(coeff.mean(0)),'coefficients_per_orientation':arr(integrated_coeff),
            'incoherent_probability_per_orientation_b':arr(probabilities),
            'coherent_images_per_orientation':arr(images).astype(np.float32),
            'incoherent_yield_per_orientation':arr(background_yields),
            'rotation_bank_1':arr(r1),'rotation_bank_2':arr(r2),
            'raw_images':arr(raw),'incoherent_image':arr(inc),'normalized_images':arr(normalized),
            'yields':arr(yields),'v2':arr(v2),'center_to_peak':arr(hole),
            'incoherent_fraction_in_disk':float(inc_yield/yields[1])}
