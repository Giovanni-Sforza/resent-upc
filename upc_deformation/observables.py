"""Mechanism-led bridge from ideal fringe/bright-region metrics to measured axes.

This is a finite-b-window forward study, not a calibrated experiment forecast.
All ensemble sums are performed on unnormalized intensity/Stokes bilinears.
"""
from __future__ import annotations
import copy
import math
import time
import numpy as np
import torch
from scipy.integrate import simpson, cumulative_trapezoid
from scipy.optimize import brentq
from .forward import UPCForwardModel
from .source_amplitudes import source_stokes
from .decay import decay_coefficients, decay_axis_image, b_quadrature, normalize_in_disk

STAGES = ['b_axis', 'decay_fixed', 'decay_b', 'decay_by']
METRIC_NAMES = ['F_dip', 'mean_q', 'C2']


def deformed_ws_rms(radius, diffuseness, beta2=0., beta3=0.):
    """Axisymmetric 3D RMS, not the Woods--Saxon half-density radius."""
    mu, weights = np.polynomial.legendre.leggauss(64)
    r = np.linspace(0, max(24., radius * 2 + 30 * diffuseness), 2401)
    y20 = np.sqrt(5 / (16 * np.pi)) * (3 * mu**2 - 1)
    y30 = np.sqrt(7 / (16 * np.pi)) * (5 * mu**3 - 3 * mu)
    surface = radius * (1 + beta2 * y20 + beta3 * y30)
    angular = (1 / (1 + np.exp((r[:, None] - surface) / diffuseness))) @ weights
    return float(np.sqrt(simpson(angular * r**4, x=r) / simpson(angular * r**2, x=r)))


def case_physics(config, case):
    physics = copy.deepcopy(config['physics'])
    nuc = physics['nuclear_parameters']
    b2, b3 = case['beta2'], case['beta3']
    nuc.update(beta2=b2, beta3=b3)
    rp = deformed_ws_rms(nuc['R_ws_p'], nuc['d_ws_p'], b2, b3)
    target = rp + case['skin_fm']
    nuc['R_ws_n'] = brentq(lambda radius: deformed_ws_rms(radius, nuc['d_ws_n'], b2, b3) - target,
                            .6*nuc['R_ws_p'], 1.8*nuc['R_ws_p'])
    return physics, {'proton_rms_fm': rp, 'neutron_rms_fm': target,
                     'skin_rms_fm': case['skin_fm'], 'R_ws_n_fm': nuc['R_ws_n']}


def integrate_to(r, values, upper):
    """Piecewise-linear integration with the exact requested upper boundary."""
    r, values = np.asarray(r), np.asarray(values)
    if not r[0] <= upper <= r[-1]:
        raise ValueError('Integration endpoint outside radial support')
    keep = r < upper
    rr = np.r_[r[keep], upper]
    tail = np.array([np.interp(upper, r, v) for v in values.reshape(-1, len(r))])
    vv = np.concatenate((values[..., keep], tail.reshape(*values.shape[:-1], 1)), axis=-1)
    return np.trapezoid(vv, rr, axis=-1)


def coefficient_moments(coefficients, radii, qmax=.17, q0=.02):
    """Unnormalized yield and numerators for Fdip, mean q and cos2alpha."""
    c = np.asarray(coefficients)
    a0, c2 = c[..., 0, :], c[..., 1, :]
    den = 2*np.pi*integrate_to(radii, a0*radii, qmax)
    nums = np.stack((2*np.pi*integrate_to(radii, a0*radii, q0),
                     2*np.pi*integrate_to(radii, a0*radii**2, qmax),
                     -.5*np.pi*integrate_to(radii, c2*radii, qmax)), axis=-1)
    return den, nums


def coefficient_metrics(coefficients, radii, qmax=.17, q0=.02):
    den, nums = coefficient_moments(coefficients, radii, qmax, q0)
    if np.any(den <= 0):
        raise ValueError('Nonpositive coefficient yield')
    values = nums / den[..., None]
    if values.ndim != 1:
        raise ValueError('Expected one set of radial coefficients')
    a0 = coefficients[0]
    second = 2*np.pi*integrate_to(radii, a0*radii**3, qmax)/den
    variance = np.array([values[0]*(1-values[0]), second-values[1]**2, .5-values[2]**2])
    cdf = cumulative_trapezoid(a0*radii, radii, initial=0)
    total = float(integrate_to(radii, a0*radii, qmax))
    return {**dict(zip(METRIC_NAMES, map(float, values))), 'yield': float(den),
            'q50': float(np.interp(.5*total, cdf, radii)),
            'event_variance': variance.tolist()}


def orientation_metric_se(per_orientation_coeff, radii, qmax, q0):
    """Delta-method Monte Carlo error of intensity-averaged orientation ratios."""
    den, num = coefficient_moments(per_orientation_coeff, radii, qmax, q0)
    value = num.mean(0)/den.mean()
    influence = (num-den[:, None]*value)/den.mean()
    se = np.std(influence, axis=0, ddof=1)/np.sqrt(len(den)) if len(den)>1 else np.zeros(3)
    return value, se, influence


def image_metrics(image, axis, qmax, dark_mask=None):
    qx, qy = np.meshgrid(axis, axis, indexing='ij')
    disk = qx*qx+qy*qy <= qmax*qmax
    dx2 = float(axis[1]-axis[0])**2
    norm = image/(image[disk].sum()*dx2)
    values = norm[disk]
    lowest = max(1, int(np.ceil(.05*len(values))))
    result = {'area_halfmax': float(np.count_nonzero(values > .5*values.max())*dx2),
              'fringe_low5': float(np.partition(values, lowest-1)[:lowest].mean())}
    if dark_mask is not None:
        result['dark_fixed_mean'] = float(norm[dark_mask].mean())
    return result


def calculate_case(config, case, log=print):
    cfg = config['study']
    physics, density = case_physics(config, case)
    model = UPCForwardModel(physics, cfg['longitudinal_units']).eval()
    if model._output_L/2 <= cfg['b_max_fm'] + 16:
        raise ValueError('Field domain does not cover registered b window plus nuclear support')
    b = torch.linspace(cfg['b_min_fm'], cfg['b_max_fm'], cfg['b_nodes'], dtype=torch.float64)
    bw = b_quadrature(b)
    y, yw = np.polynomial.legendre.leggauss(cfg['n_y'])
    y, yw = y*cfg['ymax'], yw*cfg['ymax']
    if cfg['n_y'] % 2 != 1:
        raise ValueError('Odd Gauss node count required to retain y=0 bridge')
    yi = int(np.argmin(abs(y)))
    bi_fixed = int(torch.argmin(abs(b-cfg['fixed_b_fm'])))
    if abs(float(b[bi_fixed])-cfg['fixed_b_fm']) > 1e-8:
        raise ValueError('fixed_b must be present in b grid')
    axis = model._p_axis_q
    r = torch.linspace(0, float(axis.abs().max())*math.sqrt(2)*(1+1e-6), cfg['n_radial'])
    n = cfg['n_orientations'] if case['beta2'] or case['beta3'] else 1
    banks=[]
    for seed in [cfg['orientation_seed'], cfg['orientation_seed']+1]:
        generator=torch.Generator(device='cpu').manual_seed(seed)
        banks.append(model._sample_uniform_rotations(1,n,torch.device('cpu'),torch.float32,generator)[0]
                     if n>1 else torch.eye(3)[None])
    shape=(len(b),len(y),2,3,len(r))
    sums=np.zeros(shape,dtype=np.float64)
    half_sums=np.zeros_like(sums)
    fixed_maps=np.zeros((2,len(axis),len(axis)),dtype=np.float64)
    per_orientation=np.zeros((n,3,2,3,len(r)),dtype=np.float64)
    incoherent=np.zeros((len(b),len(y)),dtype=np.float64)
    started=time.monotonic()
    center=len(axis)//2
    with torch.inference_mode():
        for oi in range(n):
            node=np.empty(shape,dtype=np.float64)
            for bi,bvalue in enumerate(b):
                full,self_map,pfull,inc=source_stokes(model,torch.tensor(y[None],dtype=torch.float32),
                    [float(bvalue)],banks[0][oi:oi+1],banks[1][oi:oi+1],return_incoherent=True)
                maps=torch.stack((full[0,0],self_map[0,0]),dim=1)
                node[bi]=decay_coefficients(maps,pfull,r,cfg['n_phi']).double().numpy()
                # Empirical template is B/pi exp(-Bq²); recover its total probability.
                incoherent[bi]+=inc[0,0,:,center,center].double().numpy()*np.pi/model.b_slope_incoh/n
                if bi==bi_fixed:
                    projection,_=model.project_full_grid_intensity_to_output(maps[yi])
                    fixed_maps+=projection[:,0].double().numpy()/n
                del full,self_map,maps,inc
            sums+=node/n
            if oi<n//2:
                half_sums+=node/max(1,n//2)
            per_orientation[oi,0]=node[bi_fixed,yi]
            per_orientation[oi,1]=np.einsum('b,bscr->scr',bw.numpy(),node[:,yi])
            per_orientation[oi,2]=np.einsum('b,y,byscr->scr',bw.numpy(),yw,node)
            log(f'{case["name"]}: orientations {oi+1}/{n}, {time.monotonic()-started:.1f}s',flush=True)
    coeff=per_orientation.mean(0)
    if n==1:half_sums=sums.copy()
    images=np.empty((4,2,len(axis),len(axis)))
    images[0]=fixed_maps
    for stage in range(3):
        images[stage+1]=decay_axis_image(torch.from_numpy(coeff[stage]),r.double(),axis.double()).numpy()
    norm,yields=normalize_in_disk(torch.from_numpy(images),axis.double(),cfg['qmax'])
    br=physics['output_settings']['b_slope_incoh']
    background=np.zeros((3,len(r)))
    background[0]=np.einsum('b,y,by->',bw.numpy(),yw,incoherent)*br/np.pi*np.exp(-br*r.numpy()**2)
    total_coeff=coeff[-1,0]+background
    checks={'min_full_intensity':float(images[:,0].min()),'min_self_intensity':float(images[:,1].min()),
            'fixed_axis_vs_decay_yield_relative':float(yields[1,0]/yields[0,0]-1),
            'b_weights_sum_fm2':float(bw.sum()),'y_weights_sum':float(yw.sum())}
    # Same native coefficient cache, coarser b quadrature and narrower y acceptance.
    coarse=np.arange(0,len(b),2)
    if coarse[-1]==len(b)-1 and len(coarse)>1:
        coarse_coeff=np.einsum('b,y,byscr->scr',b_quadrature(b[coarse]).numpy(),yw,sums[coarse])
    else:coarse_coeff=coeff[-1].copy()
    endpoint_fraction=[]
    for sl in [slice(0,min(6,len(b))),slice(max(0,len(b)-6),len(b))]:
        bb=b[sl]
        part=np.einsum('b,y,byscr->scr',b_quadrature(bb).numpy(),yw,sums[sl])
        pd,_=coefficient_moments(part[0],r.numpy(),cfg['qmax'],cfg['q0'])
        td,_=coefficient_moments(coeff[-1,0],r.numpy(),cfg['qmax'],cfg['q0'])
        endpoint_fraction.append(float(pd/td))
    checks['edge_b_band_yield_fractions']=endpoint_fraction
    return {'case':case,'density':density,'physics':physics,'stage_names':STAGES,'axis':axis.numpy(),
            'radii':r.numpy(),'b_values':b.numpy(),'b_weights':bw.numpy(),'y_values':y,'y_weights':yw,
            'coefficients':coeff,'node_coefficients':sums,'half_node_coefficients':half_sums,
            'orientation_coefficients':per_orientation,'coarse_b_coefficients':coarse_coeff,
            'images':images,'normalized_images':norm.numpy(),'yields':yields.numpy(),
            'incoherent_probabilities':incoherent,'background_coefficients':background,
            'total_coefficients':total_coeff,'rotation_bank_1':banks[0].numpy(),'rotation_bank_2':banks[1].numpy(),
            'checks':checks,'n_orientations':n,'elapsed_seconds':time.monotonic()-started}


def analyze_results(results, config):
    cfg=config['study'];baseline=next(x for x in results if x['case']['name']=='spherical')
    axis=baseline['axis'];qx,qy=np.meshgrid(axis,axis,indexing='ij');disk=qx*qx+qy*qy<=cfg['qmax']**2
    dark_masks=[]
    for image in baseline['normalized_images'][:,0]:
        threshold=np.quantile(image[disk],.05)
        dark_masks.append(disk&(image<=threshold))
    for result in results:
        result['metrics']=[];result['orientation_se']=[];result['q0_scan']={}
        for stage in range(4):
            stage_metrics=[];stage_se=[]
            for branch in range(2):
                cm=coefficient_metrics(result['coefficients'][max(0,stage-1),branch],result['radii'],cfg['qmax'],cfg['q0'])
                cm.update(image_metrics(result['images'][stage,branch],axis,cfg['qmax'],dark_masks[stage]))
                if stage==0:
                    image=result['normalized_images'][stage,branch]
                    angle=np.arctan2(qy,qx); harmonic=np.cos(2*angle); harmonic[len(axis)//2,len(axis)//2]=0
                    cm['C2']=float((image*disk*harmonic).sum()/(image*disk).sum())
                    cm['event_variance'][2]=float((image*disk*harmonic**2).sum()/(image*disk).sum()-cm['C2']**2)
                _,se,_=orientation_metric_se(result['orientation_coefficients'][:,max(0,stage-1),branch],
                    result['radii'],cfg['qmax'],cfg['q0'])
                se_list=se.tolist()
                if stage==0:se_list[2]=None  # true-axis per-orientation images are not retained
                stage_metrics.append(cm);stage_se.append(se_list)
            result['metrics'].append(stage_metrics);result['orientation_se'].append(stage_se)
        for q0 in cfg['q0_candidates']:
            result['q0_scan'][str(q0)]=[coefficient_metrics(result['coefficients'][-1,branch],result['radii'],cfg['qmax'],q0)['F_dip'] for branch in range(2)]
        result['coarse_b_metrics']=[coefficient_metrics(c,result['radii'],cfg['qmax'],cfg['q0']) for c in result['coarse_b_coefficients']]
        result['background_metrics']=coefficient_metrics(result['total_coefficients'],result['radii'],cfg['qmax'],cfg['q0'])
        bg_yield=coefficient_metrics(result['background_coefficients'],result['radii'],cfg['qmax'],cfg['q0'])['yield']
        result['background_fraction']=bg_yield/result['background_metrics']['yield']
    return results
