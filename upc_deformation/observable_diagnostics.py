"""Mechanism interaction contrasts; no claim of additive normalized observables."""
from __future__ import annotations
import numpy as np
from .observables import METRIC_NAMES, coefficient_metrics, orientation_metric_se
from .decay import b_quadrature
import torch


def mechanism_diagnostics(results,config):
    cfg=config['study'];base=next(r for r in results if r['case']['name']=='spherical')
    # Audit support without inferring mechanism from radius: outer high-q arms
    # can belong to the same I-shaped interference fringe as its central stem.
    qx,qy=np.meshgrid(base['axis'],base['axis'],indexing='ij')
    radius=np.hypot(qx,qy);disk=radius<=cfg['qmax'];dark_regions=[]
    for stage,image in zip(base['stage_names'],base['normalized_images'][:,0]):
        mask=disk&(image<=np.quantile(image[disk],.05))
        dark_regions.append({'stage':stage,'q_quantiles_GeV':np.quantile(radius[mask],[0,.25,.5,.75,1]).tolist(),
            'mask_pixel_fraction_below_q0':float(np.mean(radius[mask]<cfg['q0'])),
            'mask_pixel_fraction_above_0p9_qmax':float(np.mean(radius[mask]>.9*cfg['qmax'])),
            'fraction_of_analysis_yield':float(image[mask].sum()/image[disk].sum()),
            'interpretation':'Location only: high-q support may contain horizontal interference-fringe arms. Pixel fractions are not yield fractions; independently selected stage masks do not establish pixel transport.'})
    records=[]
    for result in results:
        if result is base:continue
        for stage in range(3):
            values=[];influences=[]
            for branch in range(2):
                v,se,influence=orientation_metric_se(result['orientation_coefficients'][:,stage,branch],
                    result['radii'],cfg['qmax'],cfg['q0'])
                bmetrics=coefficient_metrics(base['coefficients'][stage,branch],base['radii'],cfg['qmax'],cfg['q0'])
                values.append(v-np.array([bmetrics[k] for k in METRIC_NAMES]));influences.append(influence)
            interaction=values[0]-values[1]
            paired=influences[0]-influences[1]
            paired_se=np.std(paired,axis=0,ddof=1)/np.sqrt(len(paired)) if len(paired)>1 else np.zeros(3)
            records.append({'case':result['case']['name'],'stage':result['stage_names'][stage+1],
                'metrics':METRIC_NAMES,'delta_full':values[0].tolist(),'delta_self':values[1].tolist(),
                'interference_interaction_contrast':interaction.tolist(),'interaction_orientation_se':paired_se.tolist(),
                'interpretation':'Difference of structural responses with/without cross term; not an additive decomposition of a normalized observable.'})
    ranges=[]
    candidate_ranges=[(cfg['b_min_fm'],20.),(cfg['b_min_fm'],40.),(20.,60.),(30.,60.),(40.,60.)]
    for blo,bhi in dict.fromkeys(candidate_ranges):
        if bhi<=blo:continue
        for result in results:
            b=result['b_values'];keep=(b>=blo)&(b<=bhi)
            if keep.sum()<2 or b[keep][0]!=blo or b[keep][-1]!=bhi:continue
            weights=b_quadrature(torch.from_numpy(b[keep])).numpy()
            coeff=np.einsum('b,y,byscr->scr',weights,result['y_weights'],result['node_coefficients'][keep])
            m=coefficient_metrics(coeff[0],result['radii'],cfg['qmax'],cfg['q0'])
            ranges.append({'case':result['case']['name'],'b_range_fm':[blo,bhi],
                           **{k:m[k] for k in METRIC_NAMES}})
    windows=[]
    # The old manuscript's 0.005 window is a labelled reference, not a new
    # selection replacing the registered primary q0 after inspecting results.
    for q0 in sorted(set([.005,*cfg['q0_candidates']])):
        for result in results:
            if result is base:continue
            for stage in [0,2]:
                deltas=[];errors=[];influences=[]
                for branch in [0,1]:
                    v,se,inf=orientation_metric_se(result['orientation_coefficients'][:,stage,branch],result['radii'],cfg['qmax'],q0)
                    vb=coefficient_metrics(base['coefficients'][stage,branch],base['radii'],cfg['qmax'],q0)['F_dip']
                    deltas.append(float(v[0]-vb));errors.append(float(se[0]));influences.append(inf[:,0])
                paired=influences[0]-influences[1]
                se=float(np.std(paired,ddof=1)/np.sqrt(len(paired))) if len(paired)>1 else 0.
                windows.append({'case':result['case']['name'],'stage':result['stage_names'][stage+1],
                    'q0_GeV':q0,'window_role':'manuscript_window_reference' if q0==.005 else 'pre_registered_candidate',
                    'delta_full':deltas[0],'full_orientation_se':errors[0],
                    'delta_self':deltas[1],'self_orientation_se':errors[1],
                    'interaction_contrast':deltas[0]-deltas[1],'interaction_orientation_se':se})
    mixed=None
    byname={r['case']['name']:r for r in results}
    names=['mixed_validation','beta2_mid','beta3_mid','skin_mid']
    if all(n in byname for n in names):
        vals={name:np.array([byname[name]['metrics'][-1][0][k] for k in METRIC_NAMES]) for name in names}
        vb=np.array([base['metrics'][-1][0][k] for k in METRIC_NAMES])
        additive=sum(vals[n]-vb for n in names[1:])
        actual=vals['mixed_validation']-vb
        # Shared seeds permit a paired orientation-error calculation.
        influences=[]
        for name in ['mixed_validation','beta2_mid','beta3_mid']:
            r=byname[name]
            _,_,inf=orientation_metric_se(r['orientation_coefficients'][:,-1,0],r['radii'],cfg['qmax'],cfg['q0'])
            influences.append(inf)
        if len({len(v) for v in influences})==1:
            for name in ['beta2_mid','beta3_mid']:
                for key in ['rotation_bank_1','rotation_bank_2']:
                    if not np.array_equal(byname['mixed_validation'][key],byname[name][key]):
                        raise ValueError('Mixed diagnostic requires genuinely paired orientation banks')
            residual_inf=influences[0]-influences[1]-influences[2]
            se=np.std(residual_inf,axis=0,ddof=1)/np.sqrt(len(residual_inf))
        else:se=np.full(3,np.nan)
        mixed={'case':'mixed_validation','metrics':METRIC_NAMES,'actual_difference':actual.tolist(),
               'sum_of_separate_midpoint_differences':additive.tolist(),'nonadditivity':(actual-additive).tolist(),
               'nonadditivity_orientation_se':[float(x) if np.isfinite(x) else None for x in se],
               'interpretation':'Held-out interaction check; additivity is a diagnostic approximation, not an imposed law or an inference fit.'}
    return {'dark_region_location':dark_regions,'mechanism_contrasts':records,
            'geometry_subwindows':ranges,'window_contrasts':windows,'mixed_validation':mixed}
