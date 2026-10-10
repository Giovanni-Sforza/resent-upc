"""Load verified bridge caches and combine independent raw orientation shards."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml

CORE_FILES=('upc_deformation/observables.py','upc_deformation/source_amplitudes.py',
            'upc_deformation/forward.py','upc_deformation/decay.py','upc_deformation/vendor/upc_forward.py')

def load_case(directory,name):
    directory=Path(directory)
    result=json.loads((directory/f'{name}.json').read_text())
    with np.load(directory/f'{name}.npz',allow_pickle=False) as z:
        result.update({k:z[k] for k in z.files})
    return result

def _same(left,right):
    return json.dumps(left,sort_keys=True)==json.dumps(right,sort_keys=True)

def merge_orientation_shards(base,extra,qmax=.17):
    """Combine equal-size independent MC shards BEFORE any normalization.

    The first half of the merged node average is exactly the archived shard.
    This intentionally does not pretend that a changed RNG sample count would
    reproduce a prefix of a vectorized SO(3) sampler.
    """
    for key in ('case','physics','density','stage_names'):
        if not _same(base[key],extra[key]):raise ValueError(f'Shard {key} mismatch')
    for key in ('axis','radii','b_values','b_weights','y_values','y_weights'):
        if not np.array_equal(base[key],extra[key]):raise ValueError(f'Shard {key} mismatch')
    n=int(base['n_orientations']);m=int(extra['n_orientations'])
    if n<2 or n!=m:raise ValueError('Require two equally sized deformed orientation shards')
    for shard in (base,extra):
        if len(shard['orientation_coefficients'])!=n:raise ValueError('Orientation count mismatch')
        if not np.allclose(shard['orientation_coefficients'].mean(0),shard['coefficients'],rtol=1e-12,atol=1e-13):
            raise ValueError('Orientation means do not reproduce raw coefficients')
    pairs=[]
    for shard in (base,extra):
        pairs.append(np.concatenate([shard['rotation_bank_1'].reshape(n,-1),shard['rotation_bank_2'].reshape(n,-1)],axis=1))
    if any(np.array_equal(a,b) for a in pairs[0] for b in pairs[1]):
        raise ValueError('Shards contain duplicate orientation pairs')
    result=dict(base)
    average_keys=('node_coefficients','coarse_b_coefficients','images',
                  'incoherent_probabilities','background_coefficients')
    for key in average_keys:
        if base[key].shape!=extra[key].shape:raise ValueError(f'Shard {key} shape mismatch')
        result[key]=.5*(base[key]+extra[key])
    for key in ('orientation_coefficients','rotation_bank_1','rotation_bank_2'):
        result[key]=np.concatenate((base[key],extra[key]),axis=0)
    result['coefficients']=result['orientation_coefficients'].mean(0)
    result['half_node_coefficients']=base['node_coefficients'].copy()
    result['total_coefficients']=result['coefficients'][-1,0]+result['background_coefficients']
    axis=result['axis'].astype(np.float64);x,y=np.meshgrid(axis,axis,indexing='ij')
    # Original bridge analysis window, separate from later finite-q bins.
    disk=x*x+y*y<=qmax**2;pixel_area=float(axis[1]-axis[0])**2
    result['yields']=(result['images']*disk).sum(axis=(-2,-1))*pixel_area
    result['normalized_images']=result['images']/result['yields'][...,None,None]
    result['n_orientations']=n+m
    result['elapsed_seconds']=base['elapsed_seconds']+extra['elapsed_seconds']
    result['checks']={
        'min_full_intensity':float(result['images'][:,0].min()),
        'min_self_intensity':float(result['images'][:,1].min()),
        'fixed_axis_vs_decay_yield_relative':float(result['yields'][1,0]/result['yields'][0,0]-1),
        'b_weights_sum_fm2':float(result['b_weights'].sum()),
        'y_weights_sum':float(result['y_weights'].sum()),
        'shard_mean_coefficients_max_relative_error':float(np.max(np.abs(result['coefficients']-.5*(base['coefficients']+extra['coefficients'])))/np.max(np.abs(result['coefficients']))),
    }
    # The old per-case computation hash labels only the original sample.
    result.pop('calculation_hash',None)
    result.pop('b_window_reuse',None)
    result['orientation_extension']={
        'original_n':n,'additional_n':m,'additional_seed':extra.get('orientation_seed'),
        'shard_fingerprint':extra.get('shard_fingerprint'),
        'combination':'equal-weight raw intensities and coefficients; concatenate independent rotation pairs',
        'uncertainty_scope':'orientation Monte Carlo only; not a convergence certificate',
    }
    return result

def load_angular_cases(source,extension=None):
    source=Path(source).resolve();root=Path(__file__).resolve().parents[1]
    config=yaml.safe_load((source/'config.yaml').read_text())
    manifest=json.loads((source/'manifest.json').read_text())
    if manifest.get('status')!='completed':raise ValueError('Source run is not completed')
    hashes={k:hashlib.sha256((root/k).read_bytes()).hexdigest() for k in CORE_FILES}
    if any(manifest['source_hashes'].get(k)!=v for k,v in hashes.items()):
        raise ValueError('Current forward implementation differs from source run')
    shard_manifest=None
    if extension is not None:
        extension=Path(extension).resolve()
        shard_manifest=json.loads((extension/'manifest.json').read_text())
        if shard_manifest.get('status')!='completed':raise ValueError('Orientation extension is not completed')
        if shard_manifest.get('source')!=str(source) or shard_manifest.get('source_hashes')!=hashes:
            raise ValueError('Orientation extension provenance mismatch')
    results=[];provenance=[]
    for case in config['study']['cases']:
        name=case['name'];result=load_case(source,name)
        paths=[source/f'{name}.npz',source/f'{name}.json']
        if shard_manifest and name in shard_manifest['completed_cases']:
            extra=load_case(extension,name)
            if extra.get('shard_fingerprint')!=shard_manifest['fingerprint']:
                raise ValueError('Orientation shard fingerprint mismatch')
            result=merge_orientation_shards(result,extra,config['study']['qmax'])
            paths.extend([extension/f'{name}.npz',extension/f'{name}.json'])
        mean_error=float(np.max(np.abs(result['orientation_coefficients'].mean(0)-result['coefficients'])) /
                         max(float(np.max(np.abs(result['coefficients']))),1e-300))
        if mean_error>1e-12:raise ValueError(f'{name}: raw orientation average does not close')
        provenance.append({'case':name,'n_orientations':result['n_orientations'],
                           'orientation_mean_relative_error':mean_error,
                           'orientation_extension':result.get('orientation_extension'),
                           'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}})
        results.append(result)
    return results,config,provenance
