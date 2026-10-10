#!/usr/bin/env python3
"""Calculate independent orientation shards for finite-q angular diagnostics."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import yaml
from upc_deformation.observables import calculate_case

CORE_FILES=['upc_deformation/observables.py','upc_deformation/source_amplitudes.py',
            'upc_deformation/forward.py','upc_deformation/decay.py','upc_deformation/vendor/upc_forward.py']

def jsonable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [jsonable(v) for v in value]
    return value

def save_json(path,value):
    path.write_text(json.dumps(jsonable(value),indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='outputs/jpsi_observables_revision/b12_20261007')
    p.add_argument('--output',default='outputs/jpsi_angular_revision/b12_20261008/orientation_extension')
    p.add_argument('--seed',type=int,default=20261018)
    p.add_argument('--orientations',type=int,default=32)
    p.add_argument('--cases',nargs='+',default=['beta2','beta3'])
    args=p.parse_args()
    src=Path(args.source).resolve();dest=Path(args.output).resolve()
    if src==dest:raise ValueError('Use a separate shard directory')
    original=yaml.safe_load((src/'config.yaml').read_text())
    if args.seed in [original['study']['orientation_seed']-1,original['study']['orientation_seed'],original['study']['orientation_seed']+1]:
        raise ValueError('Use independent rotation seeds for both nuclei')
    if args.orientations<2:raise ValueError('At least two orientations required')
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in CORE_FILES}
    archived=json.loads((src/'manifest.json').read_text())['source_hashes']
    if any(archived.get(k)!=v for k,v in hashes.items()):
        raise ValueError('Forward source differs from archived production')
    config=copy.deepcopy(original)
    cfg=config['study'];cfg.update(orientation_seed=args.seed,n_orientations=args.orientations)
    cfg['cases']=[c for c in cfg['cases'] if c['name'] in args.cases]
    if {c['name'] for c in cfg['cases']}!=set(args.cases):raise ValueError('Unknown case')
    if any(not(c['beta2'] or c['beta3']) for c in cfg['cases']):raise ValueError('Refine deformed cases only')
    torch.set_num_threads(cfg['threads'])
    dest.mkdir(parents=True,exist_ok=True)
    frozen=yaml.safe_dump(config,sort_keys=False)
    if (dest/'config.yaml').exists() and (dest/'config.yaml').read_text()!=frozen:
        raise ValueError('Different shard config; choose a new output')
    (dest/'config.yaml').write_text(frozen)
    fingerprint=hashlib.sha256(json.dumps({'config':config,'source_hashes':hashes},sort_keys=True).encode()).hexdigest()
    log_file=(dest/'run.log').open('a',buffering=1)
    def log(msg,**kwargs):print(msg,**kwargs);log_file.write(str(msg)+'\n')
    manifest={'status':'running','source':str(src),'source_hashes':hashes,'fingerprint':fingerprint,
              'seed':args.seed,'orientations_per_case':args.orientations,'completed_cases':[],
              'scope':'Independent orientation Monte Carlo shard; no changes to b/y/grid/physics'}
    started=time.monotonic();save_json(dest/'manifest.json',manifest)
    for case in cfg['cases']:
        name=case['name'];meta=dest/f'{name}.json';arr=dest/f'{name}.npz'
        if meta.exists() and arr.exists():
            result=json.loads(meta.read_text())
            if result.get('shard_fingerprint')!=fingerprint:raise ValueError('Stale shard fingerprint')
            log(f'Reusing independent shard: {name}')
        else:
            result=calculate_case(config,case,log)
            result.update(shard_fingerprint=fingerprint,orientation_seed=args.seed)
            tmp=dest/f'{name}.partial.npz'
            np.savez_compressed(tmp,**{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
            tmp.replace(arr)
            save_json(meta,{k:v for k,v in result.items() if not isinstance(v,np.ndarray)})
        manifest['completed_cases'].append(name);save_json(dest/'manifest.json',manifest)
    manifest.update(status='completed',elapsed_seconds=time.monotonic()-started)
    save_json(dest/'manifest.json',manifest);log(f'Completed independent shards in {manifest["elapsed_seconds"]:.1f}s')

if __name__=='__main__':main()
