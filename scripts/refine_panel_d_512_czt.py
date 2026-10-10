#!/usr/bin/env python3
"""Complete the registered orientation bank with validated native-frequency CZT.

Previously completed dense-FFT checkpoints are retained with their original
checksums and kernel provenance. The CZT run has a separate immutable identity.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch

from refine_panel_d_512 import ROOT, prepare, save_json, save_npz, settings, sha

CZT_KERNEL = 'upc_deformation/orientation_czt_refinement.py'
_CONFIG = None
_REFINERS = {}


def initialize(config, threads):
    global _CONFIG
    torch.set_num_threads(threads)
    _CONFIG = config


def calculate(name, index, rotation1, rotation2):
    from upc_deformation.orientation_czt_refinement import PanelDOrientationCZTRefiner
    if name not in _REFINERS:
        case = next(c for c in _CONFIG['study']['cases'] if c['name'] == name)
        _REFINERS[name] = PanelDOrientationCZTRefiner(_CONFIG, case)
    result = _REFINERS[name].calculate(rotation1, rotation2, return_nodes=False)
    return name, index, {k: result[k] for k in ('coefficients', 'coarse_b_coefficients', 'elapsed_seconds')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/panel_d_c2_convergence_czt.yaml')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--workers', type=int, choices=(1,2,3,4),
        help='Runtime process count only; each forward call keeps the validated thread count')
    args = parser.parse_args()
    cfg = settings(args.config)
    workers = args.workers or cfg['workers']
    original, bases, extras, bank, identity, _ = prepare(cfg)
    out = Path(cfg['output'])
    dense = ROOT/cfg['dense_checkpoint_source']
    dense_provenance = json.loads((dense/'provenance.json').read_text())
    dense_manifest = json.loads((dense/'production_manifest.json').read_text())
    if dense_manifest['status'] == 'running':
        raise ValueError('Stop the owned dense production before freezing its checkpoints')
    if dense_manifest['fingerprint'] != dense_provenance['fingerprint']:
        raise ValueError('Dense manifest and provenance differ')
    for key in ('input_sha256', 'core_sha256', 'kernel_sha256'):
        if identity[key] != dense_provenance[key]:
            raise ValueError(f'Dense checkpoint source differs: {key}')
    dense_proof = json.loads((dense/'kernel_validation.json').read_text())
    if dense_proof['status'] != 'passed' or dense_proof['kernel_sha256'] != identity['kernel_sha256']:
        raise ValueError('Dense source kernel was not validated')
    with np.load(dense/'orientation_banks.npz') as saved:
        if any(not np.array_equal(saved[f'rotation_bank_{i+1}'], value) for i, value in enumerate(bank)):
            raise ValueError('Dense and CZT orientation banks differ')
    dense_paths = sorted(p for p in (dense/'checkpoints').glob('*/*.npz')
        if p.parent.name in cfg['cases'] and len(p.stem)==4 and p.stem.isdigit() and 64<=int(p.stem)<512)
    identity.update(additional_kernel_sha256={CZT_KERNEL: sha(ROOT/CZT_KERNEL)},
        reused_dense_provenance={str(dense/'provenance.json'): sha(dense/'provenance.json'),
            str(dense/'production_manifest.json'): sha(dense/'production_manifest.json'),
            str(dense/'kernel_validation.json'): sha(dense/'kernel_validation.json')},
        reused_dense_checkpoint_sha256={str(p): sha(p) for p in dense_paths},
        transform='same native Fourier frequencies; Stokes before original bilinear projection',
        driver_dependencies_sha256={'scripts/refine_panel_d_512.py':sha(ROOT/'scripts/refine_panel_d_512.py')})
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    previous = json.loads((out/'provenance.json').read_text()) if (out/'provenance.json').exists() else None
    if previous and previous['fingerprint'] != fingerprint:
        raise ValueError('Immutable CZT provenance changed')
    save_json(out/'provenance.json', {**identity, 'fingerprint': fingerprint})
    if args.prepare_only:
        print(f'Prepared CZT production: {out}', flush=True)
        return
    proof = json.loads((out/'kernel_validation.json').read_text())
    if (proof.get('status') != 'passed' or proof.get('kernel_sha256') != identity['additional_kernel_sha256'][CZT_KERNEL]
            or proof.get('base_kernel_sha256') != identity['kernel_sha256']
            or proof.get('core_sha256') != identity['core_sha256']
            or proof.get('threads_per_worker') != cfg['threads_per_worker']
            or proof.get('fourier_backend') != 'native_grid_czt'
            or proof.get('source_config_sha256') != sha(Path(cfg['source'])/'config.yaml')):
        raise ValueError('Validated current CZT kernel is required')
    for p in dense_paths:
        name, index = p.parent.name, int(p.stem)
        with np.load(p) as row:
            if str(row['fingerprint']) != dense_provenance['fingerprint']:
                raise ValueError(f'Dense fingerprint mismatch: {p}')
            if not np.array_equal(row['rotation1'], bank[0][index]) or not np.array_equal(row['rotation2'], bank[1][index]):
                raise ValueError(f'Dense orientation mismatch: {p}')
            values = {k: row[k] for k in ('coefficients', 'coarse_b_coefficients', 'elapsed_seconds', 'rotation1', 'rotation2')}
            if any(values[k].shape!=(3,757) or not np.isfinite(values[k]).all()
                   for k in ('coefficients','coarse_b_coefficients')):
                raise ValueError(f'Invalid dense coefficients: {p}')
        target = out/'checkpoints'/name/p.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            save_npz(target, **values, fingerprint=np.asarray(fingerprint), engine=np.asarray('validated_dense_fft'),
                original_checkpoint=np.asarray(str(p)), original_sha256=np.asarray(sha(p)))
    missing, completed = [], []
    for name in cfg['cases']:
        (out/'checkpoints'/name).mkdir(parents=True, exist_ok=True)
        for i in range(64, 512):
            path = out/'checkpoints'/name/f'{i:04d}.npz'
            if not path.exists():
                missing.append((name, i)); continue
            with np.load(path) as row:
                if (str(row['fingerprint']) != fingerprint or row['coefficients'].shape != (3,757)
                        or not np.isfinite(row['coefficients']).all()
                        or not np.array_equal(row['rotation1'], bank[0][i]) or not np.array_equal(row['rotation2'], bank[1][i])):
                    raise ValueError(f'Invalid registered checkpoint: {path}')
            completed.append(path)
    manifest = dict(status='running', fingerprint=fingerprint, target_pairs_per_case=512,
        reused_pairs_per_case=64, reused_dense_checkpoints=len(dense_paths), new_pairs_total=896,
        completed_new_pairs=len(completed), workers=workers, configured_workers=cfg['workers'],
        threads_per_worker=cfg['threads_per_worker'], parent_pid=os.getpid(),
        scope='Only orientation Monte Carlo refined; original physics and quadratures fixed')
    save_json(out/'production_manifest.json', manifest)
    log_file = (out/'production.log').open('a', buffering=1)
    def log(message):
        print(message, flush=True); log_file.write(message+'\n')
    log(f'Resume: {len(completed)}/896 checkpoints; {len(missing)} CZT calculations remaining; workers={workers}, threads={cfg["threads_per_worker"]}.')
    started = time.monotonic()
    pool = ProcessPoolExecutor(max_workers=workers, initializer=initialize,
        initargs=(original, cfg['threads_per_worker']))
    futures, pending = {}, iter(sorted(missing, key=lambda v:(v[1],v[0])))
    def enqueue():
        try: name, index = next(pending)
        except StopIteration: return
        futures[pool.submit(calculate,name,index,bank[0][index],bank[1][index])] = (name,index)
    try:
        for _ in range(2*workers): enqueue()
        while futures:
            done,_ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                name,index,result = future.result()
                path=out/'checkpoints'/name/f'{index:04d}.npz'
                save_npz(path,**result,rotation1=bank[0][index],rotation2=bank[1][index],
                    fingerprint=np.asarray(fingerprint),engine=np.asarray('validated_native_frequency_czt'))
                completed.append(path)
                manifest.update(completed_new_pairs=len(completed),current_session_seconds=time.monotonic()-started)
                save_json(out/'production_manifest.json',manifest)
                if len(completed)%16==0 or len(completed)==896:
                    log(f'New pairs {len(completed)}/896; last {name} index{index}; worker {result["elapsed_seconds"]:.2f}s; session {time.monotonic()-started:.1f}s')
                enqueue()
    except BaseException as error:
        manifest.update(status='interrupted',error=repr(error),current_session_seconds=time.monotonic()-started)
        save_json(out/'production_manifest.json',manifest)
        for future in futures: future.cancel()
        raise
    finally:
        pool.shutdown(wait=True,cancel_futures=True)
    for name in cfg['cases']:
        chunks=[bases[name]['orientation_coefficients'][:,-1,0],extras[name]['orientation_coefficients'][:,-1,0]]
        coarse=[]
        for i in range(64,512):
            with np.load(out/'checkpoints'/name/f'{i:04d}.npz') as row:
                chunks.append(row['coefficients'][None]);coarse.append(row['coarse_b_coefficients'])
        save_npz(out/f'{name}_orientations.npz',orientation_coefficients=np.concatenate(chunks),
            new_coarse_b_coefficients=np.stack(coarse),radii=bases[name]['radii'],axis=bases[name]['axis'],
            rotation_bank_1=bank[0],rotation_bank_2=bank[1],fingerprint=np.asarray(fingerprint))
    for group in ('input_sha256','reused_dense_provenance','reused_dense_checkpoint_sha256'):
        if any(sha(p)!=h for p,h in identity[group].items()):raise ValueError(f'Input changed: {group}')
    for group in ('core_sha256','additional_kernel_sha256','driver_dependencies_sha256'):
        if any(sha(ROOT/p)!=h for p,h in identity[group].items()):raise ValueError(f'Code changed: {group}')
    if sha(ROOT/'upc_deformation/orientation_refinement.py')!=identity['kernel_sha256']:
        raise ValueError('Base kernel changed')
    manifest.update(status='completed',completed_new_pairs=896,current_session_seconds=time.monotonic()-started,
        output_sha256={f'{n}_orientations.npz':sha(out/f'{n}_orientations.npz') for n in cfg['cases']},
        driver_sha256=sha(__file__),inputs_unchanged=True)
    save_json(out/'production_manifest.json',manifest)
    log(f'Completed512pairs per deformation: {out}')


if __name__=='__main__':main()
