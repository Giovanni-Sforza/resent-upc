#!/usr/bin/env python3
"""Resume a 512-pair panel-d orientation integral with immutable checkpoints."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import torch
import yaml
from upc_deformation.angular_cache import CORE_FILES, load_case
from upc_deformation.vendor.upc_forward import UPCForwardModel


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict): return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    return value


def save_json(path, value):
    temp = path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def save_npz(path, **values):
    temp = path.with_suffix('.partial.npz')
    np.savez_compressed(temp, **values)
    temp.replace(path)


def settings(path):
    cfg = yaml.safe_load(Path(path).read_text())
    for key in ('source', 'extension', 'original_curves', 'output'):
        cfg[key] = str((ROOT/cfg[key]).resolve())
    if cfg['n_orientations'] != 512 or cfg['new_block_size'] != 32 or len(cfg['new_block_seeds']) != 14:
        raise ValueError('This study requires original64 plus14 independent32-pair blocks')
    if cfg['cases'] != ['beta2', 'beta3']:
        raise ValueError('Only the original two pure deformation endpoints are refined')
    return cfg


def prepare(cfg):
    source, extension, out = [Path(cfg[k]) for k in ('source', 'extension', 'output')]
    if out in (source, extension) or source in out.parents or extension in out.parents:
        raise ValueError('Use a separate output tree')
    original = yaml.safe_load((source/'config.yaml').read_text())
    study = original['study']
    expected = {'b_min_fm': 12., 'b_max_fm': 60., 'b_nodes': 25, 'n_y': 5,
                'ymax': 1., 'n_radial': 757, 'n_phi': 256, 'qmax': .17,
                'longitudinal_units': 'physical', 'n_orientations': 32}
    if any(study[k] != v for k, v in expected.items()):
        raise ValueError('Original forward scope changed')
    kernel_path = ROOT/'upc_deformation/orientation_refinement.py'
    core_hashes = {p: sha(ROOT/p) for p in CORE_FILES}
    manifests = [json.loads((p/'manifest.json').read_text()) for p in (source, extension)]
    for manifest in manifests:
        if manifest['status'] != 'completed' or any(manifest['source_hashes'].get(k) != v for k, v in core_hashes.items()):
            raise ValueError('Archived forward provenance does not match the current core')
    if manifests[1]['source'] != str(source) or manifests[1]['orientations_per_case'] != 32:
        raise ValueError('Invalid existing extension')
    inputs = [source/'config.yaml', source/'manifest.json', extension/'config.yaml', extension/'manifest.json', Path(cfg['original_curves'])]
    bases, extras = {}, {}
    for name in ('spherical', 'beta2', 'beta3', 'skin'):
        bases[name] = load_case(source, name)
        inputs += [source/f'{name}.json', source/f'{name}.npz']
        if name in cfg['cases']:
            extras[name] = load_case(extension, name)
            inputs += [extension/f'{name}.json', extension/f'{name}.npz']
            if extras[name]['shard_fingerprint'] != manifests[1]['fingerprint']:
                raise ValueError('Extension case fingerprint changed')
            for key in ('case', 'physics', 'density', 'stage_names'):
                if bases[name][key] != extras[name][key]: raise ValueError(f'{name}: {key} mismatch')
            for key in ('axis', 'radii', 'b_values', 'b_weights', 'y_values', 'y_weights'):
                if not np.array_equal(bases[name][key], extras[name][key]): raise ValueError(f'{name}: {key} mismatch')
    old_seeds = {int(study['orientation_seed']), int(study['orientation_seed'])+1,
                 int(manifests[1]['seed']), int(manifests[1]['seed'])+1}
    used = set(old_seeds)
    for seed in cfg['new_block_seeds']:
        if seed in used or seed+1 in used: raise ValueError('Overlapping nuclear random streams')
        used.update((seed, seed+1))
    bank = []
    for number in (1, 2):
        key = f'rotation_bank_{number}'
        if not np.array_equal(bases['beta2'][key], bases['beta3'][key]) or not np.array_equal(extras['beta2'][key], extras['beta3'][key]):
            raise ValueError('Original structures do not share paired orientation banks')
        blocks = [bases['beta2'][key], extras['beta2'][key]]
        for seed in cfg['new_block_seeds']:
            generator = torch.Generator(device='cpu').manual_seed(seed+number-1)
            blocks.append(UPCForwardModel._sample_uniform_rotations(1, 32, torch.device('cpu'), torch.float32, generator)[0].numpy())
        bank.append(np.concatenate(blocks))
    pairs = np.concatenate([v.reshape(512, -1) for v in bank], axis=1)
    if len(np.unique(pairs, axis=0)) != 512: raise ValueError('Duplicate orientation pairs')
    out.mkdir(parents=True, exist_ok=True)
    frozen = yaml.safe_dump(cfg, sort_keys=False)
    if (out/'config.yaml').exists() and (out/'config.yaml').read_text() != frozen:
        raise ValueError('Different configuration requires a new output directory')
    (out/'config.yaml').write_text(frozen)
    banks_path = out/'orientation_banks.npz'
    if banks_path.exists():
        with np.load(banks_path) as saved:
            if any(not np.array_equal(saved[f'rotation_bank_{i+1}'], v) for i, v in enumerate(bank)):
                raise ValueError('Immutable orientation bank mismatch')
    else:
        save_npz(banks_path, rotation_bank_1=bank[0], rotation_bank_2=bank[1], new_block_seeds=cfg['new_block_seeds'])
    input_hashes = {str(p): sha(p) for p in inputs}
    identity = {'config': cfg, 'input_sha256': input_hashes, 'core_sha256': core_hashes,
                'kernel_sha256': sha(kernel_path), 'bank_sha256': sha(banks_path)}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return original, bases, extras, bank, identity, fingerprint


_WORK_CONFIG = None
_WORK_MODE = None
_REFINERS = {}


def initialize_worker(config, threads, mode):
    global _WORK_CONFIG, _WORK_MODE
    torch.set_num_threads(threads)
    _WORK_CONFIG, _WORK_MODE = config, mode


def calculate_one(name, index, rotation1, rotation2):
    from upc_deformation.orientation_refinement import PanelDOrientationRefiner
    if name not in _REFINERS:
        case = next(c for c in _WORK_CONFIG['study']['cases'] if c['name'] == name)
        _REFINERS[name] = PanelDOrientationRefiner(_WORK_CONFIG, case, fft_mode=_WORK_MODE)
    result = _REFINERS[name].calculate(torch.from_numpy(rotation1[None]), torch.from_numpy(rotation2[None]), return_nodes=False)
    return name, index, {k: result[k] for k in ('coefficients', 'coarse_b_coefficients', 'elapsed_seconds')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/panel_d_c2_convergence.yaml')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    cfg = settings(args.config)
    original, bases, extras, bank, identity, fingerprint = prepare(cfg)
    out = Path(cfg['output'])
    proof_path = out/'kernel_validation.json'
    if args.prepare_only:
        save_json(out/'provenance.json', {**identity, 'fingerprint': fingerprint})
        print(f'Prepared immutable512 orientation banks: {out}', flush=True)
        return
    proof = json.loads(proof_path.read_text())
    if (proof.get('status') != 'passed' or proof.get('kernel_sha256') != identity['kernel_sha256']
            or proof.get('fft_mode') != cfg['fft_mode'] or proof.get('threads_per_worker') != cfg['threads_per_worker']
            or proof.get('core_sha256') != identity['core_sha256']
            or proof.get('source_config_sha256') != sha(Path(cfg['source'])/'config.yaml')):
        raise ValueError('Validated current forward kernel is required before production')
    save_json(out/'provenance.json', {**identity, 'fingerprint': fingerprint})
    old_manifest = json.loads((out/'production_manifest.json').read_text()) if (out/'production_manifest.json').exists() else {}
    if old_manifest and old_manifest['fingerprint'] != fingerprint: raise ValueError('Production fingerprint changed')
    checkpoints = []
    missing = []
    for name in cfg['cases']:
        directory = out/'checkpoints'/name
        directory.mkdir(parents=True, exist_ok=True)
        for i in range(64, 512):
            path = directory/f'{i:04d}.npz'
            if path.exists():
                with np.load(path, allow_pickle=False) as data:
                    if str(data['fingerprint']) != fingerprint or not np.array_equal(data['rotation1'], bank[0][i]) or not np.array_equal(data['rotation2'], bank[1][i]):
                        raise ValueError(f'Invalid checkpoint {path}')
                    if data['coefficients'].shape != (3, 757) or not np.isfinite(data['coefficients']).all():
                        raise ValueError(f'Invalid coefficients {path}')
                checkpoints.append(path)
            else: missing.append((name, i))
    manifest = {'status': 'running', 'fingerprint': fingerprint, 'reused_pairs_per_case': 64,
                'target_pairs_per_case': 512, 'new_pairs_total': 896,
                'completed_new_pairs': len(checkpoints), 'workers': cfg['workers'],
                'threads_per_worker': cfg['threads_per_worker'], 'scope': 'Only orientation Monte Carlo refined; original physics and quadratures fixed'}
    save_json(out/'production_manifest.json', manifest)
    log_file = (out/'production.log').open('a', buffering=1)
    def log(message):
        print(message, flush=True)
        log_file.write(message+'\n')
    started = time.monotonic()
    log(f'Resume: {len(checkpoints)}/896 new orientation calculations available; {len(missing)} remaining.')
    pool = ProcessPoolExecutor(max_workers=cfg['workers'], initializer=initialize_worker,
                               initargs=(original, cfg['threads_per_worker'], cfg['fft_mode']))
    futures = {}
    pending = iter(sorted(missing, key=lambda row: (row[1], row[0])))
    def enqueue():
        try: name, i = next(pending)
        except StopIteration: return
        futures[pool.submit(calculate_one, name, i, bank[0][i], bank[1][i])] = (name, i)
    try:
        for _ in range(2*cfg['workers']): enqueue()
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                name, i, result = future.result()
                path = out/'checkpoints'/name/f'{i:04d}.npz'
                save_npz(path, **result, rotation1=bank[0][i], rotation2=bank[1][i], fingerprint=np.asarray(fingerprint))
                checkpoints.append(path)
                manifest.update(completed_new_pairs=len(checkpoints), current_session_seconds=time.monotonic()-started)
                save_json(out/'production_manifest.json', manifest)
                if len(checkpoints) % 8 == 0 or len(checkpoints) == 896:
                    log(f'New pairs {len(checkpoints)}/896; last {name} index{i}; worker {result["elapsed_seconds"]:.2f}s; session {time.monotonic()-started:.1f}s')
                enqueue()
    except BaseException as error:
        manifest.update(status='interrupted', error=repr(error), current_session_seconds=time.monotonic()-started)
        save_json(out/'production_manifest.json', manifest)
        for future in futures: future.cancel()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    for name in cfg['cases']:
        chunks = [bases[name]['orientation_coefficients'][:, -1, 0], extras[name]['orientation_coefficients'][:, -1, 0]]
        coarse = []
        for i in range(64, 512):
            with np.load(out/'checkpoints'/name/f'{i:04d}.npz', allow_pickle=False) as row:
                chunks.append(row['coefficients'][None])
                coarse.append(row['coarse_b_coefficients'])
        save_npz(out/f'{name}_orientations.npz', orientation_coefficients=np.concatenate(chunks),
                 new_coarse_b_coefficients=np.stack(coarse), radii=bases[name]['radii'], axis=bases[name]['axis'],
                 rotation_bank_1=bank[0], rotation_bank_2=bank[1], fingerprint=np.asarray(fingerprint))
    if any(sha(path) != value for path, value in identity['input_sha256'].items()):
        raise ValueError('Original inputs changed during production')
    if sha(ROOT/'upc_deformation/orientation_refinement.py') != identity['kernel_sha256'] or any(sha(ROOT/path) != value for path, value in identity['core_sha256'].items()):
        raise ValueError('Forward implementation changed during production')
    manifest.update(status='completed', completed_new_pairs=896, current_session_seconds=time.monotonic()-started,
                    output_sha256={f'{name}_orientations.npz': sha(out/f'{name}_orientations.npz') for name in cfg['cases']},
                    driver_sha256=sha(__file__), inputs_unchanged=True)
    save_json(out/'production_manifest.json', manifest)
    log(f'Completed512pairs perdeformation: {out}')


if __name__ == '__main__':
    main()
