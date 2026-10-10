#!/usr/bin/env python3
"""Reproduce the native-grid CZT gate against archived and dense references.

This performs three complete orientation calculations and writes a hash-bound
proof only after Cartesian C2, image and coefficient regressions and unit tests
pass. It does not modify either kernel, source data or production checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import torch
import yaml
from scipy.interpolate import RegularGridInterpolator

from upc_deformation.angular_cache import CORE_FILES, load_case
from upc_deformation.decay import decay_axis_image
from upc_deformation.orientation_czt_refinement import PanelDOrientationCZTRefiner


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='outputs/jpsi_observables_revision/b12_20261007')
    parser.add_argument('--dense-checkpoint-source', default='outputs/panel_d_c2_convergence/b12_512_20261010')
    parser.add_argument('--output', default='outputs/panel_d_c2_convergence/b12_512_czt_20261010')
    args = parser.parse_args()
    source, dense, out = [(ROOT/value).resolve() for value in (args.source, args.dense_checkpoint_source, args.output)]
    # A separate --output directory permits validation while production runs;
    # never rewrite the live run's proof file.
    manifest_path = out/'production_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get('status') == 'running':
        raise ValueError('Production is active in this output directory; validate into a separate --output directory')
    frozen_paths = {
        'kernel_sha256': ROOT/'upc_deformation/orientation_czt_refinement.py',
        'base_kernel_sha256': ROOT/'upc_deformation/orientation_refinement.py',
        'source_config_sha256': source/'config.yaml'}
    frozen_hashes = {key: sha(path) for key, path in frozen_paths.items()}
    config = yaml.safe_load((source/'config.yaml').read_text())
    source_manifest = json.loads((source/'manifest.json').read_text())
    dense_identity = json.loads((dense/'provenance.json').read_text())
    core = {p: sha(ROOT/p) for p in CORE_FILES}
    if any(source_manifest['source_hashes'].get(p) != h for p, h in core.items()):
        raise ValueError('Original forward core changed')
    if (dense_identity['core_sha256'] != core
            or dense_identity['kernel_sha256'] != sha(ROOT/'upc_deformation/orientation_refinement.py')):
        raise ValueError('Dense checkpoint core or kernel changed')
    torch.set_num_threads(2)
    q = np.linspace(0., .17, 341)
    angles = np.arange(720)*2*np.pi/720
    coords = np.stack((q[:, None]*np.cos(angles), q[:, None]*np.sin(angles)), axis=-1)
    results, references = {}, {}
    for name, label, index in (('beta3', 'beta3_archived0', 0),
                              ('spherical', 'spherical_archived0', 0),
                              ('beta2', 'beta2_new64_dense', 64)):
        old = load_case(source, name)
        calculator = PanelDOrientationCZTRefiner(config, old['case'])
        if index == 0:
            rotation1, rotation2 = old['rotation_bank_1'][0], old['rotation_bank_2'][0]
            target = old['orientation_coefficients'][0, -1, 0]
            reference_path = source/f'{name}.npz'
        else:
            reference_path = dense/'checkpoints'/name/f'{index:04d}.npz'
            with np.load(reference_path, allow_pickle=False) as saved:
                if str(saved['fingerprint']) != dense_identity['fingerprint']:
                    raise ValueError('Dense checkpoint fingerprint mismatch')
                rotation1, rotation2 = saved['rotation1'], saved['rotation2']
                target = saved['coefficients']
        references[str(reference_path)] = sha(reference_path)
        new = calculator.calculate(rotation1, rotation2, return_nodes=False)
        axis, radii = np.asarray(old['axis'], float), np.asarray(old['radii'], float)
        def project(coefficients):
            with torch.no_grad():
                image = decay_axis_image(torch.from_numpy(np.asarray(coefficients, float)),
                    torch.from_numpy(radii), torch.from_numpy(axis)).numpy()
            polar = RegularGridInterpolator((axis, axis), image, bounds_error=True)(coords)
            return image, (polar*np.cos(2*angles)).mean(1)/polar.mean(1)
        image, c2 = project(new['coefficients'])
        reference_image, reference_c2 = project(target)
        errors = {
            'max_absolute_cartesian_C2_error': float(np.max(abs(c2[1:]-reference_c2[1:]))),
            'global_relative_coefficient_error': float(np.max(abs(new['coefficients']-target))/np.max(abs(target))),
            'global_relative_image_error': float(np.max(abs(image-reference_image))/np.max(abs(reference_image)))}
        if (not np.isfinite(list(errors.values())).all()
                or errors['max_absolute_cartesian_C2_error'] >= 1e-6
                or errors['global_relative_coefficient_error'] >= 1e-5
                or errors['global_relative_image_error'] >= 1e-5):
            raise ValueError(f'CZT regression failed for {label}: {errors}')
        results[label] = {'passed': True, 'case': name, 'orientation_index': index,
            'reference_path': str(reference_path), 'elapsed_seconds': new['elapsed_seconds'],
            'preparation_seconds': new['preparation_seconds'], 'native_dft_side': new['native_dft_side'],
            'crop_side': new['czt_crop_side'], 'crop_bounds': new['czt_crop_bounds'], **errors}
        print(label, json.dumps(results[label]), flush=True)
    tests = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_orientation_czt_refinement.py', '-q'],
                           cwd=ROOT, capture_output=True, text=True)
    if tests.returncode:
        raise RuntimeError(tests.stdout+tests.stderr)
    if any(sha(p) != h for p, h in references.items()) or any(sha(ROOT/p) != h for p, h in core.items()):
        raise ValueError('Reference data or forward core changed during validation')
    if any(sha(path) != frozen_hashes[key] for key, path in frozen_paths.items()):
        raise ValueError('CZT/base kernel or source configuration changed during validation')
    proof = {'status': 'passed', **frozen_hashes,
        'core_sha256': core,
        'threads_per_worker': 2, 'fft_mode': 'combined', 'fourier_backend': 'native_grid_czt',
        'results': results,
        'tests': {'exit_code': tests.returncode,
                  'command': 'OPENBLAS_NUM_THREADS=1 /opt/anaconda3/bin/python3 -m pytest tests/test_orientation_czt_refinement.py -q',
                  'result': tests.stdout.strip()},
        'reference_sha256': references, 'source_manifest_sha256': sha(source/'manifest.json'),
        'dense_provenance_sha256': sha(dense/'provenance.json'),
        'validator_sha256': sha(__file__),
        'acceptance': {'absolute_cartesian_C2_error': 1e-6,
                       'global_relative_coefficient_error': 1e-5, 'global_relative_image_error': 1e-5},
        'scope': {'native_dft_grid_preserved': True, 'Stokes_before_interpolation': True,
                  'interpolation_weights': 'original global float32 bilinear corner coordinates and weights',
                  'CZT_precision': 'complex128 then complex64 before Stokes',
                  'padding_phase': 'common unit phase cancels all Stokes bilinears',
                  'source_physics_and_b_y_radial_phi_grids_unchanged': True,
                  'pointwise_curve': 'Cartesian decay_axis_image then 341x720 bilinear circle moment; pT=0 omitted'},
        'limitations': ['This is numerical implementation agreement, not an orientation, b-grid or detector convergence certificate.',
                       'Timing depends on concurrent machine load.']}
    out.mkdir(parents=True, exist_ok=True)
    path = out/'kernel_validation.json'
    temporary = out/'kernel_validation.partial.json'
    temporary.write_text(json.dumps(proof, indent=2)+'\n')
    temporary.replace(path)
    print(f'Passed: {path}', flush=True)


if __name__ == '__main__':
    main()
