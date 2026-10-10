#!/usr/bin/env python3
"""Package the portable precision workflow, without historical outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'dist/panel_d_precision_a100.tar.gz')
    a = p.parse_args()
    paths = list((ROOT/'upc_deformation').rglob('*.py'))
    paths += [ROOT/'upc_deformation/requirements.txt', ROOT/'upc_deformation/vendor/provenance.json', ROOT/'configs/panel_d_precision_a100.yaml',
              ROOT/'configs/assets/panel_d_orientation_prefix512.npz', ROOT/'configs/assets/panel_d_orientation_prefix512.json',
              ROOT/'docs/panel_d_precision_a100_runbook.md']
    paths += [ROOT/'scripts'/name for name in ('run_panel_d_precision.py', 'validate_panel_d_precision.py',
              'analyze_panel_d_precision.py', 'compare_panel_d_precision.py', 'make_panel_d_precision_configs.py', 'package_panel_d_precision.py')]
    paths += list((ROOT/'tests').glob('test_precision_*.py'))
    missing = [str(p) for p in paths if not p.is_file()]
    if missing: raise FileNotFoundError(missing)
    paths = sorted(set(paths))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {'files': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                'scope': 'Portable code/configuration/orientation-prefix bundle. No production results or server credentials.'}
    with tarfile.open(a.output, 'w:gz') as tar:
        for path in paths: tar.add(path, arcname='panel_d_precision/'+str(path.relative_to(ROOT)), recursive=False)
    manifest['archive_sha256'] = hashlib.sha256(a.output.read_bytes()).hexdigest()
    a.output.with_suffix(a.output.suffix+'.manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(a.output)


if __name__ == '__main__': main()
