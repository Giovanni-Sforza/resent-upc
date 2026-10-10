#!/usr/bin/env python3
"""Create matched-orientation, one-change-at-a-time precision pilot configs."""
import argparse
import copy
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'configs/panel_d_precision_a100.yaml')
    p.add_argument('--output', type=Path, default=ROOT/'configs/panel_d_precision_pilots')
    p.add_argument('--n-orientations', type=int, default=64)
    a = p.parse_args()
    if a.n_orientations < 4 or a.n_orientations % 4:
        p.error('n-orientations must be positive, >=4 and divisible by four')
    base = yaml.safe_load(a.config.read_text())
    base['orientations']['n_orientations'] = a.n_orientations
    bank = base['orientations'].get('import_bank')
    if bank:
        base['orientations']['import_bank'] = os.path.relpath((a.config.parent/bank).resolve(), a.output.resolve())
    variants = {'baseline': (), 'float32': (('runtime', 'dtype', 'float32'),),
                'y9': (('study', 'n_y', 9),), 'phi512': (('study', 'n_phi', 512),),
                'radial1513': (('study', 'n_radial', 1513),),
                'b025': (('study', 'b_step_fm', .25),)}
    a.output.mkdir(parents=True, exist_ok=True)
    for name in (*variants, 'ws96', 'output480', 'padding8', 'pixels849'):
        cfg = copy.deepcopy(base)
        for section, key, value in variants.get(name, ()):
            cfg[section][key] = value
        if name == 'ws96':
            for grid in ('WS', 'T_prim'): cfg['physics']['grid_parameters'][grid]['N'] = 96
        if name == 'output480': cfg['physics']['grid_parameters']['output']['N'] = 480
        if name == 'padding8': cfg['physics']['output_settings']['safe_padding_factor'] = 8
        if name == 'pixels849': cfg['physics']['output_settings']['target_size_q'] = 849
        cfg['study']['b_nodes'] = round((cfg['study']['b_max_fm']-cfg['study']['b_min_fm'])/cfg['study']['b_step_fm'])+1
        cfg['runtime']['output'] = f'outputs/panel_d_precision/pilots_n{a.n_orientations}/{name}'
        path = a.output/f'{name}.yaml'
        text = yaml.safe_dump(cfg, sort_keys=False)
        if path.exists() and path.read_text() != text:
            raise ValueError(f'Existing different pilot configuration: {path}; choose another --output')
        path.write_text(text)
        print(path)


if __name__ == '__main__': main()
