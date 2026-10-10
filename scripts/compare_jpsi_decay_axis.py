#!/usr/bin/env python3
"""Copy-independent, deterministic Torch UPC comparison and figure export."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs'/'mpl_cache'))
import numpy as np
import torch
import yaml
from upc_deformation.comparison import calculate_case, render_comparisons


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/jpsi_decay_axis.yaml')
    parser.add_argument('--output',default='outputs/jpsi_decay_axis_final')
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--only',nargs='+')
    parser.add_argument('--orientations',type=int)
    parser.add_argument('--output-grid',type=int)
    parser.add_argument('--density-grid',type=int)
    parser.add_argument('--field-extent',type=float)
    parser.add_argument('--b-nodes',type=int)
    parser.add_argument('--b-max',type=float)
    parser.add_argument('--longitudinal-units',choices=['physical','upstream'])
    args=parser.parse_args()
    config=yaml.safe_load(Path(args.config).read_text())
    cfg=config['comparison']; grids=config['physics']['grid_parameters']
    if args.quick:
        grids['WS']['N']=grids['T_prim']['N']=32
        grids['output']['N']=128
        config['physics']['output_settings']['target_size_q']=241
        cfg.update(n_orientations=8,n_phi=256,n_radial=401,b_nodes=9)
    if args.orientations: cfg['n_orientations']=args.orientations
    if args.output_grid: grids['output']['N']=args.output_grid
    if args.density_grid: grids['WS']['N']=grids['T_prim']['N']=args.density_grid
    if args.field_extent: grids['output']['L']=args.field_extent
    if args.b_nodes: cfg['b_nodes']=args.b_nodes
    if args.b_max: cfg['b_max_fm']=args.b_max
    if args.longitudinal_units: cfg['longitudinal_units']=args.longitudinal_units
    cfg.setdefault('longitudinal_units','upstream')
    if args.only: cfg['cases']=[case for case in cfg['cases'] if case['name'] in args.only]
    if not cfg['cases']: raise ValueError('No matching case')
    torch.set_num_threads(cfg['threads'])
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    (output/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    logpath=output/'run.log'
    def log(message,**kwargs):
        print(message,flush=True)
        with logpath.open('a') as f:f.write(message+'\n')
    log(f'Starting {len(cfg["cases"])} cases; Torch {torch.__version__}; device={cfg["device"]}; qL={cfg["longitudinal_units"]}')
    started=time.monotonic(); results=[]
    for case in cfg['cases']:
        result=calculate_case(config,case,log)
        np.savez_compressed(output/f'{case["name"]}.npz',**{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
        results.append(result)
    render_comparisons(results,cfg,output)
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    summary={'status':'completed','scope':'ideal_coherent_born_level_diagnostic','quick':args.quick,
             'display':{'q_color_scale':'log','kt_color_scale':cfg.get('kt_color_scale','log'),
                        'q_component_limit_GeV':config['physics']['output_settings']['p_limit_q']},
             'command':sys.argv,'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,
             'elapsed_seconds':time.monotonic()-started,'config':config,
             'provenance':json.loads((ROOT/'upc_deformation/vendor/provenance.json').read_text()),
             'cases':[{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in r.items()
                        if k in ('case','checks','yields','v2','elapsed_seconds','orientations')} for r in results],
             'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'upc_deformation').rglob('*.py'))},
             'output_hashes':{p.name:sha(p) for p in sorted(output.glob('*.npz'))}}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    report=['# J/psi 衰变轴比较结果','',
            '状态：实际 Torch 正演完成。理想全衰变角接受度、Born level、相干贡献；不是探测器级预测。',
            '主图 (b) 为固定 b 的轻子 kT 概率密度，使用独立坐标范围和色标；(c)、(d) 的 x 轴沿轻子衰变轴。perpendicular 图将该参考轴转 90 度。',
            f'纵向动量单位模式：{cfg["longitudinal_units"]}；原始版本及修正版本不可混作相同物理输入。',
            '核结构为显式示例参数；b 积分是登记的 hard window，不等同于已标定中子类别。','',
            '| 核结构 | 取向数 | 已知 b 轴 <cos2phi> | 固定 b 衰变轴 | b 积分衰变轴 |',
            '|---|---:|---:|---:|---:|']
    for r in results:
        report.append(f'| {r["case"]["name"]} | {r["orientations"]} | '+ ' | '.join(f'{v:.6g}' for v in r['v2'])+' |')
        log(r['case']['name']+' checks: '+json.dumps(r['checks']))
    report+=['','J/psi 图 (a,c,d) 在相同 pT<0.17 GeV 圆盘内归一化；原始强度与积分权重见 npz。',
             '(a,c,d) 色标统一；(b) 在整个物理 kT 圆盘归一化，包含相空间 Jacobian，使用有限像素积分和独立色标。',
             f'J/psi 的两个动量分量均覆盖正负 {config["physics"]["output_settings"]["p_limit_q"]:g} GeV/c；a/c/d 使用对数色标，b 的独立色标为 {cfg.get("kt_color_scale","log")}；没有逐图峰值归一化。',
             f'固定 b={cfg["fixed_b_fm"]:g} fm，积分区间 {cfg["b_min_fm"]:g}–{cfg["b_max_fm"]:g} fm，{cfg["b_nodes"]} 个节点。','',
             '数值检查详见 summary.json：原始 forward 对拍、正性、产额守恒、b 求积粗细对比、有限取向收敛诊断。',
             '取向 16/32 的差异是数值诊断，不是实验误差条。','',
             f'总时间：{summary["elapsed_seconds"]:.1f} 秒。']
    (output/'report.md').write_text('\n'.join(report)+'\n')
    log(f'Completed in {summary["elapsed_seconds"]:.1f}s; outputs: {output}')


if __name__=='__main__':main()
