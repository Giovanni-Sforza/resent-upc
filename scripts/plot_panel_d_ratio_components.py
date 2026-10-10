#!/usr/bin/env python3
"""Decompose saved panel-d shape ratios; no forward or response simulation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'outputs/mpl_cache'))
import numpy as np
import yaml
from upc_deformation.angular_cache import load_case
from upc_deformation.panel_d_ratios import decompose_panel_d
from upc_deformation.panel_d_ratio_figures import render_panel_d_ratios


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(value):
    if isinstance(value, dict):return {k: jsonable(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):return jsonable(value.tolist())
    if isinstance(value, np.generic):return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):return None
    return value


def save_json(path, value):
    path.write_text(json.dumps(jsonable(value), indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def write_outputs(data, destination, config, inputs, figures):
    arrays = {k:v for k,v in data.items() if isinstance(v,np.ndarray)}
    for name,case in data['cases'].items():
        arrays.update({f'{name}__{k}':v for k,v in case.items() if isinstance(v,np.ndarray)})
    np.savez_compressed(destination/'decomposition.npz', **arrays)
    with (destination/'radial_ratios.csv').open('w') as stream:
        writer=csv.writer(stream)
        writer.writerow(['case','q_GeV','radial_ratio','radial_change_percent'])
        for name,case in data['cases'].items():
            for q,ratio in zip(data['radii'],case['radial_ratio']):
                writer.writerow([name,q,ratio,100*(ratio-1)])
    summaries={}
    for name,case in data['cases'].items():
        mask=case['valid_cart']
        summaries[name]={
            'case':case['case'],'n_orientations':case['n_orientations'],
            **{key+'_range': [float(np.nanmin(case[key][mask])),float(np.nanmax(case[key][mask]))]
               for key in ['full_ratio_cart','radial_map','angular_map']},
        }
    save_json(destination/'summary.json', {'cases':summaries,'diagnostics':data['diagnostics']})
    save_json(destination/'figures.json', figures)
    code=[Path(__file__),ROOT/'upc_deformation/panel_d_ratios.py',ROOT/'upc_deformation/panel_d_ratio_figures.py']
    manifest={
        'status':'completed','source':config['source'],'inputs_sha256':inputs,
        'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in code},
        'config_sha256':sha(destination/'config.yaml'),
        'input_scope':config['input_scope'],
        'actual_orientation_counts':{name:c['n_orientations'] for name,c in data['cases'].items()},
        'calculation':'Bilinear circle sampling of the saved linear Cartesian panel-d density; no new physical assumptions, forward calls or response MC',
        'replay_command':f'cd {ROOT}\nOPENBLAS_NUM_THREADS=1 /opt/anaconda3/bin/python3 scripts/plot_panel_d_ratio_components.py --config {destination/"config.yaml"}',
        'outputs_sha256':{str(p.name):sha(p) for p in [destination/'decomposition.npz',destination/'radial_ratios.csv',destination/'summary.json']},
    }
    save_json(destination/'manifest.json', manifest)
    lines=['# Panel d 比值中的径向变化与条件角向变化','',
        '本轮只分解此前结构比值图的同一组panel d，不重新计算产生过程或径迹响应。β₂、β₃为分开的纯形变端点，各32对取向；不是后来64取向结果，也不是y=0的混合大形变图。',
        '统一设置为b=12–60 fm、步长2 fm、|y|<1，完整相干信号，x轴沿轻子参考方向k。原始线性密度分别在q<0.17 GeV/c内归一化；所有比值都是形状比，不是绝对产额比。','',
        '## 两类变化的定义','',
        r'$J_\theta(q)=(2\pi)^{-1}\int d\alpha\,I_\theta(q,\alpha)$。径向形状比为$R_{\rm rad}=J_\theta/J_0$，表示同一半径上整体增减。这里的J为角平均密度；改用角积分时共同因子2π约去，两个分解比值不变。',
        r'条件角向比为$R_{\rm ang}=(I_\theta/J_\theta)/(I_0/J_0)$，表示除去径向共同变化后同半径的方向性明暗变化。完整比值满足$R_{\rm full}=R_{\rm rad}R_{\rm ang}$。',
        '角积分保留分布权重；不能以二维比值对角度的均匀平均代替径向比。圆周取样来自已保存Cartesian数组的双线性插值，属于既有离散分布的后处理。取样点增加不等于提升原前向网格的物理精度。','',
        '## 读图','',
        '颜色显示100×(比值−1)，红色为增加、蓝色为减少。每类图三种结构共用色标，径向和角向图的色标范围分别标注；不截掉有效区域的极值。',
        '灰区为q>0.17或球形参照密度低于其图峰值0.1%的显示掩膜；该掩膜不是新观测量窗口。条件角向图的(q,alpha)视图直接显示固定半径上的方向变化。',
        '这里显示32取向的中心估计，没有绘制取向标准误。因此细小角向纹理可能含取向积分残差，不能仅凭颜色认定它已统计分辨。高q仍有此前已识别的b求积敏感性。',
        '径向分量不能自动等同中子皮，角向分量也不能自动等同两源干涉；二者是分布形状的分解，具体机制仍需联系原形貌和交叉项对照。','',
        '| 样本 | 原完整比值范围 | 径向分量范围 | 条件角向分量范围 |',
        '|---|---:|---:|---:|']
    for name,s in summaries.items():
        ranges=['–'.join(f'{v:.5g}' for v in s[key+'_range']) for key in ['full_ratio_cart','radial_map','angular_map']]
        lines.append('| '+name+' | '+' | '.join(ranges)+' |')
    lines += ['', '范围只针对共同Cartesian显示掩膜；极值不代表该处产额或参数提取灵敏度。','']
    for file in figures['files']:
        if file.endswith('.png'):lines += [f'![{Path(file).stem}]({file})','']
    lines += ['## 复现与检查','', '```bash',manifest['replay_command'],'```','',
        '线性二维数组及极坐标数组在decomposition.npz；径向曲线在radial_ratios.csv；summary.json记录分解闭合与加权角平均检查，manifest.json记录输入和代码哈希。',
        '旧缓存、旧图及论文正文均未写入。']
    (destination/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/panel_d_ratio_components.yaml')
    args=parser.parse_args()
    config=yaml.safe_load(Path(args.config).read_text())
    source=(ROOT/config['source']).resolve();destination=(ROOT/config['output']).resolve()
    if destination==source or source in destination.parents:
        raise ValueError('Use a separate output directory')
    if config['stage']!='decay_by' or config['branch']!='full':
        raise ValueError('This figure uses panel d full coherent intensity')
    input_paths=[source/'config.yaml',source/'manifest.json']
    input_paths += [source/f'{name}.{suffix}' for name in config['cases'] for suffix in ['json','npz']]
    inputs={str(p):sha(p) for p in input_paths}
    results=[load_case(source,name) for name in config['cases']]
    for result in results:
        if result['stage_names'][3]!='decay_by':raise ValueError('Wrong panel-d stage')
    source_config=yaml.safe_load((source/'config.yaml').read_text())['study']
    if (source_config['b_min_fm'],source_config['b_max_fm'],source_config['b_nodes'],source_config['ymax'])!=(12.,60.,25,1.):
        raise ValueError('Input differs from the discussed b12 panel-d convention')
    destination.mkdir(parents=True,exist_ok=True)
    config.update(source=str(source),output=str(destination))
    frozen=yaml.safe_dump(config,sort_keys=False,allow_unicode=True)
    if (destination/'config.yaml').exists() and (destination/'config.yaml').read_text()!=frozen:
        raise ValueError('Choose another output directory for different settings')
    (destination/'config.yaml').write_text(frozen)
    data=decompose_panel_d(results,**{k:config[k] for k in ['qmax','n_radii','n_angles','floor_fraction']})
    figures=render_panel_d_ratios(data,destination)
    if inputs!={str(p):sha(p) for p in input_paths}:raise ValueError('Input cache changed during analysis')
    write_outputs(data,destination,config,inputs,figures)
    print(f'Completed panel-d decomposition: {destination}',flush=True)


if __name__=='__main__':main()
