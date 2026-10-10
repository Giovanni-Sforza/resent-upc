#!/usr/bin/env python3
"""Compare direct panel-d C2 in common bins selected from the skin ratio map."""
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
from upc_deformation.panel_d_three_bins import analyze_three_bins
from upc_deformation.panel_d_three_bin_figures import render_three_bins


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {key:clean(item) for key,item in value.items() if not key.startswith('_')}
    if isinstance(value, (list,tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, value):
    path.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def collect_arrays(value, prefix=''):
    result={}
    for key,item in value.items():
        name=f'{prefix}__{key}' if prefix else key
        if isinstance(item,np.ndarray):
            result[name]=item
        elif isinstance(item,dict):
            result.update(collect_arrays(item,name))
    return result


def report(data, figures, destination):
    edges=data['edges_GeV']
    lines=['# 中子皮角向换向划分的三个区间：C2比较','',
        '本次直接读取原panel d线性密度，按中子皮角向变化的两处换号位置固定区间，所有核结构共用同一组区间。',
        '数据沿用b=12–60 fm、步长2 fm、|y|<1；两种纯形变各32对取向。未重新运行前向模型。','',
        f'分界为{edges[1]:.9f}、{edges[2]:.9f} GeV/c。第一区间从0开始，最后止于{edges[-1]:g} GeV/c。这里的多位小数只记录当前离散图中的分界，不能视为物理精度。',
        '分界来自Delta C2(q)=C2_skin(q)-C2_spherical(q)的换号，不是C2本身过零。仅在寻找非平凡根时排除q<=0.01的原点数值零；第一区间的统计没有去掉低q。','',
        r'$C_{2,k}=\sum_{i\in k}I_i\cos(2\alpha_i)/\sum_{i\in k}I_i$。这是同一Cartesian像素网格上的强度加权和。原点角向权重为0；不对比值像素平均，不采用显示图的低亮度掩膜。',
        'Delta C2为同区间的结构值减去球形参照。所有数值均为无量纲；下表按10^-3显示。','',
        '| 样本 | 区间I C2 | 区间II C2 | 区间III C2 |',
        '|---|---:|---:|---:|']
    for name,case in data['cases'].items():
        entries=[f'{1e3*v:.4f} ± {1e3*s:.4f}' if case['n_orientations']>1 else f'{1e3*v:.4f}'
                 for v,s in zip(case['C2'],case['C2_orientation_se'])]
        lines.append('| '+name+' | '+' | '.join(entries)+' |')
    lines+=['','| 样本 | 区间I Delta C2 | 区间II Delta C2 | 区间III Delta C2 |',
            '|---|---:|---:|---:|']
    for name,case in data['cases'].items():
        if name=='spherical':continue
        entries=[f'{1e3*v:.4f} ± {1e3*s:.4f}' if case['n_orientations']>1 else f'{1e3*v:.4f}'
                 for v,s in zip(case['delta_C2'],case['delta_C2_orientation_se'])]
        lines.append('| '+name+' | '+' | '.join(entries)+' |')
    lines+=['','误差棒仅为有限取向积分的1标准误。球形和中子皮为确定性球形密度，未给取向误差棒；这不代表不存在b网格、前向模型或实验误差。','',
            '| 样本 | 区间I事件份额 | 区间II事件份额 | 区间III事件份额 |',
            '|---|---:|---:|---:|']
    for name,case in data['cases'].items():
        lines.append('| '+name+' | '+' | '.join(f'{100*v:.4f}%' for v in case['yield_fraction'])+' |')
    lines+=['','事件份额仅以当前pT<0.17范围内的相干信号归一化，不能替代实验预期事件数。区间III较窄、产额较低，角关联变化与统计可测性须分开判断。',
            '这三个区间由当前中子皮模板启发，属于探索性比较；未将分界优化为最佳灵敏度。第一段仍包含中心区域。','']
    for file in figures['files']:
        if file.endswith('.png'):lines+=[f'![三区间角关联]({file})','']
    lines+=['## 复现','', '```bash', f'cd {ROOT}',
            f'OPENBLAS_NUM_THREADS=1 /opt/anaconda3/bin/python3 scripts/analyze_panel_d_three_bins.py --config {destination/"config.yaml"}',
            '```','',
            'results.csv包含主结果；summary.json包含像素与极坐标核对以及取向平均还原检查；arrays.npz保存曲线及图像数据；manifest.json记录输入和代码哈希。']
    (destination/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/panel_d_three_bins.yaml')
    args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    for key in ['source','decomposition','output']:cfg[key]=str((ROOT/cfg[key]).resolve())
    source=Path(cfg['source']);destination=Path(cfg['output'])
    if destination==source or source in destination.parents:
        raise ValueError('Keep this analysis separate from its source')
    source_cfg=yaml.safe_load((source/'config.yaml').read_text())['study']
    if (source_cfg['b_min_fm'],source_cfg['b_max_fm'],source_cfg['b_nodes'],source_cfg['ymax'])!=(12.,60.,25,1.):
        raise ValueError('Wrong original b12 source')
    paths=[source/'config.yaml',source/'manifest.json',Path(cfg['decomposition'])]
    paths += [source/f'{name}.{suffix}' for name in cfg['cases'] for suffix in ['npz','json']]
    hashes={str(p):digest(p) for p in paths}
    results=[load_case(source,name) for name in cfg['cases']]
    with np.load(cfg['decomposition'],allow_pickle=False) as z:decomposition={k:z[k] for k in z.files}
    data=analyze_three_bins(results,decomposition,root_search_min=cfg['root_search_min'])
    if not np.isclose(data['edges_GeV'][-1],cfg['qmax']):raise ValueError('Wrong qmax')
    destination.mkdir(parents=True,exist_ok=True)
    frozen=yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True)
    target=destination/'config.yaml'
    if target.exists() and target.read_text()!=frozen:raise ValueError('Use a new directory for different settings')
    target.write_text(frozen)
    figures=render_three_bins(data,destination)
    save_json(destination/'figures.json',figures)
    save_json(destination/'summary.json',{k:data[k] for k in ['edges_GeV','config','cases','diagnostics']})
    np.savez_compressed(destination/'arrays.npz',**collect_arrays(data))
    with (destination/'results.csv').open('w') as stream:
        writer=csv.writer(stream)
        writer.writerow(['case','bin','lower_GeV','upper_GeV','C2','C2_orientation_se','delta_C2','delta_C2_orientation_se','yield_fraction','S2'])
        for name,case in data['cases'].items():
            for i in range(3):
                writer.writerow([name,i+1,*data['edges_GeV'][i:i+2],*[case[k][i] for k in ['C2','C2_orientation_se','delta_C2','delta_C2_orientation_se','yield_fraction','S2']]])
    report(data,figures,destination)
    if hashes!={str(p):digest(p) for p in paths}:raise ValueError('Source changed during analysis')
    code=[Path(__file__),ROOT/'upc_deformation/panel_d_three_bins.py',ROOT/'upc_deformation/panel_d_three_bin_figures.py',
          ROOT/'upc_deformation/panel_d_ratio_figures.py',ROOT/'upc_deformation/angular_observables.py',ROOT/'upc_deformation/decay.py']
    save_json(destination/'manifest.json',{'status':'completed','input_sha256':hashes,
        'code_sha256':{str(p.relative_to(ROOT)):digest(p) for p in code},
        'config_sha256':digest(target),'orientation_counts':{k:v['n_orientations'] for k,v in data['cases'].items()},
        'estimator':cfg['primary_estimator'],'boundary_source':cfg['boundary_source'],
        'output_sha256':{p.name:digest(p) for p in destination.iterdir() if p.suffix in ['.csv','.npz','.pdf','.svg','.png']}})
    print(f'Completed: {destination}',flush=True)


if __name__=='__main__':main()
