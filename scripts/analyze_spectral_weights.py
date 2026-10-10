#!/usr/bin/env python3
"""Analyze broad-annulus spectral weights from the original saved panel d."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/mpl_cache'))
import numpy as np
import yaml
from upc_deformation.angular_cache import load_case
from upc_deformation.spectral_weights import analyze_spectral_weights
from upc_deformation.spectral_weight_figures import render_spectral_weights


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items() if not k.startswith('_')}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def save_json(path,value):
    path.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def arrays(value,prefix=''):
    result={}
    for key,item in value.items():
        name=f'{prefix}__{key}' if prefix else key
        if isinstance(item,np.ndarray):result[name]=item
        elif isinstance(item,dict):result.update(arrays(item,name))
    return result


def export_tables(data,out):
    with (out/'metrics.csv').open('w') as stream:
        writer=csv.writer(stream)
        keys=['fractions','fractions_orientation_se','ratios','ratios_orientation_se',
              'fraction_relative_change','fraction_relative_change_orientation_se',
              'ratio_relative_change','ratio_relative_change_orientation_se']
        columns=[]
        for key in keys:
            columns.extend(f'{key}_{label}' for label in (['L','M','H'] if key.startswith('fraction') else ['R1','R2']))
        writer.writerow(['case','beta2','beta3','skin_fm','n_orientations','window_yield',*columns])
        for name,row in data['cases'].items():
            meta=row['case']
            writer.writerow([name,meta['beta2'],meta['beta3'],meta['skin_fm'],row['n_orientations'],row['window_yield'],
                             *[v for key in keys for v in row[key]]])
    with (out/'spectra.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['case','pT_GeV','window_normalized_dN_dpT','spectrum_ratio_to_spherical'])
        for name,row in data['cases'].items():
            writer.writerows([name,q,p,r] for q,p,r in zip(data['radii'],row['spectrum'],row['spectrum_ratio_to_spherical']))
    with (out/'boundary_variations.csv').open('w') as stream:
        writer=csv.writer(stream)
        writer.writerow(['variation','case','pmin','edge1','edge2','pmax','f_L','f_M','f_H','R1','R2','relative_change_R1','relative_change_R2'])
        for variant in data['boundary_variants']:
            for name,row in variant['cases'].items():
                writer.writerow([variant['label'],name,*variant['edges_GeV'],*row['fractions'],*row['ratios'],*row['ratio_relative_change']])


def write_report(data,figures,out):
    e=data['edges_GeV']
    lines=['# Panel d 的宽环带谱权重','',
           '本轮直接统计原panel d线性强度。b积分为12–60 fm、步长2 fm、|y|<1；所有形变样本各32对取向。球形与纯中子皮为确定性计算。',
           f'共同示例区间：L=[{e[0]:g},{e[1]:g})，M=[{e[1]:g},{e[2]:g})，H=[{e[2]:g},{e[3]:g}] GeV/c。边界可调整，本轮未优化窗口。','',
           r'$N_k=\sum_{i\in k}I_i\,\Delta q_x\Delta q_y$，$f_k=N_k/(N_L+N_M+N_H)$。',
           r'$R_1=N_M/N_L$，$R_2=N_H/N_M$。相对变化统一定义为$R_a/R_{a,\rm sph}-1$（或$f_k/f_{k,\rm sph}-1$）。',
           '份额只在共同选定窗口内归一化；它不是整个0–0.17动量范围的份额，也不是绝对截面。比值先平均原始取向强度再计算，不能平均每个取向的比值。',
           '曲线显示同一Cartesian图圆周取样后的径向概率密度，包含2πpT因子；主环带统计直接使用原始像素，无显示亮度阈值。','',
           '| 样本 | L份额 | M份额 | H份额 | R1=M/L | R2=H/M |',
           '|---|---:|---:|---:|---:|---:|']
    for name,row in data['cases'].items():
        fs=[f'{100*x:.4f}%' for x in row['fractions']]
        rs=[f'{x:.6f}' for x in row['ratios']]
        lines.append('| '+name+' | '+' | '.join(fs+rs)+' |')
    lines+=['','| 样本 | R1相对球形变化 | R2相对球形变化 |','|---|---:|---:|']
    for name,row in data['cases'].items():
        if name=='spherical':continue
        values=[]
        for value,error in zip(row['ratio_relative_change'],row['ratio_relative_change_orientation_se']):
            values.append(f'{100*value:+.3f}%'+(f' ± {100*error:.3f}个百分点' if row['n_orientations']>1 else ''))
        lines.append('| '+name+' | '+' | '.join(values)+' |')
    lines+=['','误差棒为有限取向积分的1标准误，协方差保留在数组和summary.json中。球形与中子皮没有取向误差，不代表无前向模型、b求积或实验误差。图中的中间参数点与端点均来自已有计算，连接线仅引导视线，不代表连续参数拟合。',
            '本轮还预先指定了少量边界扰动：两个内部边界分别单独移动±0.005 GeV/c，最低动量阈值单独移动±0.01 GeV/c（仅保留合法区间）。没有根据扫描挑选最大分离点。结果在boundary_variations.csv及summary.json中。','',
            '| 端点样本 | 边界扰动下R1相对变化范围 | 边界扰动下R2相对变化范围 |',
            '|---|---:|---:|']
    for name in ('beta2','beta3','skin'):
        changes=np.asarray([v['cases'][name]['ratio_relative_change'] for v in data['boundary_variants']])
        if changes.size:
            ranges=[f'{100*lo:+.3f}% 至 {100*hi:+.3f}%' for lo,hi in zip(changes.min(0),changes.max(0))]
            lines.append('| '+name+' | '+' | '.join(ranges)+' |')
    diagnostics=data['diagnostics']
    lines+=['','以上范围只描述这几组边界的变化，不是系统误差带。改变最低动量阈值不会改变H/M，因为该比值只使用中间与外环产额。',
            '数值核对：原始取向强度重建并平均后，与存档panel d的最大相对误差为'
            f"{diagnostics['max_orientation_image_relative_closure']:.3e}；连续绘图谱积分与直接像素积分的最大差别为"
            f"{100*diagnostics['max_abs_curve_integral_minus_one']:.4f}%。主观测量采用像素积分，未用曲线积分替代。",'']
    for file in figures['files']:
        if file.endswith('.png'):lines.extend([f'![{Path(file).stem}]({file})',''])
    lines+=['## 复现与改边界','', '```bash',f'cd {ROOT}',
            f'OPENBLAS_NUM_THREADS=1 /opt/anaconda3/bin/python3 scripts/analyze_spectral_weights.py --config {out/"config.yaml"}',
            '```','',
            '可以通过--edges pmin edge1 edge2 pmax指定四个边界，同时用--output选择新的结果目录；也可复制并修改配置文件。旧目录拒绝覆盖不同配置。',
            'metrics.csv保存份额、产额比和取向误差，spectra.csv保存绘图曲线，arrays.npz保存完整协方差，manifest.json保存输入和代码哈希。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/panel_d_spectral_weights.yaml')
    parser.add_argument('--edges',nargs=4,type=float)
    parser.add_argument('--output')
    args=parser.parse_args();cfg=yaml.safe_load(Path(args.config).read_text())
    if args.edges is not None:cfg['edges_GeV']=args.edges
    if args.output is not None:cfg['output']=args.output
    for key in ('source','output'):cfg[key]=str((ROOT/cfg[key]).resolve())
    source=Path(cfg['source']);out=Path(cfg['output'])
    if out==source or source in out.parents:raise ValueError('Use an output directory separate from the source')
    if cfg['stage']!='decay_by' or cfg['branch']!='full' or cfg['normalization']!='selected_window':raise ValueError('Wrong panel-d scope')
    study=yaml.safe_load((source/'config.yaml').read_text())['study']
    if (study['b_min_fm'],study['b_max_fm'],study['b_nodes'],study['ymax'])!=(12.,60.,25,1.):raise ValueError('Wrong original b12 configuration')
    inputs=[source/'config.yaml',source/'manifest.json']+[source/f'{name}.{ext}' for name in cfg['cases'] for ext in ('npz','json')]
    hashes={str(p):sha(p) for p in inputs}
    frozen=yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True)
    if (out/'config.yaml').exists() and (out/'config.yaml').read_text()!=frozen:raise ValueError('Use a new output directory for changed settings')
    results=[load_case(source,name) for name in cfg['cases']]
    data=analyze_spectral_weights(results,edges=cfg['edges_GeV'])
    out.mkdir(parents=True,exist_ok=True);(out/'config.yaml').write_text(frozen)
    figures=render_spectral_weights(data,out)
    save_json(out/'figures.json',figures)
    summary={k:v for k,v in data.items() if k!='radii'}
    summary['cases']={name:{k:v for k,v in row.items() if k not in ('spectrum','spectrum_ratio_to_spherical')} for name,row in data['cases'].items()}
    save_json(out/'summary.json',summary)
    np.savez_compressed(out/'arrays.npz',**arrays(data))
    export_tables(data,out);write_report(data,figures,out)
    if hashes!={str(p):sha(p) for p in inputs}:raise ValueError('Input cache changed')
    code=[Path(__file__),ROOT/'upc_deformation/spectral_weights.py',ROOT/'upc_deformation/spectral_weight_figures.py',ROOT/'upc_deformation/decay.py',ROOT/'upc_deformation/panel_d_ratio_figures.py']
    save_json(out/'manifest.json',{'status':'completed','input_sha256':hashes,'config_sha256':sha(out/'config.yaml'),
        'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in code},'edges_GeV':data['edges_GeV'],
        'estimator':'direct panel-d Cartesian pixel annuli; common-window-normalized fractions and ratios of mean intensities',
        'orientation_counts':{k:v['n_orientations'] for k,v in data['cases'].items()},
        'outputs_sha256':{p.name:sha(p) for p in out.iterdir() if p.suffix in ('.csv','.npz','.png','.pdf','.svg')}})
    print(f'Completed: {out}',flush=True)


if __name__=='__main__':main()
