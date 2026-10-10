#!/usr/bin/env python3
"""Analyze completed 512-pair panel-d coefficients without new forward calls."""
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
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import torch
import yaml
from upc_deformation.angular_cache import CORE_FILES, load_case
from upc_deformation.c2_convergence import (project_orientation_moments,
    summarize_orientation_moments, check_original_prefix)
from upc_deformation.c2_convergence_figures import render_c2_convergence

CASE_ORDER = ('spherical', 'beta2', 'beta3', 'skin')
POINTS = (.05, .10, .13, .15, .17)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict): return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    return value


def save_json(path, value):
    path.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def array_fields(value, prefix='', result=None):
    result = {} if result is None else result
    if isinstance(value, np.ndarray): result[prefix] = value
    elif isinstance(value, dict):
        for k, v in value.items(): array_fields(v, prefix+'__'+k if prefix else k, result)
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value): array_fields(v, prefix+'__'+str(i), result)
    return result


def validate_inputs(config_path):
    cfg = yaml.safe_load(config_path.read_text())
    for key in ('source', 'extension', 'original_curves', 'output'):
        cfg[key] = str((ROOT/cfg[key]).resolve())
    out, source = Path(cfg['output']), Path(cfg['source'])
    production = json.loads((out/'production_manifest.json').read_text())
    provenance = json.loads((out/'provenance.json').read_text())
    if production.get('status') != 'completed' or production.get('completed_new_pairs') != 896:
        raise ValueError('All production orientation checkpoints must be completed before analysis')
    if production['fingerprint'] != provenance['fingerprint'] or cfg != provenance['config']:
        raise ValueError('Analysis configuration or production identity changed')
    for path, checksum in provenance['input_sha256'].items():
        if sha(path) != checksum: raise ValueError(f'Original input changed: {path}')
    if {p: sha(ROOT/p) for p in CORE_FILES} != provenance['core_sha256']:
        raise ValueError('Original forward core changed')
    if sha(ROOT/'upc_deformation/orientation_refinement.py') != provenance['kernel_sha256']:
        raise ValueError('Production kernel changed')
    for path, checksum in provenance.get('additional_kernel_sha256', {}).items():
        if sha(ROOT/path) != checksum: raise ValueError(f'Additional production kernel changed: {path}')
    for group in ('reused_dense_provenance', 'reused_dense_checkpoint_sha256'):
        for path, checksum in provenance.get(group, {}).items():
            if sha(path) != checksum: raise ValueError(f'Reused dense checkpoint provenance changed: {path}')
    if sha(out/'orientation_banks.npz') != provenance['bank_sha256']:
        raise ValueError('Production orientation bank changed')
    inputs = dict(provenance['input_sha256'])
    for group in ('reused_dense_provenance', 'reused_dense_checkpoint_sha256'):
        inputs.update(provenance.get(group, {}))
    if (out/'kernel_validation.json').exists():
        inputs[str(out/'kernel_validation.json')] = sha(out/'kernel_validation.json')
    inputs.update({str(config_path): sha(config_path), str(out/'production_manifest.json'): sha(out/'production_manifest.json'),
                   str(out/'provenance.json'): sha(out/'provenance.json'), str(out/'orientation_banks.npz'): sha(out/'orientation_banks.npz')})
    for name in cfg['cases']:
        p = out/f'{name}_orientations.npz'
        if sha(p) != production['output_sha256'][p.name]: raise ValueError('Completed coefficients changed')
        inputs[str(p)] = sha(p)
    if (cfg['n_orientations'] != 512 or cfg['cases'] != ['beta2', 'beta3']
            or cfg['prefix_counts'] != [32, 64, 128, 256, 512]):
        raise ValueError('Expected two 512-pair deformation endpoints and registered nested prefixes')
    return cfg, production, provenance, inputs


def estimates(row):
    yield 'final', str(row['n_orientations']), row
    for label, value in row.get('nested', {}).items(): yield 'nested', label, value
    for i, value in enumerate(row.get('halves', [])): yield 'half', str(i+1), value
    for i, value in enumerate(row.get('blocks', [])): yield 'block', str(i+1), value


def export_tables(data, out):
    base = data['cases']['spherical']
    with (out/'curves.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['case','pT_GeV','C2','orientation_se','delta_C2','C2_original32','orientation_se_original32','delta_C2_original32','S2','S2_orientation_se','half_difference','half_difference_se','half_difference_z_pointwise'])
        for name, row in data['cases'].items():
            for i, q in enumerate(data['q']):
                writer.writerow([name,q,row['C2'][i],row['orientation_se'][i],row['C2'][i]-base['C2'][i],row['C2_original32'][i],row['orientation_se_original32'][i],row['C2_original32'][i]-base['C2_original32'][i],row['S2'][i],row['S2_orientation_se'][i],*[row.get(k, np.zeros_like(data['q']))[i] for k in ('half_difference','half_difference_se','half_difference_z_pointwise')]])
    with (out/'convergence.csv').open('w') as stream:
        writer=csv.writer(stream)
        writer.writerow(['case','estimate_type','estimate_label','sample_start','sample_stop','n_orientations','pT_GeV','C2','orientation_se','delta_C2'])
        for name, row in data['cases'].items():
            for kind,label,entry in estimates(row):
                for i,q in enumerate(data['q']):
                    writer.writerow([name,kind,label,entry['sample_start'],entry['sample_stop'],entry['n_orientations'],q,entry['C2'][i],entry['orientation_se'][i],entry['C2'][i]-base['C2'][i]])
    with (out/'bin_metrics.csv').open('w') as stream:
        writer=csv.writer(stream)
        writer.writerow(['case','estimate_type','estimate_label','n_orientations','bin','q_low_GeV','q_high_GeV','C2','orientation_se','delta_C2','S2','S2_orientation_se','yield_fraction_in_selected_window'])
        for name,row in data['cases'].items():
            for kind,label,entry in estimates(row):
                for i,bin_name in enumerate(('L','M','H')):
                    b=entry['bins']
                    writer.writerow([name,kind,label,entry['n_orientations'],bin_name,*data['edges_GeV'][i:i+2],b['C2'][i],b['orientation_se'][i],b['C2'][i]-base['bins']['C2'][i],b['S2'][i],b['S2_orientation_se'][i],b['fractions'][i]])


def point_checks(data):
    base=data['cases']['spherical']
    values=[]
    for name,row in data['cases'].items():
        for p in POINTS:
            i=int(np.argmin(abs(data['q']-p)))
            before_se=float(row['orientation_se_original32'][i]);after_se=float(row['orientation_se'][i])
            values.append({'case':name,'pT_GeV':float(data['q'][i]),
                'C2_original32':float(row['C2_original32'][i]),'C2':float(row['C2'][i]),
                'delta_C2_original32':float(row['C2_original32'][i]-base['C2_original32'][i]),
                'delta_C2':float(row['C2'][i]-base['C2'][i]),
                'orientation_se_original32':before_se,'orientation_se':after_se,
                'SE512_over_SE32':after_se/before_se if before_se>0 else None,
                'center_change_from_original32':float(row['C2'][i]-row['C2_original32'][i]),
                'half_difference_z_pointwise':float(row['half_difference_z_pointwise'][i]) if 'halves' in row else None})
    return values


def write_report(data, points, figures, out):
    lines=['# 512对取向的 C2 曲线与积分收敛检查','',
        '两种形变均包含512对取向：原始32对、已有独立32对扩展，以及14组新的32对。原始样本逐项保留，32、64、128、256、512是实际嵌套前缀，不能视为互相独立的结果。球形及球形中子皮沿用确定性计算。', '',
        '每对取向先通过原有 decay_axis_image 重建425×425的 Cartesian panel d，再用双线性插值在341个半径、每圈720个角度上取样。C2为所有取向原始角矩分子的平均除以原始强度的平均，没有先逐取向归一化。没有新的物理近似或模型训练。', '',
        r'记每个取向的角矩分子与强度为 $N_i,D_i$，则 $C_2=\bar N/\bar D$；1SE为 $(N_i-C_2D_i)/\bar D$ 的样本标准差除以 $\sqrt n$。完整协方差保留了分子分母相关性以及不同动量点之间的相关性。', '',
        '计算范围保持 b=12–60 fm、步长2 fm、|y|<1和原网格。图中阴影仅是有限取向积分的点态1SE，不包含实验事件数、探测器、b网格或模型误差。512对不是预先保证收敛的阈值。球形/中子皮零取向SE也不表示无其他误差。', '',
        data['config']['numerical_implementation'], '',
        '## 固定动量点的前后比较','',
        '以下差值均相对确定性球形参照，数字为 10³×ΔC2 或 10³×SE。','',
        '| 结构 | pT (GeV) | 原32 ΔC2 ± SE | 512 ΔC2 ± SE | SE512/SE32 | 两半差值/合成SE |',
        '|---|---:|---:|---:|---:|---:|']
    for p in points:
        if p['case'] not in ('beta2','beta3'):continue
        lines.append(f"| {p['case']} | {p['pT_GeV']:.3f} | {p['delta_C2_original32']*1e3:+.4f} ± {p['orientation_se_original32']*1e3:.4f} | {p['delta_C2']*1e3:+.4f} ± {p['orientation_se']*1e3:.4f} | {p['SE512_over_SE32']:.3f} | {p['half_difference_z_pointwise']:+.3f} |")
    lines.extend(['','两个256对半样本互不重叠；另保存四个128对块。两半的差值/合成SE只用于已列动量点的描述性比较。没有把动量扫描的最大值作为整体显著性检验，也没有把相关的嵌套前缀当作独立样本。图中SE(32)/4只是1/√n缩放参考线，并未强制512的误差达到该数值。', '',
        '## 有限动量区间与数据','',
        '附带的 L/M/H 统计采用原先用于谱权重的共同边界 0.03、0.08、0.13、0.17 GeV。它们仅作固定区间诊断；从原始Cartesian像素直接按强度加权计算，未重新挑选或优化C2窗口。', '',
        '- `curves.csv`：512曲线、原32曲线、SE及两半诊断。',
        '- `convergence.csv`：所有真实嵌套前缀、两半与四块的曲线。',
        '- `bin_metrics.csv`：固定L/M/H区间的C2、S2、SE和事件份额。',
        '- `arrays.npz`：原始逐取向分子/分母、C2完整协方差和所有诊断数组。',
        '- `panel_d_images.npz`：当前各案例的平均原始强度图、在pT≤0.17内归一化后的图及平均系数；形变图使用512对。',
        '- `qa.json`与`analysis_manifest.json`：原32闭合检查和输入/代码哈希。',''])
    for p in figures['files']:
        if p.endswith('.png'): lines.extend([f'![{Path(p).stem}]({p})',''])
    command='OPENBLAS_NUM_THREADS=1 /opt/anaconda3/bin/python3 scripts/analyze_c2_convergence.py --config '+data['config']['analysis_config']
    lines.extend(['复现分析（仅在生产全部完成后运行）：','```bash',command,'```',''])
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/panel_d_c2_convergence.yaml')
    args=parser.parse_args()
    config_path=(ROOT/args.config).resolve()
    cfg,production,provenance,inputs=validate_inputs(config_path)
    out,source=Path(cfg['output']),Path(cfg['source'])
    torch.set_num_threads(1)
    with np.load(cfg['original_curves'],allow_pickle=False) as z: original={k:z[k] for k in z.files}
    q=np.linspace(0.,.17,cfg['q_points']);alpha=np.arange(cfg['angular_points'])*2*np.pi/cfg['angular_points']
    if not np.array_equal(q,original['pT_GeV']):raise ValueError('Original and current radial grids differ')
    with np.load(out/'orientation_banks.npz',allow_pickle=False) as z: banks={k:z[k] for k in z.files}
    data={'q':q,'angles':alpha,'edges_GeV':np.asarray(cfg['diagnostic_edges_GeV']),
          'cases':{},'config':{'b_min_fm':12.,'b_max_fm':60.,'b_step_fm':2.,'ymax':1.,'n_orientations':512,
              'q_points':len(q),'angular_points':len(alpha),'prefix_counts':cfg['prefix_counts'],
              'analysis_config':str(config_path.relative_to(ROOT)),
              'numerical_implementation':('原始64对缓存和已验证的完整FFT检查点保留不变，其余取向使用CZT计算原FFT网格所需的频率点；形成Stokes之后再使用原插值权重。两种数值实现的回归检查及逐检查点来源保存在生产记录中。' if provenance.get('additional_kernel_sha256') else '新增取向采用经旧缓存回归验证的完整FFT实现。'),
              'primary_estimator':'ratio of raw orientation means after exact Cartesian panel-d sampling',
              'uncertainty_scope':'pointwise orientation Monte Carlo 1SE; no simultaneous interval or detector claim'}}
    qa={'status':'passed','source_inputs_unchanged':True,'original_prefix_tolerance':1e-12,'original_prefix_closure':{},
        'independent_half_diagnostic':'pointwise descriptive z only; no global maximum-z hypothesis test',
        'nested_prefixes_independent':False,'visual_QA':'reported separately by figure_qa.json'}
    image_arrays={}
    for name in CASE_ORDER:
        old=load_case(source,name)
        if name in cfg['cases']:
            with np.load(out/f'{name}_orientations.npz',allow_pickle=False) as z: payload={k:z[k] for k in z.files}
            if str(payload['fingerprint'])!=production['fingerprint']:raise ValueError('Case fingerprint differs')
            coeff=payload['orientation_coefficients']
            if coeff.shape!=(512,3,len(old['radii'])) or not np.array_equal(coeff[:32],old['orientation_coefficients'][:,-1,0]):raise ValueError('Original32 coefficients were not preserved exactly')
            for key in ('axis','radii'):
                if not np.array_equal(payload[key],old[key]):raise ValueError('Archived projection axes changed')
            for key in ('rotation_bank_1','rotation_bank_2'):
                if not np.array_equal(payload[key],banks[key]):raise ValueError('Case rotation bank differs from registered bank')
        else:coeff=old['orientation_coefficients'][:,-1,0]
        print(f'Project {name}: {len(coeff)} retained orientations on original Cartesian grid',flush=True)
        moments=project_orientation_moments(coeff,old['radii'],old['axis'],q,alpha,data['edges_GeV'])
        entry=summarize_orientation_moments(moments,cfg['prefix_counts'] if len(coeff)>1 else ())
        qa['original_prefix_closure'][name]=check_original_prefix(entry,original,name)
        entry.update(case=old['case'],C2_original32=original[name+'_C2'],orientation_se_original32=original[name+'_orientation_se'])
        for key,value in moments.items():
            if key.startswith('mean_') or key=='normalization_yield':
                image_arrays[f'{name}__{key}']=np.asarray(value)
            else:entry[key]=value
        data['cases'][name]=entry
        image_arrays['axis']=old['axis'];image_arrays['radii']=old['radii']
        print(f'Validated original32 closure {name}: {qa["original_prefix_closure"][name]}',flush=True)
    base=data['cases']['spherical']
    for row in data['cases'].values():
        row['delta_C2']=row['C2']-base['C2'];row['bins']['delta_C2']=row['bins']['C2']-base['bins']['C2']
    if any(sha(p)!=h for p,h in inputs.items()):raise ValueError('Input changed during analysis')
    if {p:sha(ROOT/p) for p in CORE_FILES}!=provenance['core_sha256']:raise ValueError('Forward core changed during analysis')
    if any(sha(ROOT/p)!=h for p,h in provenance.get('additional_kernel_sha256',{}).items()):raise ValueError('Additional kernel changed during analysis')
    np.savez_compressed(out/'arrays.npz',**array_fields(data))
    np.savez_compressed(out/'panel_d_images.npz',**image_arrays)
    export_tables(data,out)
    figures=render_c2_convergence(data,out)
    save_json(out/'figures.json',figures)
    points=point_checks(data)
    summary={'config':data['config'],'edges_GeV':data['edges_GeV'],'point_checks':points,
             'cases':{name:{'case':row['case'],'n_orientations':row['n_orientations'],'bins':row['bins'],
                  'minimum_orientation_image':row['minimum_orientation_image']} for name,row in data['cases'].items()},
             'interpretation':'Only orientation quadrature refined. Inspect actual error contraction and disjoint samples; 512 does not itself certify convergence.'}
    save_json(out/'summary.json',summary)
    write_report(data,points,figures,out)
    save_json(out/'qa.json',qa)
    code=['scripts/analyze_c2_convergence.py','upc_deformation/c2_convergence.py','upc_deformation/c2_convergence_figures.py','upc_deformation/decay.py','upc_deformation/spectral_weights.py']
    outputs=['arrays.npz','panel_d_images.npz','curves.csv','convergence.csv','bin_metrics.csv','figures.json','summary.json','report.md','qa.json']+[Path(p).name for p in figures['files']]
    save_json(out/'analysis_manifest.json',{'status':'completed','production_fingerprint':production['fingerprint'],
        'input_sha256':inputs,'production_kernel_sha256':provenance['kernel_sha256'],
        'additional_production_kernel_sha256':provenance.get('additional_kernel_sha256',{}),
        'code_sha256':{p:sha(ROOT/p) for p in code},'output_sha256':{p:sha(out/p) for p in outputs},
        'source_inputs_unchanged':True,'scope':'existing512rawcoefficients→sameCartesianpixelprojection; no newforward calls'})
    print(f'Completed: {out}',flush=True)


if __name__=='__main__':main()
