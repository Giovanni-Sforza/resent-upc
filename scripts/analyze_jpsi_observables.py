#!/usr/bin/env python3
"""Reproducible mechanism bridge; all outputs remain separate from the manuscript."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/mpl_cache'))
import numpy as np
import torch
import yaml
from upc_deformation.observables import calculate_case,analyze_results,METRIC_NAMES
from upc_deformation.observable_response import (make_common_random,sample_decay_pairs,
    apply_track_response,summarize_observables,paired_contrast)
from upc_deformation.observable_figures import render_observable_figures
from upc_deformation.observable_diagnostics import mechanism_diagnostics
from upc_deformation.observable_dark_regions import dark_region_diagnostics


def jsonable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [jsonable(v) for v in value]
    return value


def save_json(path,value):
    path.write_text(json.dumps(jsonable(value),indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def load_case(destination,name):
    result=json.loads((destination/f'{name}.json').read_text())
    with np.load(destination/f'{name}.npz') as arrays:
        result.update({k:arrays[k] for k in arrays.files})
    return result


def response_screen(results,config,log):
    cfg=config['study']
    common=make_common_random(cfg['response_events'],cfg['response_seed'])
    baseline=next(r for r in results if r['case']['name']=='spherical')
    baseline_summaries={}
    for result in [baseline]+[r for r in results if r is not baseline]:
        pairs=sample_decay_pairs(result['coefficients'][-1,0],result['radii'],common,
                                 mass_gev=config['physics']['physical_constants']['Mjpsi'])
        result['response']=[]
        reconstructed=[]
        for sigma in cfg['fractional_resolutions']:
            reco=apply_track_response(pairs,common,fractional_sigma=sigma,angular_sigma=cfg['angular_sigma_rad'])
            reconstructed.append(reco)
            summary=summarize_observables(reco,q0=cfg['q0'],qmax=cfg['qmax'])
            if result is baseline:baseline_summaries[sigma]=summary
            contrast=paired_contrast(summary,baseline_summaries[sigma])
            row={'fractional_sigma':sigma,'values':summary.values.tolist(),
                 'mc_se':summary.mc_se.tolist(),'event_covariance':summary.event_covariance.tolist(),
                 'delta_vs_spherical':contrast['delta'],'delta_mc_se':contrast['mc_se'],
                 'selection_fraction':float(summary.selected_fraction),
                 'conditional_se':{str(n):summary.conditional_se(n).tolist() for n in cfg['conditional_event_counts']}}
            result['response'].append(row)
        # Propagate finite orientation integration through the same response.
        # q is sampled from the ensemble A0; reweight each orientation by its
        # unnormalized polar/azimuth density, retaining orientation yields.
        q=pairs.q_truth;alpha=pairs.alpha_truth;u=pairs.polar_cosine
        a0=np.interp(q,result['radii'],result['coefficients'][-1,0,0])
        factor=(1-u*u)/(1+u*u)
        c2=np.cos(2*alpha);s2=np.sin(2*alpha)
        numerator=[];denominator=[]
        for orientation in result['orientation_coefficients'][:,-1,0]:
            a,c,d=[np.interp(q,result['radii'],v) for v in orientation]
            w=np.divide(a-factor*(c*c2+d*s2),a0,out=np.zeros_like(a),where=a0>0)
            nums=[];dens=[]
            for reco in reconstructed:
                selected=reco.q<cfg['qmax'];ws=w[selected];qr=reco.q[selected]
                dens.append(ws.sum())
                nums.append([ws[qr<cfg['q0']].sum(),ws@qr,ws@np.cos(2*reco.alpha[selected])])
            numerator.append(nums);denominator.append(dens)
        numerator=np.asarray(numerator);denominator=np.asarray(denominator)
        center=numerator.mean(0)/denominator.mean(0)[:,None]
        influence=(numerator-denominator[...,None]*center)/denominator.mean(0)[None,:,None]
        ori_se=np.std(influence,axis=0,ddof=1)/np.sqrt(len(influence)) if len(influence)>1 else np.zeros_like(center)
        for ri,row in enumerate(result['response']):
            row['orientation_se']=ori_se[ri].tolist()
            row['delta_numerical_se']=np.hypot(ori_se[ri],row['delta_mc_se']).tolist()
            row['orientation_reweight_closure']=(center[ri]-row['values']).tolist()
        log(f'Response complete: {result["case"]["name"]}',flush=True)


def write_products(results,config,dest,manifest):
    fields=['case','stage','branch',*METRIC_NAMES,'q50','area_halfmax','fringe_low5','dark_fixed_mean','yield']
    with (dest/'observables.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for result in results:
            for si,stage in enumerate(result['stage_names']):
                for bi,branch in enumerate(['full','source_self']):
                    writer.writerow({'case':result['case']['name'],'stage':stage,'branch':branch,
                                     **{k:result['metrics'][si][bi].get(k) for k in fields[3:]}})
    summary={'manifest':manifest,'config':config,'cases':[]}
    for result in results:
        summary['cases'].append({k:v for k,v in result.items() if not isinstance(v,np.ndarray)})
    figures=render_observable_figures(results,config,dest)
    summary['figures']=figures
    diagnostics=mechanism_diagnostics(results,config)
    save_json(dest/'mechanism_diagnostics.json',diagnostics)
    summary['mechanism_diagnostics']=diagnostics
    dark_regions=dark_region_diagnostics(results,config)
    save_json(dest/'dark_region_diagnostics.json',dark_regions)
    summary['dark_region_diagnostics']=dark_regions
    save_json(dest/'observable_summary.json',summary)
    lines=['# Fig. 7 机制到可测投影：计算结果','',
        '本轮直接实施机制桥接与候选检验；未追溯旧稿生成链、未修改正文、未训练网络。',
        '稿件基准：用户指定的 arXiv 版本。计算使用本项目现有 physical 单位前向。','',
        '## 样本与定义','',
        f"- b={config['study']['b_min_fm']:g}–{config['study']['b_max_fm']:g} fm，2πb db 权重；|y|<{config['study']['ymax']:g}，按原始产生强度积分。",
        '- 这是有限 b 硬区间诊断样本；没有中子标记、平滑强子生存概率或特定实验接受度，不称为 inclusive 实验预测。',
        '- 横轴平行轻子相对横动量；Born level、横向偏振、小 qT/M、完整衰变极角接受度。',
        '- source_self 仅去掉两个发射方向之间的交叉项，保留各源内部相干衍射；它不是非相干核破裂样本。',
        '- 单独提供原经验非相干模板的各向同性背景检验；无跨生成器检验。',
        f"- 预先固定 qmax={config['study']['qmax']:g} GeV/c、q0={config['study']['q0']:g} GeV/c。没有根据验证点重选窗口。",
        '- A1/2 是超过半峰高的二维面积；fringe_low5 保留最暗5%定义；dark_fixed_mean 使用球形参照固定的暗区。新 Fdip 及 mean_q 的联系由机制对照检验，而不预设成立。',
        '- 所有结构图先对原始强度平均，最后统一范围归一化；完整/去交叉项形状各自归一化，原始 full/self 比值另存。','',
        '## 最终 b、y 平均下的观测量','',
        '| 样本 | Fdip | mean qT [GeV/c] | C2 | Fdip取向数值误差 | mean q取向数值误差 | 非相干占比 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        m=r['metrics'][-1][0];e=r['orientation_se'][-1][0]
        lines.append(f"| {r['case']['name']} | {m['F_dip']:.7g} | {m['mean_q']:.7g} | {m['C2']:.7g} | {e[0]:.2g} | {e[1]:.2g} | {r['background_fraction']:.3%} |")
    lines += ['', '## 暗区位置核查', '',
              '将最暗5%重新计算到每一阶段，会改变所选的动量位置。下表针对当前球形诊断图，不能据此重新评价原 Fig. 7；像素比例与产额比例分开报告。', '',
              '| 阶段 | 暗区q最小值 | 暗区q中位数 | 暗区像素落在q<q0的比例 | 暗区像素落在q>0.9qmax的比例 | 暗区占分析产额 |',
              '|---|---:|---:|---:|---:|---:|']
    for row in diagnostics['dark_region_location']:
        lines.append(f"| {row['stage']} | {row['q_quantiles_GeV'][0]:.5g} | {row['q_quantiles_GeV'][2]:.5g} | {row['mask_pixel_fraction_below_q0']:.4%} | {row['mask_pixel_fraction_above_0p9_qmax']:.4%} | {row['fraction_of_analysis_yield']:.4%} |")
    lines += ['', '外围暗区本身不是问题：原稿“工字型”暗条纹包括高pT的上下横臂，因而大q位置不能被直接归因为衍射尾部，也不能据此淘汰干涉观测量。这里的半径表只描述支撑位置。',
              '各阶段重新取最暗5%不等于同一组像素的严格输运。新二维图同时保留方向结构与原始强度的两源交叉项对照；中心Fdip只是候选之一，不要求最终干涉指标必须位于中心，或去交叉项后结构响应必须完全消失。']
    lines += ['', '## 暗区的原始交叉项与结构变化', '',
              '令 S=I_self、X=I_full−I_self，使用相同原始归一化。K_R=∫R X/∫R S 表示区域净交叉项；A_R=∫R |X|/∫R S 检查区域内部正负抵消。这里先做取向平均再取绝对值，A_R的抽样误差未估计。它们是理论机制诊断，不是实验中可单独选择的事件样本。', '',
              '下表区域由每阶段球形完整分布在q≤qmax内的最暗5%像素定义，并在该阶段跨结构点冻结；outer_dark仅为其中q>0.9qmax的子集。二维交叉项图显示整个已计算方形，参考圆外的高q横臂也保留，但未因此扩大观测量积分窗口。', '',
              '| 球形阶段 | 区域 | K_R | A_R | 区域完整产额占比 |',
              '|---|---|---:|---:|---:|']
    for row in dark_regions['records']:
        if row['case']!='spherical' or row['status']!='available':continue
        lines.append(f"| {row['stage']} | {row['region']} | {row['K_R']:.6g} | {row['A_R']:.6g} | {row['region_full_yield_fraction']:.4%} |")
    lines += ['', '最终b/y平均后的结构变化按同一个Y0（球形自项在分析圆盘内的产额）归一化，因而ΔS/Y0+ΔX/Y0=Δfull/Y0严格成立。表中±为配对取向积分SE；不包含求积系统误差或实验误差。所有阶段与8个样本的完整数值保存在dark_region_diagnostics.json。', '',
              '| 样本 | 区域 | ΔS/Y0 | ΔX/Y0 | Δfull/Y0 |',
              '|---|---|---:|---:|---:|']
    for row in dark_regions['records']:
        if row['stage']!='decay_by' or row['case'] not in ['beta2','beta3','skin'] or row['status']!='available':continue
        cells=[f"{row[f'delta_N_{key}_over_Y0']:.5g} ± {row['orientation_se'][f'delta_N_{key}_over_Y0']:.2g}" for key in ['self','cross','full']]
        lines.append('| '+row['case']+' | '+row['region']+' | '+' | '.join(cells)+' |')
    baseline=next(r for r in results if r['case']['name']=='spherical')
    lines += ['', '## 当前候选的数值采样分辨状态', '',
              '这里只检查结构差是否超过3倍取向积分标准误，不以此判定物理机制成立或实验可测；空间与求积系统变化另行检查。', '',
              '| 样本 | Fdip | mean qT | C2 |', '|---|---|---|---|']
    for r in results:
        if r is baseline:continue
        flags=[]
        for i,key in enumerate(METRIC_NAMES):
            delta=r['metrics'][-1][0][key]-baseline['metrics'][-1][0][key]
            se=r['orientation_se'][-1][0][i]
            flags.append('取向采样可分辨' if abs(delta)>3*se else '取向采样尚未分辨')
        lines.append('| '+r['case']['name']+' | '+' | '.join(flags)+' |')
    lines += ['', '## 干涉机制对照', '',
              '下表为最终分布相对球形参照的结构差。交互对照 = Δfull−Δself；它衡量移除两源交叉项如何改变结构响应，不能解释成非线性指标的唯一加法分解。', '',
              '| 样本 | 指标 | Δfull | Δself | 交互对照 | 交互对照取向SE |',
              '|---|---|---:|---:|---:|---:|']
    for record in diagnostics['mechanism_contrasts']:
        if record['stage']!='decay_by':continue
        for i,key in enumerate(METRIC_NAMES):
            lines.append(f"| {record['case']} | {key} | {record['delta_full'][i]:.5g} | {record['delta_self'][i]:.5g} | {record['interference_interaction_contrast'][i]:.5g} | {record['interaction_orientation_se'][i]:.2g} |")
    lines += ['', '## 中心窗口依赖', '',
              f"q0=0.01、0.02、0.03、0.05为预列窗口；0.005另标为原稿窗口参照，采用本轮b={config['study']['b_min_fm']:g} fm起的物理设置，不是旧稿生产链复现。主窗口始终为0.02，没有按下表改选。", '',
              '| 样本 | q0 [GeV/c] | 完整Fdip结构差 | 取向SE | 去交叉项结构差 | 配对交互差 | 交互SE |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in diagnostics['window_contrasts']:
        if row['stage']!='decay_by' or row['case'] not in ['beta2','beta3','skin']:continue
        lines.append(f"| {row['case']} | {row['q0_GeV']:.3g} | {row['delta_full']:.5g} | {row['full_orientation_se']:.2g} | {row['delta_self']:.5g} | {row['interaction_contrast']:.5g} | {row['interaction_orientation_se']:.2g} |")
    lines += ['', '## 几何选择与保留验证点', '',
              '下列子区间从同一组原始 b/y 节点重积分，只检查已计算范围内的选择依赖；它们不代表实验中子类别，也不检验 b>60 fm 尾部。', '',
              '| 样本 | b区间 [fm] | Fdip | mean qT [GeV/c] | C2 |',
              '|---|---|---:|---:|---:|']
    for row in diagnostics['geometry_subwindows']:
        if row['case'] not in ['spherical','beta2','beta3','skin']:continue
        lines.append(f"| {row['case']} | {row['b_range_fm'][0]:g}–{row['b_range_fm'][1]:g} | {row['F_dip']:.7g} | {row['mean_q']:.7g} | {row['C2']:.7g} |")
    mixed=diagnostics['mixed_validation']
    if mixed is not None:
        lines += ['', '混合点没有参与选窗。这里检验单参数中点变化能否近似相加；不把相加预设成物理规律，也不将此表作为参数反演精度。', '',
                  '| 指标 | 混合点实际差值 | 三个单参数差值之和 | 非加性残差 | 残差配对取向SE |',
                  '|---|---:|---:|---:|---:|']
        for i,key in enumerate(METRIC_NAMES):
            se=mixed['nonadditivity_orientation_se'][i]
            se_text=f'{se:.3g}' if se is not None else '未估计'
            lines.append(f"| {key} | {mixed['actual_difference'][i]:.6g} | {mixed['sum_of_separate_midpoint_differences'][i]:.6g} | {mixed['nonadditivity'][i]:.6g} | {se_text} |")
    lines += ['', '## 各向同性经验非相干背景检验', '',
              '下表是在每个结构样本自身的模型背景产额下重新归一化得到；响应与条件事件数图仅使用相干信号。', '',
              '| 样本 | 背景占比 | total Fdip | total mean qT [GeV/c] | total C2 |',
              '|---|---:|---:|---:|---:|']
    for r in results:
        m=r['background_metrics']
        lines.append(f"| {r['case']['name']} | {r['background_fraction']:.3%} | {m['F_dip']:.7g} | {m['mean_q']:.7g} | {m['C2']:.7g} |")
    lines += ['', '## 为什么中心占比也会响应衍射尺度', '',
        r'作为尺度变化的示意，若径向强度满足 $I_s(q)=s^{-2}I_0(q/s)$，并记 $G(x)=2\pi\int_0^x qI_0(q)\,dq$，则固定窗口占比为 $F_s=G(q_0/s)/G(q_{\max}/s)$。因此，仅改变衍射包络的宽度，也会改变固定中心窗口的产额；这不需要新增干涉填充。',
        r'在半高区域没有被分析边界截断时，$\sqrt{A_{1/2}}$ 随 $s$ 缩放；全范围平均动量也随 $s$ 缩放。固定 $q_{\max}$ 下平均动量还有截断修正，故本轮用实际数值验证它与半高面积的关系。这个尺度示意不是将完整UPC模型强行分解为两个独立因子。']
    lines+=['','## 解释与统计约定','',
        '- 取向数值误差是随机取向积分的 delta-method 标准误，不能解释为实验误差或模型不确定度。',
        '- 图中组合数值误差仅包含取向和响应抽样；b、y、空间网格等系统数值变化单独报告。有限取向尚未分辨的结果保留为候选失败或待检验记录。',
        '- 三种轨迹动量误差0、0.5%、1%均为未标定 toy 假设，忽略材料辐射、接受度及完整重建；同一对轻子用于重新计算 qT 和 kT，保留相关误差。',
        '- 响应模拟包含分析窗口外的真值迁入；真值支持延伸至约0.283 GeV/c，未模拟更远尾部。',
        '- 条件统计按选中样本N计算。数值MC误差与该统计误差分开；不预测特定运行产额。',
        '- 条件3σ事件数采用单一观测样本与精确模型模板比较；不包含理论、背景拟合、参数边缘化或实验系统误差。']
    convergence=dest.parent/'convergence'/'report.md'
    if convergence.exists():
        lines += ['', f'历史b20定向求积检查见[加密报告]({convergence})：它只覆盖原20–60 fm范围的前四组β2取向；不能认证新增12–20 fm区域或完整取向总体。当前QA会核查其适用范围，不自动视为本轮收敛证据。']
    lines += ['', '## 图','']
    for name in ['baseline_bridge','dark_region_audit','structure_bridge_ratios','mechanism_radial','feature_translation','response_screening']:
        if str(dest/f'{name}.png') in figures['files']:lines += [f'![{name}]({dest/name}.png)','']
    lines+=['## 复现','', '```bash',manifest['replay_command'],'```','',
            '原始数值：每个样本 npz；节点与取向均保留。observables.csv 为线性数值；observable_summary.json 包含参数、近似、误差和图表索引。',
            f'保存数据的闭合检查与数值警告见[QA]({dest/"qa.json"})；代码测试与成图核查记录见[执行规划第14节]({ROOT/"docs/XU11124_Fig7之后观测量修订执行规划.md"})。']
    (dest/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/jpsi_observables_revision.yaml')
    p.add_argument('--output',default='outputs/jpsi_observables_revision/main')
    p.add_argument('--quick',action='store_true');p.add_argument('--render-only',action='store_true')
    p.add_argument('--cases',nargs='+');p.add_argument('--orientations',type=int)
    p.add_argument('--skip-response',action='store_true')
    p.add_argument('--extend-from',help='Reuse a validated higher-b cache and calculate only the additional lower-b segment')
    args=p.parse_args()
    if args.render_only and (args.quick or args.orientations is not None or args.cases or args.extend_from):
        p.error('--render-only uses the complete frozen configuration; do not override cases, grids or orientation count')
    dest=Path(args.output).resolve();dest.mkdir(parents=True,exist_ok=True)
    previous_manifest=json.loads((dest/'manifest.json').read_text()) if (dest/'manifest.json').exists() else None
    config=yaml.safe_load((dest/'config.yaml' if args.render_only else Path(args.config)).read_text())
    source_directory=Path(args.extend_from).resolve() if args.extend_from else None
    source_config=yaml.safe_load((source_directory/'config.yaml').read_text()) if source_directory else None
    if source_directory==dest:raise ValueError('B-window extension must use a new output directory')
    cfg=config['study']
    if args.orientations:cfg['n_orientations']=args.orientations
    if args.cases:cfg['cases']=[c for c in cfg['cases'] if c['name'] in args.cases]
    if not any(c['name']=='spherical' for c in cfg['cases']):raise ValueError('spherical baseline required')
    if args.quick and not args.render_only:
        config['physics']['grid_parameters']={'WS':{'N':24,'L':24.},'T_prim':{'N':24,'L':24.},'output':{'N':96,'L':160.}}
        config['physics']['output_settings'].update(safe_padding_factor=2,target_size_q=101)
        config['physics']['incoherent_quadrature'].update(n_r=64,n_phi=64)
        cfg.update(b_nodes=5,n_y=3,n_orientations=2,n_radial=151,n_phi=64,response_events=50000)
    torch.set_num_threads(cfg['threads'])
    frozen=yaml.safe_dump(config,sort_keys=False,allow_unicode=True)
    if (dest/'config.yaml').exists() and not args.render_only and (dest/'config.yaml').read_text()!=frozen:
        raise ValueError('Output directory has a different configuration; choose a new run directory')
    (dest/'config.yaml').write_text(frozen)
    log_file=(dest/'run.log').open('a',buffering=1)
    def log(message,**kwargs):print(message,**kwargs);log_file.write(str(message)+'\n')
    source_files=[Path(__file__),*sorted((ROOT/'upc_deformation').rglob('*.py'))]
    hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files}
    calculation_files=['upc_deformation/observables.py','upc_deformation/source_amplitudes.py',
                       'upc_deformation/forward.py','upc_deformation/decay.py','upc_deformation/vendor/upc_forward.py',
                       'upc_deformation/observable_b_extension.py']
    calculation_hash=hashlib.sha256(json.dumps({k:hashes[k] for k in calculation_files},sort_keys=True).encode()).hexdigest()
    started=time.monotonic()
    manifest={'status':'running','quick':args.quick,'run_mode':'render_only' if args.render_only else 'calculate_or_resume',
              'source_hashes':hashes,'command':sys.argv,
              'replay_command':f"cd {ROOT}\n/opt/anaconda3/bin/python3 scripts/analyze_jpsi_observables.py --config {dest/'config.yaml'} --output {dest}",
              'completed_cases':[]}
    if source_directory:
        manifest['extension_source']=str(source_directory)
        manifest['replay_command']+=f' --extend-from {source_directory}'
    if args.render_only and previous_manifest is not None:
        manifest['original_execution_elapsed_seconds']=previous_manifest.get('original_execution_elapsed_seconds',previous_manifest.get('elapsed_seconds'))
        if previous_manifest.get('extension_source'):
            manifest['extension_source']=previous_manifest['extension_source']
            manifest['replay_command']+=f" --extend-from {previous_manifest['extension_source']}"
    save_json(dest/'manifest.json',manifest)
    results=[]
    for case in cfg['cases']:
        name=case['name']
        if (dest/f'{name}.npz').exists() and (dest/f'{name}.json').exists():
            result=load_case(dest,name);log(f'Reusing completed cache: {name}')
            if not args.render_only and result.get('calculation_hash')!=calculation_hash:
                raise ValueError(f'Cached calculation code differs for {name}; use a new run directory or --render-only for archived data')
        elif args.render_only:raise FileNotFoundError(f'Missing cache for {name}')
        else:
            if source_directory:
                from upc_deformation.observable_b_extension import extend_case_b_window
                old=load_case(source_directory,name)
                old.update(cache_config=source_config,cache_source=str(source_directory/f'{name}.npz'))
                result=extend_case_b_window(config,case,old,log)
            else:
                result=calculate_case(config,case,log)
            result['calculation_hash']=calculation_hash
            tmp=dest/f'{name}.partial.npz'
            np.savez_compressed(tmp,**{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
            tmp.replace(dest/f'{name}.npz')
            save_json(dest/f'{name}.json',{k:v for k,v in result.items() if not isinstance(v,np.ndarray)})
        results.append(result);manifest['completed_cases'].append(name);save_json(dest/'manifest.json',manifest)
    analyze_results(results,config)
    if not args.skip_response:response_screen(results,config,log)
    manifest.update(status='completed',elapsed_seconds=time.monotonic()-started,
                    cached_case_compute_seconds=sum(r['elapsed_seconds'] for r in results))
    write_products(results,config,dest,manifest)
    save_json(dest/'manifest.json',manifest)
    log(f'Completed: {dest} in {manifest["elapsed_seconds"]:.1f}s')


if __name__=='__main__':main()
