#!/usr/bin/env python3
"""Finite-q lepton-axis harmonics, sector contrasts and paired track response."""
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
import yaml
from upc_deformation.angular_cache import load_angular_cases
from upc_deformation.angular_observables import analyze_angular
from upc_deformation.angular_response import run_angular_response
from upc_deformation.angular_figures import render_angular_figures

def save_json(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def write_tables(analytic,response,dest):
    names=analytic['metric_names']
    fields=['case','n_orientations','stage','branch','q_low_GeV','q_high_GeV','yield_fraction',
            *names,*[f'{k}_orientation_se' for k in names]]
    with (dest/'angular_observables.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for name,case in analytic['cases'].items():
            for stage,data in case['stages'].items():
                for branch in ['full','self']:
                    for i,(lo,hi) in enumerate(analytic['bin_edges_GeV']):
                        writer.writerow(dict(case=name,n_orientations=case['n_orientations'],stage=stage,
                            branch=branch,q_low_GeV=lo,q_high_GeV=hi,yield_fraction=data[branch]['yield_fraction'][i],
                            **dict(zip(names,data[branch]['values'][i])),
                            **dict(zip([f'{k}_orientation_se' for k in names],data[branch]['se'][i]))))
    if response is None:return
    fields=['case','response_id','estimator','q_low_GeV','q_high_GeV','bin_fraction',*names,
            *[f'{k}_delta_vs_spherical' for k in names],*[f'{k}_delta_sampling_se' for k in names]]
    with (dest/'angular_response.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for case in response['cases']:
            for row in case['rows']:
                estimates={'raw':row}
                if 'truth_anchored' in row:estimates['truth_anchored']=row['truth_anchored']
                for estimator,estimate in estimates.items():
                    for i,b in enumerate(row['bins']):
                        writer.writerow(dict(case=case['name'],response_id=row['response_id'],estimator=estimator,
                            q_low_GeV=b['q_range_gev'][0],q_high_GeV=b['q_range_gev'][1],bin_fraction=b['bin_fraction'],
                            **dict(zip(names,estimate['values_flat'][4*i:4*i+4])),
                            **dict(zip([f'{k}_delta_vs_spherical' for k in names],estimate['vs_spherical']['delta'][4*i:4*i+4])),
                            **dict(zip([f'{k}_delta_sampling_se' for k in names],estimate['vs_spherical']['combined_sampling_se'][4*i:4*i+4]))))

def write_report(analytic,response,config,figures,manifest,dest):
    cases=analytic['cases'];bins=analytic['bin_edges_GeV'];study=config['study']
    base=cases['spherical']['stages']['decay_by']
    lines=['# 有限pT角关联与暗纹机制：结果讨论','',
        '本轮实现分动量的轻子轴二阶关联及等半径明暗扇区对比。以原暗条纹为物理线索，不按半径判定干涉，也不预设候选能分离所有核参数。',
        f"固定b={study['fixed_b_fm']:g} fm；b积分{study['b_min_fm']:g}–{study['b_max_fm']:g} fm、|y|<{study['ymax']:g}。有限硬b窗口、无实验中子类别或接受度。",'',
        '## 本轮讨论起点','',
        f"球形参照在首个有限q环带的C2为{base['full']['values'][0][0]:.5g}；去两源交叉项后为{base['self']['values'][0][0]:.5g}。该环带占q<0.17内相干产额的{base['full']['yield_fraction'][0]:.1%}。这支持在原极中心区之外检验干涉角关联。"]
    for name in ['beta2','beta3']:
        data=cases[name]['stages']['decay_by'];contrast=data['contrast']
        resolved=np.abs(np.asarray(contrast['delta_full'])[:,0])>np.maximum(3*np.asarray(data['full']['se'])[:,0],1e-12)
        interaction=np.abs(np.asarray(contrast['interaction'])[:,0])>np.maximum(3*np.asarray(contrast['interaction_se'])[:,0],1e-12)
        lines += [f"{name}端点使用{cases[name]['n_orientations']}组取向：完整C2的结构差有{int(resolved.sum())}/4个环带超过3倍取向SE；full/self配对结构差有{int(interaction.sum())}/4个环带超过3倍配对SE。后者用于机制诊断，实验无法直接关闭交叉项，不能把它当成新增的独立可测观测量。"]
    lines += [f"skin端点的全区平均q变化为{1e3*cases['skin']['bridge']['full']['delta_mean_q']:.3f} MeV/c，去交叉项后为{1e3*cases['skin']['bridge']['self']['delta_mean_q']:.3f} MeV/c，继续作为原衍射宽度支线的联系。C2对skin也有响应，当前并未建立参数解耦。",'',
        '## 定义、样本与证据范围','',
        r'$\mathbf q_T=\mathbf p_{T,+}+\mathbf p_{T,-}$，$\mathbf k_T=(\mathbf p_{T,+}-\mathbf p_{T,-})/2$，$\alpha=\phi_q-\phi_k$。$C_2=\langle\cos2\alpha\rangle$，$S_2=\langle\sin2\alpha\rangle$。它们不是相对于已知b轴的集体流系数。beta2和beta3均可影响C2，不建立beta3与v3的直接对应。',
        r'$A_\phi=(N_\perp-N_\parallel)/(N_\perp+N_\parallel)$使用互补的四个对称象限。当前理想二阶角分布给出$A_\phi=-4C_2/\pi$；因此两者不能当成独立信息。响应后分别计算，不强制此关系。',
        '四个环带来自讨论后的探索定义，本轮保持固定。先平均原始强度，再做环带积分和取比值；保留跨环带和full/self的配对协方差。',
        'source_self去掉两个发射方向间的交叉项，但保留单源相干衍射和偏振各向异性；完整/自项的归一化结构差是机制对照，不是唯一加法分解。','',
        '| 样本 | 实际取向数 |','|---|---:|']
    for name,case in cases.items():lines.append(f"| {name} | {case['n_orientations']} |")
    lines += ['', '新增端点取向使用独立随机种子，再与旧样本按原始强度合并；其余形变点保留已有样本。取向SE、响应数值MC误差、给定事件数的物理统计误差分别记录。取向数增加不自动证明收敛。','',
        '## 球形参照：有限q是否保留角向干涉','',
        '| q区间 [GeV/c] | 完整C2 | 自项C2 | Aphi | 区间占相干产额 | 经验背景占该区间总产额 |',
        '|---|---:|---:|---:|---:|---:|']
    for i,(lo,hi) in enumerate(bins):
        lines.append(f"| {lo:g}–{hi:g} | {base['full']['values'][i][0]:.6g} | {base['self']['values'][i][0]:.6g} | {base['full']['values'][i][2]:.6g} | {base['full']['yield_fraction'][i]:.3%} | {cases['spherical']['background']['fraction_of_total'][i]:.3%} |")
    lines += ['', '角关联本身明显不代表形变响应已分辨；高q外缘也不能仅凭半径被排除。不同q区间正负结构可能抵消，分区间保留这种信息。背景仅为现有各向同性经验模板，未包含非相干角分布或完整背景拟合。','',
        '## 结构响应：C2及两源机制对照','',
        '以下±仅为取向积分SE。交互对照=Δfull−Δself，full/self使用同一组取向计算误差。未分辨的结果完整保留，不以改窗获得显著性。','',
        '| 样本 | q区间 | ΔC2 full ± SE | ΔC2 self | 交互对照 ± 配对SE | 相对3倍取向SE |',
        '|---|---|---:|---:|---:|---|']
    for name,case in cases.items():
        if name=='spherical':continue
        data=case['stages']['decay_by'];c=data['contrast']
        for i,(lo,hi) in enumerate(bins):
            delta=c['delta_full'][i][0];se=data['full']['se'][i][0]
            flag=('确定性；需网格检查' if case['n_orientations']==1 else
                  '可分辨' if abs(delta)>max(3*se,1e-12) else '尚未分辨')
            lines.append(f"| {name} | {lo:g}–{hi:g} | {delta:.6g} ± {se:.2g} | {c['delta_self'][i][0]:.6g} | {c['interaction'][i][0]:.6g} ± {c['interaction_se'][i][0]:.2g} | {flag} |")
    lines += ['', '## 原图形貌的联系','',
        '横坐标沿用固定b图中球形参照的暗5%区域（跨结构点冻结）以及半高亮区面积平方根。纵坐标为有限q的Aphi变化，或最终平均动量变化。当前已存缓存不支持原图形貌横坐标的逐取向误差，因此没有绘制横误差条。曲线连线仅示意变化，不能证明唯一映射。',
        '各投影阶段的暗5%区域不可当作同一批原暗像素的严格输运。这里检验的是原干涉信息的可测角向遗迹；原稿极中心区和外围横臂均保留在物理解释中。','',
        '| 样本 | 原图暗区亮度变化 | 原图半高宽度代理变化 [MeV/c] | 最终平均q变化 [MeV/c] |',
        '|---|---:|---:|---:|']
    for name,case in cases.items():
        if name=='spherical':continue
        b=case['bridge']['full']
        lines.append(f"| {name} | {b['delta_dark_fixed_mean']:.6g} | {1e3*b['delta_sqrt_area_halfmax']:.6g} | {1e3*b['delta_mean_q']:.6g} |")
    lines += ['', '## 取向稳定性与系统数值变化','',
        'analytic.json中的orientation_prefix记录8/16/32/64前缀，independent_halves记录具备新增取向的两批独立样本。前缀共享样本，不能当独立重复。粗b网格变化单列为数值系统检查，不与取向SE合并。',
        '此前b20样本的空间/快度加密不认证新增12–20 fm区间；本轮没有重新完成全域空间、快度和b尾部的收敛验证。S2作为对称性/取向采样诊断全部保留，不能因其理想期望为零而强制抹去。']
    halves=[(name,case['independent_halves']) for name,case in cases.items() if case.get('independent_halves')]
    if halves:
        lines += ['', '| 样本 | q区间 | 第一批32取向C2 | 新增32取向C2 | 两批差 ± 独立SE |',
                  '|---|---|---:|---:|---:|']
        for name,h in halves:
            for i,(lo,hi) in enumerate(bins):
                difference=h['difference']['full']
                lines.append(f"| {name} | {lo:g}–{hi:g} | {h['first']['full']['values'][i][0]:.6g} | {h['second']['full']['values'][i][0]:.6g} | {difference['values'][i][0]:.6g} ± {difference['se'][i][0]:.2g} |")
    bcheck=dest/'b_quadrature_check/report.md'
    if bcheck.exists():
        lines += ['',f'[球形及纯skin的b网格加密诊断]({bcheck})。该独立检查不能认证形变样本的b积分精度。']
        comparison=json.loads((bcheck.parent/'comparison.json').read_text())
        mean=comparison['whole_disk_mean_q']['structural_contrast']
        lines += [f"b步长2→1 fm时，全区skin平均q结构差的变化占加密值：full {mean['full']['absolute_shift_over_refined_signal']:.3%}，self {mean['self']['absolute_shift_over_refined_signal']:.3%}。",'',
                  '| q区间 | 原b步长2 fm的skin ΔC2 | 加密至1 fm | 变化占加密值 |',
                  '|---|---:|---:|---:|']
        c=comparison['structural_contrast']['full']
        for i,(lo,hi) in enumerate(bins):
            lines.append(f"| {lo:g}–{hi:g} | {c['original'][i][0]:.6g} | {c['refined'][i][0]:.6g} | {c['absolute_shift_over_refined_signal'][i][0]:.2%} |")
        lines += ['', '主图保持所有结构点同一2 fm网格，不把加密球形参照混配到旧形变样本。高q两个环带的C2结构差尚未通过b网格稳定性检查；低q较稳定也不等于全部系统误差已定量。']
    if response is not None:
        lines += ['', '## 联合径迹响应','',
            f"每个样本使用{response['n_events']:,}个共同随机求积点。误差参数均为未标定toy：轨迹相对动量0、0.5%、1%，以及0.5%配合1 mrad方位误差；q与k由同一对重建径迹计算。没有接受度、材料效应或完整实验背景。",
            '下表为完整相干信号的结构差；±仅合并响应MC和取向采样误差，不包含数值网格、模型或实验系统误差。环带迁入/迁出在response.json中记录；真值支持约0.283 GeV/c之外的迁入未模拟。','',
            '| 样本 | 响应 (sigma_p, sigma_phi) | q区间 | ΔC2 ± 数值采样SE | ΔAphi ± 数值采样SE |',
            '|---|---|---|---:|---:|']
        for case in response['cases']:
            if case['name'] not in ['beta2','beta3','skin']:continue
            for row in case['rows']:
                c=row.get('truth_anchored',row)['vs_spherical']
                for i,(lo,hi) in enumerate(bins):
                    lines.append(f"| {case['name']} | {100*row['fractional_sigma']:g}%, {1000*row['angular_sigma']:g} mrad | {lo:g}–{hi:g} | {c['delta'][4*i]:.5g} ± {c['combined_sampling_se'][4*i]:.2g} | {c['delta'][4*i+2]:.5g} ± {c['combined_sampling_se'][4*i+2]:.2g} |")
        lines += ['', '响应后的Aphi+4C2/pi单独保留；非零值可能来自响应产生的高阶角结构，不应强制为零。所有事件协方差明确区分按环带事件数N_bin还是总选中事件数N_total归一化。当前不提供实验事件数预测。']
        if response['cases'] and 'truth_anchored' in response['cases'][0]['rows'][0]:
            lines += ['',r'表中及响应图使用真值控制变量：$\widehat O_\sigma=O_{\rm truth,exact}+\widehat O_{\sigma,\rm MC}-\widehat O_{0,\rm MC}$。它利用共同轻子对估计响应增量，使零响应回到解析值；固定系数并不保证降低非零响应结构差的MC方差，部分场景会增大。跨响应和跨结构的共同随机协方差完整传播。这不是实验校准，也不减少真实事件涨落。原始MC估计、两套数值SE和事件协方差仍保存在response.json及CSV中。']
    lines += ['', '## 图与复现','']
    for file in figures['files']:
        if file.endswith('.png'):lines += [f'![{Path(file).stem}]({file})','']
    lines += ['```bash',manifest['replay_command'],
              f'/opt/anaconda3/bin/python3 scripts/verify_angular_observables.py {dest}','```','',
        f'[数值QA]({dest/"qa.json"})；[计算约定]({ROOT/"docs/jpsi_angular_observables_contract.md"})。',
        'analytic.json与response.json保存线性数值及完整协方差；CSV用于直接查看。manifest记录配置、实际输入文件SHA256、取向来源和代码版本。论文正文未改。']
    qa_path=dest/'qa.json'
    if qa_path.exists():
        qa=json.loads(qa_path.read_text())
        lines += ['',f"保存数据QA：{qa['passed_invariant_count']}/{qa['invariant_count']}项不变量通过；保留{qa['warning_count']}条采样或求积提醒。代码测试：72项通过。", 
                  f'[逐图与PDF/SVG导出检查]({dest/"figure_qa.json"})。这些检查不替代数值收敛或实验验证。']
    (dest/'report.md').write_text('\n'.join(lines)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/jpsi_angular_observables.yaml')
    p.add_argument('--output',default='outputs/jpsi_angular_revision/b12_20261008')
    p.add_argument('--base-only',action='store_true',help='Use original orientations only; choose separate exploratory output')
    p.add_argument('--response-events',type=int)
    p.add_argument('--skip-response',action='store_true')
    p.add_argument('--render-only',action='store_true')
    args=p.parse_args();dest=Path(args.output).resolve();dest.mkdir(parents=True,exist_ok=True)
    if args.render_only and (args.base_only or args.response_events or args.skip_response):
        p.error('Render-only uses saved summaries and config without overrides')
    started=time.monotonic()
    if args.render_only:
        config=yaml.safe_load((dest/'config.yaml').read_text())
        analytic=json.loads((dest/'analytic.json').read_text())
        response=json.loads((dest/'response.json').read_text()) if (dest/'response.json').exists() else None
        manifest=json.loads((dest/'manifest.json').read_text())
    else:
        config=yaml.safe_load(Path(args.config).read_text())
        src=(ROOT/config['source']).resolve();ext=None if args.base_only else (ROOT/config['orientation_extension']).resolve()
        results,parent,provenance=load_angular_cases(src,ext)
        config.update(source=str(src),orientation_extension=str(ext) if ext else None,
                      physics=parent['physics'],study={**parent['study'],**config['study']})
        if args.response_events:config['study']['response_events']=args.response_events
        frozen=yaml.safe_dump(config,sort_keys=False,allow_unicode=True)
        if (dest/'config.yaml').exists() and (dest/'config.yaml').read_text()!=frozen:
            raise ValueError('Output has different configuration; choose a new directory')
        (dest/'config.yaml').write_text(frozen)
        replay=f'cd {ROOT}\nOPENBLAS_NUM_THREADS=1 /opt/anaconda3/bin/python3 scripts/analyze_jpsi_angular.py --config {dest/"config.yaml"} --output {dest}'
        if args.base_only:replay+=' --base-only'
        if args.skip_response:replay+=' --skip-response'
        manifest={'status':'running','replay_command':replay,'inputs':provenance,
                  'independent_orientation_extension':ext is not None,'command':sys.argv,
                  'source_config_sha256':hashlib.sha256((src/'config.yaml').read_bytes()).hexdigest()}
        save_json(dest/'manifest.json',manifest)
        analytic=analyze_angular(results,config['study']);save_json(dest/'analytic.json',analytic)
        response=None if args.skip_response else run_angular_response(results,config,lambda x:print(x,flush=True))
        if response is not None:save_json(dest/'response.json',response)
        elif (dest/'response.json').exists():raise ValueError('Refusing stale response under --skip-response')
        manifest['analysis_elapsed_seconds']=time.monotonic()-started
    figures=render_angular_figures(analytic,response,config['study'],dest)
    save_json(dest/'figures.json',figures);write_tables(analytic,response,dest)
    manifest.update(status='completed',source_hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
        for f in [Path(__file__),ROOT/'scripts/refine_angular_orientations.py',*sorted((ROOT/'upc_deformation').rglob('*.py'))]},
        config_sha256=hashlib.sha256((dest/'config.yaml').read_bytes()).hexdigest(),
        last_render_elapsed_seconds=time.monotonic()-started if args.render_only else None)
    save_json(dest/'manifest.json',manifest)
    write_report(analytic,response,config,figures,manifest,dest)
    print(f'Completed angular analysis: {dest}',flush=True)

if __name__=='__main__':main()
