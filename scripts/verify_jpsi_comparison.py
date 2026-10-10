#!/usr/bin/env python3
"""Summarize saved grid/domain checks separately from physics changes."""
import argparse
import json
from pathlib import Path
import numpy as np


def distance(first, second):
    a=np.load(first);b=np.load(second)
    np.testing.assert_array_equal(a['axis'],b['axis'])
    x,y=np.meshgrid(a['axis'],a['axis'],indexing='ij')
    mask=x*x+y*y<=.17**2
    return ((abs(a['normalized_images']-b['normalized_images'])*mask).sum((1,2))
            /(a['normalized_images']*mask).sum((1,2))).tolist()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs',default='outputs')
    args=parser.parse_args();root=Path(args.outputs)
    final=root/'jpsi_decay_axis_final'
    summary=json.loads((final/'summary.json').read_text())
    baseline=root/'jpsi_decay_axis_physical/spherical.npz'
    report={
        'comparison_order':['fixed_b_known_axis','fixed_b_lepton_axis','integrated_b_lepton_axis'],
        'physical_grid_64_320_vs_96_480_at_25_b_nodes_relative_l1':distance(
            baseline,root/'jpsi_decay_axis_physical_grid_check/spherical.npz'),
        'field_extent_160_vs_192_fm_at_fixed_spacing_relative_l1':distance(
            baseline,root/'jpsi_decay_axis_physical_extent_check/spherical.npz'),
        'b_nodes_25_vs_49_at_same_grid_relative_l1':distance(
            baseline,final/'spherical.npz'),
        'upstream_vs_unit_corrected_same_grid_25_nodes_relative_l1':distance(
            root/'jpsi_decay_axis/spherical.npz',baseline),
        'tail_60_to_80_fm_yield_fraction_in_12_to_80':json.loads(
            (root/'jpsi_decay_axis_physical_tail_check/summary.json').read_text())['cases'][0]['checks']['last_20fm_yield_fraction'],
        'final_case_checks':{row['case']['name']:row['checks'] for row in summary['cases']}}
    (final/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    text=['# 最终图片与数值核验','',
          '最终结果为 physical 单位模式、49 个 b 节点；源版结果单独保留。',
          '三栏依次为：固定 b、已知 b 轴；固定 b、衰变轴；b 积分、衰变轴。',
          '主图 x 轴在后两栏沿轻子相对动量 k_T；perpendicular 版本仅转坐标 90 度。','']
    for row in summary['cases']:
        name=row['case']['name']
        text += [f'## {name}', '', f'![{name}]({name}_linear.png)', '',
                 f'[对数色标]({name}_log.png) · [参考轴转90度]({name}_perpendicular.png) · [数值数组]({name}.npz)', '']
    text += ['## 数值核验','',
             '下列 L1 距离均在 pT<0.17 GeV 的同一圆盘内、按单位产额形状比较；不是实验误差或拟合置信区间。','',
             '| 检查（球形基准） | 三种图的相对 L1 差异 |','|---|---|']
    for key,values in report.items():
        if key.endswith('relative_l1'):
            text.append('| '+key+' | '+', '.join(f'{100*v:.3f}%' for v in values)+' |')
    text += ['',f'扩大 b 积分上限的补充检查：60–80 fm 对 12–80 fm 总产额的贡献为 {report["tail_60_to_80_fm_yield_fraction_in_12_to_80"]*100:.4f}%。',
             '核取向平均采用两个核独立的 SO(3) 取向、32个配对样本；16/32差异详见 verification.json。',
             'b=12–60 fm 是明确的 hard window；不等同于实测中子标记选择函数。',
             '图中保留两个亮区及中心低谷，但固定真 b 图的长暗纹已大幅被角平均填充。',
             '没有启动全量数据生成或网络重训；这是可供决定后续数据表示的正演比较。']
    (final/'gallery.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
