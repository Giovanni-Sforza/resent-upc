#!/usr/bin/env python3
"""Two-row, five-column nuclear-structure comparison in the measured axis."""
from __future__ import annotations
import argparse
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
from upc_deformation.structure_noise import calculate_structure_case,neutron_radius_for_skin,ws_rms
from upc_deformation.structure_figures import render_structure_noise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/jpsi_structure_noise.yaml')
    parser.add_argument('--output',default='outputs/jpsi_structure_noise')
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--render-only',action='store_true')
    args=parser.parse_args();dest=Path(args.output).resolve();dest.mkdir(parents=True,exist_ok=True)
    if args.render_only:
        config=yaml.safe_load((dest/'config.yaml').read_text());results=[]
        for name in ['spherical','deformed','skin']:
            meta=json.loads((dest/f'{name}.json').read_text())
            with np.load(dest/f'{name}.npz') as data:meta.update({key:data[key] for key in data.files})
            results.append(meta)
        extra=render_structure_noise(results,config,dest)
        np.savez_compressed(dest/'ratios.npz',**extra)
        if (dest/'report.md').exists():
            report=(dest/'report.md').read_text().replace('structure_noise_linear.png','structure_noise_main.png')
            (dest/'report.md').write_text(report)
        if (dest/'summary.json').exists():
            summary=json.loads((dest/'summary.json').read_text())
            summary['render_source_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [Path(__file__),ROOT/'upc_deformation/structure_figures.py']}
            summary['array_hashes']['ratios.npz']=hashlib.sha256((dest/'ratios.npz').read_bytes()).hexdigest()
            summary['display']={'main_figure':'structure_noise_main','intensity_scale':'logarithmic','ratio_scale':'unchanged'}
            (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        return
    config=yaml.safe_load(Path(args.config).read_text());cfg=config['comparison'];nuc=config['physics']['nuclear_parameters']
    study=config['structure_study']
    if study['skin_type']!='skin':raise ValueError('This requested comparison uses skin-type, not diffuseness variation')
    rn=neutron_radius_for_skin(nuc['R_ws_p'],nuc['d_ws_p'],nuc['d_ws_p'],study['skin_rms_fm'])
    nuc['d_ws_n']=nuc['d_ws_p'];nuc['R_ws_n']=nuc['R_ws_p']
    config['resolved_cases']=[
        {'name':'spherical','label':'Spherical','orientation':'fixed','nuclear_overrides':{'beta2':0.,'beta3':0.}},
        {'name':'deformed','label':'Large deformation','orientation':'independent_average',
         'nuclear_overrides':{'beta2':study['deformed_beta2'],'beta3':study['deformed_beta3']}},
        {'name':'skin','label':'Neutron skin','orientation':'fixed',
         'nuclear_overrides':{'beta2':0.,'beta3':0.,'R_ws_n':rn,'d_ws_n':nuc['d_ws_p']}}]
    if args.quick:
        grid=config['physics']['grid_parameters'];grid['WS']['N']=grid['T_prim']['N']=32;grid['output']['N']=160
        config['physics']['output_settings']['target_size_q']=181
        config['physics']['incoherent_quadrature'].update(n_r=96,n_phi=128)
        cfg.update(n_orientations=4,n_radial=321,n_phi=256,b_nodes=13)
    (dest/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    torch.set_num_threads(cfg['threads']);started=time.monotonic();results=[]
    def log(message,**kwargs):
        print(message,flush=True)
        with (dest/'run.log').open('a') as f:f.write(message+'\n')
    log(f'Starting structure/noise comparison; skin-type Rn={rn:.8f} fm, rn-rp={study["skin_rms_fm"]:.3f} fm; quick={args.quick}')
    for case in config['resolved_cases']:
        result=calculate_structure_case(config,case,log);results.append(result)
        np.savez_compressed(dest/f'{case["name"]}.npz',**{key:val for key,val in result.items() if isinstance(val,np.ndarray)})
        (dest/f'{case["name"]}.json').write_text(json.dumps({key:val for key,val in result.items() if not isinstance(val,np.ndarray)},indent=2)+'\n')
        log(f'{case["name"]}: incoherent fraction={result["incoherent_fraction_in_disk"]:.5f}, checks={result["checks"]}')
    extra=render_structure_noise(results,config,dest);np.savez_compressed(dest/'ratios.npz',**extra)
    summary={'status':'completed','quick':args.quick,'scope':'ideal_born_level_shape_comparison_with_empirical_isotropic_noise',
             'elapsed_seconds':time.monotonic()-started,'command':sys.argv,'config':config,
             'skin':{'type':'skin','R_ws_n_fm':rn,'R_ws_p_fm':nuc['R_ws_p'],'a_p_equals_a_n_fm':nuc['d_ws_p'],
                     'proton_rms_fm':ws_rms(nuc['R_ws_p'],nuc['d_ws_p']),'neutron_rms_fm':ws_rms(rn,nuc['d_ws_p'])},
             'cases':[{key:(val.tolist() if isinstance(val,np.ndarray) else val) for key,val in r.items()
                       if key in ['case','checks','orientations','density_rms_first_orientation','elapsed_seconds','incoherent_fraction_in_disk','v2','yields','center_to_peak']} for r in results],
             'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*sorted((ROOT/'upc_deformation').rglob('*.py'))]},
             'array_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.glob('*.npz')}}
    (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    report=['# d 图模板：形变、中子皮及各向同性非相干背景','',
            '主图为两行五列：球形/大形变/skin-type中子皮/形变除球形/中子皮除球形。第二行加入现有模型计算出的各向同性经验非相干背景。',
            '两行均先平均与相加原始强度，最后在同一 qT<0.17 GeV 圆盘归一化；比值是形状比，不是绝对截面比。',
            f'两个动量分量均计算并显示至正负 {config["physics"]["output_settings"]["p_limit_q"]:g} GeV/c，左侧六张强度图使用统一对数色标。',
            f'中子皮只改变中子 WS 半密度半径：Rp={nuc["R_ws_p"]:.4f} fm，Rn={rn:.6f} fm，an=ap={nuc["d_ws_p"]:.4f} fm，目标 RMS 皮厚 {study["skin_rms_fm"]:.3f} fm。',
            f'形变 beta2={study["deformed_beta2"]}、beta3={study["deformed_beta3"]}，两核独立取向 {cfg["n_orientations"]} 对，逐取向强度平均；没有先平均密度或复振幅。','',
            f'![主图]({dest}/structure_noise_main.png)','',
            '| 样本 | 窗口内非相干占比 | coherent v2 | total v2 | 中心/峰值 coherent | 中心/峰值 total |',
            '|---|---:|---:|---:|---:|---:|']
    for r in results:
        report.append(f'| {r["case"]["label"]} | {r["incoherent_fraction_in_disk"]:.3%} | {r["v2"][0]:.6f} | {r["v2"][1]:.6f} | {r["center_to_peak"][0]:.6g} | {r["center_to_peak"][1]:.6g} |')
    report+=['',f'![径向与v2]({dest}/radial_and_v2.png)','',
             f'[逐取向图册]({dest}/deformed_orientation_gallery.png)','',
             '比值中基准小于该行峰值 0.1% 及圆盘以外的像素标灰，不使用 epsilon 掩饰除零。',
             '非相干在此作为各向同性经验背景；未引入探测器噪声随机抽样，不是 Good-Walker 方差计算。',
             '取向平均是沿用项目的经典强度平均；形变样本保持 R、a 参数，未额外匹配球形的 RMS 半径。',
             '完整数值诊断、原始数据及配置见 summary.json、各样本 npz/json 和 config.yaml。']
    (dest/'report.md').write_text('\n'.join(report)+'\n')
    log(f'Completed in {summary["elapsed_seconds"]:.1f}s: {dest}')


if __name__=='__main__':main()
