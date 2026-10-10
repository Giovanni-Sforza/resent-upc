#!/usr/bin/env python3
"""Replay fixed b from saved rotation banks, then add the lepton kT panel."""
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
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/mpl_cache'))
import numpy as np
import torch
import yaml
from upc_deformation.forward import UPCForwardModel
from upc_deformation.decay import stokes_maps
from upc_deformation.kt import add_kt_arrays
from upc_deformation.kt_figures import render_four_panel


def replay_fixed_stokes(config, result):
    settings=config['comparison']; case=result['case']
    physics=copy.deepcopy(config['physics'])
    physics['nuclear_parameters'].update(beta2=case['beta2'],beta3=case['beta3'])
    physics['collision_parameters']['b']=settings['fixed_b_fm']
    device=torch.device(settings.get('device','cpu'))
    model=UPCForwardModel(physics,settings['longitudinal_units']).to(device).eval()
    r1=torch.as_tensor(result['rotation_bank_1'],device=device)
    r2=torch.as_tensor(result['rotation_bank_2'],device=device)
    axis=model._p_axis_q
    total=torch.zeros(3,len(axis),len(axis),dtype=torch.float64,device=device)
    chunk=settings['orientation_chunk']
    with torch.inference_mode():
        for start in range(0,len(r1),chunk):
            stop=min(start+chunk,len(r1));count=stop-start
            mx,my,_=model.fixed_b_complex_amplitude_multi_y(None,None,
                torch.full((count,1),settings['y'],device=device),[settings['fixed_b_fm']],
                R_1=r1[start:stop],R_2=r2[start:stop])
            mean=stokes_maps(mx[:,0,0],my[:,0,0]).mean(0)
            projected,_=model.project_full_grid_intensity_to_output(mean)
            total+=projected.double()*count
            print(f'{case["name"]}: fixed-b replay {stop}/{len(r1)}',flush=True)
        total/=len(r1)
        x,y=torch.meshgrid(axis,axis,indexing='ij')
        disk=x*x+y*y<=settings['normalization_radius_GeV']**2
        integrated=(total*disk).sum((-2,-1))*(axis[1]-axis[0])**2
    data=total.cpu().numpy()
    old=result['raw_images'][0]
    error=float(np.max(np.abs(data[0]-old))/np.max(np.abs(old)))
    if error>1e-5:
        raise AssertionError(f'Fixed-b replay changed cached production image: {error}')
    result['fixed_b_integrated_stokes']=integrated.cpu().numpy()
    return error


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default='outputs/jpsi_decay_axis_final')
    parser.add_argument('--output',default='outputs/jpsi_decay_axis_four_panel')
    parser.add_argument('--render-only',action='store_true',help='Render cached augmented npz files in --output')
    args=parser.parse_args()
    source=Path(args.input).resolve();dest=Path(args.output).resolve()
    if args.render_only:
        config=yaml.safe_load((dest/'config.yaml').read_text())
        results=[]
        for case in config['comparison']['cases']:
            with np.load(dest/f'{case["name"]}.npz') as data:result={key:data[key] for key in data.files}
            result['case']=case;results.append(result)
        render_four_panel(results,config['comparison'],dest)
        if (dest/'gallery.md').exists():
            gallery=(dest/'gallery.md').read_text().replace('_linear.png','_main.png')
            (dest/'gallery.md').write_text(gallery)
        metadata=json.loads((dest/'summary.json').read_text())
        metadata['render_source_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__),ROOT/'upc_deformation/kt_figures.py']}
        metadata.setdefault('display',{}).update(main_suffix='_main',q_panels='shared logarithmic color scale',
                                                 kt_norm=config['comparison'].get('kt_color_scale','log'))
        (dest/'summary.json').write_text(json.dumps(metadata,indent=2)+'\n')
        return
    if source==dest:raise ValueError('Use a separate output directory to retain original provenance')
    dest.mkdir(parents=True,exist_ok=True)
    config=yaml.safe_load((source/'config.yaml').read_text())
    torch.set_num_threads(config['comparison']['threads'])
    mass=config['physics']['physical_constants']['Mjpsi']
    started=time.monotonic();results=[];checks={};hashes={}
    for case in config['comparison']['cases']:
        path=source/f'{case["name"]}.npz'
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as data:result={key:data[key] for key in data.files}
        result['case']=case
        replay_error=replay_fixed_stokes(config,result)
        check=add_kt_arrays(result,mass)
        check['fixed_b_replay_relative_max_error']=replay_error
        for key in ('raw_images','normalized_images','perpendicular_images'):
            with np.load(path) as old:np.testing.assert_array_equal(result[key],old[key])
        check['original_q_images_unchanged']=True
        if abs(check['kt_total_probability']-1)>1e-5 or check['kt_quadrature_32_vs_64_l1']>1e-3:
            raise AssertionError(f'kT quadrature failed: {check}')
        checks[case['name']]=check
        np.savez_compressed(dest/path.name,**{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
        results.append(result)
    render_four_panel(results,config['comparison'],dest)
    (dest/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    summary={'status':'completed','source_directory':str(source),'source_npz_hashes':hashes,
             'source_summary_sha256':hashlib.sha256((source/'summary.json').read_bytes()).hexdigest(),
             'checks':checks,'elapsed_seconds':time.monotonic()-started,
             'definition':'kT=(pT_plus-pT_minus)/2; full polar acceptance; massless; negligible qT/M boost',
             'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [Path(__file__),ROOT/'upc_deformation/kt.py',ROOT/'upc_deformation/kt_figures.py',
                                        ROOT/'upc_deformation/forward.py',ROOT/'upc_deformation/vendor/upc_forward.py']}}
    (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# 四栏图：产生动量、轻子相对动量与衰变轴','',
           '原有 a/b/c 的三个 qT 数组原样保留，现在对应 a/c/d；新增 b 为固定 b 的 kT 概率密度。',
           'b 使用独立的动量范围和色标，包含相空间 Jacobian；边界使用像素积分。',
           'angle_origin 图左栏复现旧 image_P 的矩阵元权重；右栏给出相对 b 的参考轴角度分布。',
           '图 c/d 使用完整的 q-k 联合角分布计算，并非用边缘 P(psi) 对原图做独立随机模糊。','']
    for result in results:
        name=result['case']['name'];s=checks[name]
        lines += [f'## {result["case"]["label"]}','',
                  f'![four panels]({dest/name}_main.png)','',
                  f'![angle origin]({dest/name}_angle_origin.png)','',
                  f'偏振 S1/S0={s["linear_polarization"][0]:.6f}；垂直衰变轴的二阶方向相关 R2={s["perpendicular_axis_R2"]:.6f}；',
                  f'该轴与 b 的偏差在 ±15 度内的比例={s["perpendicular_axis_within_15deg_probability"]:.2%}。','']
    (dest/'gallery.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary['checks'],indent=2),flush=True)
    print(f'Completed in {summary["elapsed_seconds"]:.1f}s: {dest}',flush=True)


if __name__=='__main__':main()
