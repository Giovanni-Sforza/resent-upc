#!/usr/bin/env python3
"""Plot three saved panel-d C2 curves for manual momentum-window selection."""
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
from scipy.interpolate import RegularGridInterpolator
from upc_deformation.angular_cache import load_case
from upc_deformation.panel_d_three_bins import angular_curve
from upc_deformation.panel_d_ratio_figures import _style

NAMES=('beta2','beta3','skin')
COLORS={'beta2':'#26749d','beta3':'#8670a4','skin':'#bc6937'}
LABELS={'beta2':r'$\beta_2=0.3$', 'beta3':r'$\beta_3=0.2$', 'skin':r'$\Delta r_{np}=0.2$ fm'}
STYLES={'beta2':'-', 'beta3':'--', 'skin':'-.'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def orientation_curve_se(result,q,alpha,expected):
    """Assess the existing orientation integral with the same pixel sampler."""
    import torch
    from upc_deformation.decay import decay_axis_image
    n=int(result['n_orientations'])
    if n==1:
        return np.zeros_like(q),{'n_orientations':1,'deterministic':True}
    if n!=32:raise ValueError('Use the original 32-orientation cache')
    axis=np.asarray(result['axis'],dtype=float)
    coordinates=np.stack((q[:,None]*np.cos(alpha),q[:,None]*np.sin(alpha)),axis=-1)
    coefficients=np.asarray(result['orientation_coefficients'][:,-1,0],dtype=float)
    den=[];num=[];image_sum=np.zeros((len(axis),len(axis)))
    with torch.no_grad():
        for c in coefficients:
            image=decay_axis_image(torch.from_numpy(c),torch.from_numpy(np.asarray(result['radii'],dtype=float)),torch.from_numpy(axis)).numpy()
            image_sum+=image
            polar=RegularGridInterpolator((axis,axis),image,method='linear',bounds_error=True)(coordinates)
            den.append(polar.mean(1));num.append((polar*np.cos(2*alpha)).mean(1))
    den=np.asarray(den);num=np.asarray(num);value=num.mean(0)/den.mean(0)
    curve_error=float(np.max(np.abs(value-expected)))
    archived=np.asarray(result['images'][3,0])
    image_error=float(np.max(np.abs(image_sum/n-archived))/np.max(np.abs(archived)))
    if curve_error>1e-12 or image_error>1e-12:raise ValueError('Orientation mean does not reproduce original panel d')
    influence=(num-den*value)/den.mean(0)
    return influence.std(0,ddof=1)/np.sqrt(n),{'n_orientations':n,'max_C2_curve_closure':curve_error,'max_image_relative_closure':image_error}


def draw(q,curves,se,destination):
    plt=_style();valid=q>0;files=[]
    for difference in (False,True):
        fig,ax=plt.subplots(figsize=(8.4,4.9))
        fig.subplots_adjust(left=.105,right=.975,bottom=.23,top=.865)
        for name in NAMES:
            values=curves[name]-curves['spherical'] if difference else curves[name]
            scale=1e3 if difference else 1.
            if difference and name!='skin':
                ax.fill_between(q[valid],scale*(values-se[name])[valid],scale*(values+se[name])[valid],color=COLORS[name],alpha=.10,linewidth=0)
            ax.plot(q[valid],scale*values[valid],color=COLORS[name],linestyle=STYLES[name],linewidth=1.8,label=LABELS[name],zorder=3)
        ax.axhline(0,color='#aaaaaa',linewidth=.7,zorder=1)
        ax.set(xlim=(0,.17),xticks=np.r_[np.arange(0,.161,.02),.17],xlabel=r'$p_T$ (GeV/$c$)',
               ylabel=r'$10^3\,[C_2(p_T)-C_{2,\mathrm{sph}}(p_T)]$' if difference else r'$C_2(p_T)=\langle\cos 2\alpha\rangle_{p_T}$')
        ax.grid(color='#e5e5e5',linewidth=.5)
        ax.legend(loc='lower left',ncol=3,fontsize=8.5)
        ax.margins(y=.12)
        fig.suptitle('Change of the angular moment from the spherical reference' if difference else 'Angular moment versus transverse momentum',fontsize=11,y=.965)
        note=('Shading: pointwise 1 SE of the 32-pair orientation integral; not experimental uncertainty.' if difference else
              'Central estimates from the same panel-d densities; deformation orientation errors are shown in the companion figure.')
        fig.text(.105,.053,r'$b=12$–$60$ fm, $\Delta b=2$ fm, $|y|<1$; $x\parallel k_T$. No momentum windows selected.'+'\n'+note,fontsize=7,linespacing=1.5)
        stem='delta_C2_vs_pT' if difference else 'C2_vs_pT'
        for ext in ('png','pdf','svg'):
            path=destination/f'{stem}.{ext}';fig.savefig(path,dpi=300,bbox_inches='tight',pad_inches=.08);files.append(str(path))
        plt.close(fig)
    return files


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='outputs/jpsi_observables_revision/b12_20261007')
    parser.add_argument('--curves',default='outputs/panel_d_three_bins/b12_original32_skin_boundaries_20261009/arrays.npz')
    parser.add_argument('--decomposition',default='outputs/panel_d_ratio_decomposition/b12_original32_20261009/decomposition.npz')
    parser.add_argument('--output',default='outputs/panel_d_c2_curves/b12_original32_20261009')
    args=parser.parse_args();paths={k:(ROOT/v).resolve() for k,v in vars(args).items()};out=paths['output']
    if out==paths['source'] or paths['source'] in out.parents:raise ValueError('Separate output directory required')
    inputs=[paths['curves'],paths['decomposition']]+[paths['source']/f'{n}.{ext}' for n in ('spherical',*NAMES) for ext in ('npz','json')]
    hashes={str(p):sha(p) for p in inputs}
    with np.load(paths['curves']) as z:saved={k:z[k] for k in z.files}
    with np.load(paths['decomposition']) as z:decomp={k:z[k] for k in z.files}
    q=saved['radii'];alpha=saved['angles'];curves={};errors={};qa={}
    if not np.array_equal(q,decomp['radii']) or not np.array_equal(alpha,decomp['angles']):raise ValueError('Different curve coordinates')
    for name in ('spherical',*NAMES):
        polar=decomp['base_polar'] if name=='spherical' else decomp[name+'__polar_image']
        calculated=angular_curve(polar,alpha)['C2'];curve=saved[f'curves__{name}__C2']
        if not np.array_equal(calculated,curve):raise ValueError('Saved curve differs from its panel-d density')
        curves[name]=curve
        errors[name],qa[name]=orientation_curve_se(load_case(paths['source'],name),q,alpha,curve)
    out.mkdir(parents=True,exist_ok=True)
    save_json(out/'config.json',{k:str(p) for k,p in paths.items()})
    files=draw(q,curves,errors,out)
    arrays={'pT_GeV':q}
    for name in curves:arrays.update({f'{name}_C2':curves[name],f'{name}_delta_C2':curves[name]-curves['spherical'],f'{name}_orientation_se':errors[name]})
    np.savez_compressed(out/'curves.npz',**arrays)
    with (out/'curves.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(arrays)
        writer.writerows(zip(*arrays.values()))
    if hashes!={str(p):sha(p) for p in inputs}:raise ValueError('Input changed')
    save_json(out/'qa.json',{'status':'pass','all_curves_bitwise_equal_to_saved_panel_d_projection':True,'inputs_unchanged':True,'origin_omitted_from_plot':True,'preset_windows':False,'orientation_checks':qa})
    code=[Path(__file__),ROOT/'upc_deformation/panel_d_three_bins.py',ROOT/'upc_deformation/panel_d_ratio_figures.py',ROOT/'upc_deformation/decay.py']
    save_json(out/'manifest.json',{'status':'completed','inputs_sha256':hashes,'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in code},'figure_files':files,'outputs_sha256':{p.name:sha(p) for p in out.iterdir() if p.suffix in ('.png','.pdf','.svg','.csv','.npz')}})
    lines=['# C2随动量变化的三条曲线','',
           '主图为四极形变β2=0.3、八极形变β3=0.2和中子皮Δr_np=0.2 fm的C2(pT)。曲线直接复用原panel d密度在固定半径的强度加权角平均，不进行拟合或平滑。辅助图显示相对球形参照的Delta C2及形变的点态取向积分1SE。',
           '保持原b=12–60 fm、步长2 fm、|y|<1、各纯形变32对取向。本轮不预设区间。图中不画方向未定义的pT=0点；CSV/NPZ保留完整网格，0点只作数值占位。',
           '误差阴影只反映有限取向积分，不是实验误差，也不是同时置信带；球形/skin的零取向误差不表示无系统误差。高pT仍需结合原b求积敏感性。',
           '后续选定有限区间后，需要按该区间的原始强度加权统计C2，不能对曲线高度做等权平均。','']
    for p in files:
        if p.endswith('.png'):lines.extend([f'![{Path(p).stem}]({p})',''])
    lines.extend(['复现：','```bash',f'cd {ROOT}','OPENBLAS_NUM_THREADS=1 /opt/anaconda3/bin/python3 scripts/plot_panel_d_c2_curves.py','```'])
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(f'Completed: {out}',flush=True)


if __name__=='__main__':main()
