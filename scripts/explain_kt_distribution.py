#!/usr/bin/env python3
"""Separate decay weight, phase-space projection, coloring and angular bias.

Use the saved physical Stokes vector; no new forward run or event sampling.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/mpl_cache'))
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,Normalize
from matplotlib.patches import Circle
from upc_deformation.kt import polarization


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default='outputs/jpsi_decay_axis_b20/deformed_average.npz')
    parser.add_argument('--output',default='outputs/jpsi_kt_explanation_b20')
    args=parser.parse_args()
    source=Path(args.input).resolve()
    dest=Path(args.output).resolve();dest.mkdir(parents=True,exist_ok=True)
    config=yaml.safe_load((source.parent/'config.yaml').read_text())
    fixed_b=config['comparison']['fixed_b_fm']
    with np.load(source) as z:r={key:z[key] for key in z.files}
    p1,p2=polarization(r['fixed_b_integrated_stokes'])
    density=r['kt_density'];edges=r['kt_edges'];k0=float(edges[-1])
    probability=float(density.sum()*(edges[1]-edges[0])**2)
    assert abs(probability-1)<1e-8
    masked=np.ma.masked_where(density<=0,density)
    extent=[-k0,k0,-k0,k0]
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':10,'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':.7})
    fig=plt.figure(figsize=(15.6,5.3))
    axes=[fig.add_axes([left,.285,.19,.54]) for left in [.05,.285,.52]]
    curve=fig.add_axes([.775,.285,.21,.54])
    panels=[(r['kt_matrix_weight'],Normalize(0,1),'(a)  Matrix-element weight\nBefore phase-space projection',r'$W/(M^2\Sigma)$'),
            (masked,Normalize(0,float(density.max())),'(b)  Physical density\nLinear color scale',r'$dP/d^2k_T$ [(GeV/$c$)$^{-2}$]'),
            (masked,LogNorm(float(density[density>0].min()),float(density.max())),'(c)  Same physical density\nLogarithmic color scale',r'$dP/d^2k_T$ [(GeV/$c$)$^{-2}$]')]
    for i,(ax,(values,norm,title,label)) in enumerate(zip(axes,panels)):
        im=ax.imshow(values.T,origin='lower',extent=extent,cmap='magma',norm=norm,interpolation='nearest')
        ax.set_facecolor('#eeeeee');ax.set_title(title,fontsize=11,pad=10)
        ax.set_xlabel(r'$k_x$ (GeV/$c$)');ax.set_ylabel(r'$k_y$ (GeV/$c$)')
        ax.set_xticks([-1.5,0,1.5]);ax.set_yticks([-1.5,0,1.5])
        if i==0:ax.add_patch(Circle((0,0),k0,fill=False,ls='--',lw=1.1,color='#20bec7'))
        box=ax.get_position();cax=fig.add_axes([box.x0,.155,box.width,.024])
        fig.colorbar(im,cax=cax,orientation='horizontal',label=label).ax.tick_params(labelsize=9)
    psi=r['kt_psi'];keep=psi<=np.pi
    curve.plot(np.degrees(psi[keep]),2*np.pi*r['kt_azimuth_density'][keep],color='#2878a1',lw=2)
    curve.axhline(1,color='#999999',ls='--',lw=1)
    curve.axvspan(75,105,color='#2878a1',alpha=.1)
    curve.set(xlim=(0,180),ylim=(0,1.65),xticks=[0,45,90,135,180],
              xlabel=r'$\psi=\angle(k_T,b)$ (degrees)',ylabel='Angular density / uniform')
    curve.set_title('(d)  Directional preference\nFull polar-angle acceptance',fontsize=11,pad=10)
    curve.spines[['top','right']].set_visible(False)
    fraction15=1/6+p1/(4*np.pi)
    curve.text(.5,.05,rf'$P(75^\circ<\psi\ {{\rm mod}}\ 180^\circ<105^\circ)={100*fraction15:.1f}\%$',
               transform=curve.transAxes,ha='center',fontsize=9)
    fig.suptitle(rf'Fixed $b={fixed_b:g}$ fm, $x\parallel b$ | Same polarization input in all four panels',fontsize=13,y=.98)
    fig.text(.5,.025,r'(a): cyan circle marks $k_T=M/2$; square corners are outside the on-shell decay region.  '
             '(b,c): identical finite-bin probabilities, different color normalization.\n'
             r'Born level; massless leptons; negligible transverse boost; the same $q_T<0.17$ GeV/$c$ sample.',
             ha='center',fontsize=8)
    for ext in ['png','pdf','svg']:fig.savefig(dest/f'kt_weight_vs_density.{ext}',dpi=210,bbox_inches='tight')
    plt.close(fig)
    stats={'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'fixed_b_fm':fixed_b,
           'p1':float(p1),'p2':float(p2),'kt_max_GeV':k0,'total_probability':probability,
           'angular_density_y_over_x':float((1+.5*p1)/(1-.5*p1)),
           'axis_within_15deg_of_perpendicular_probability':float(fraction15),
           'axis_within_45deg_of_perpendicular_probability':float(.5+p1/(2*np.pi)),
           'scope':'current saved sample, not a reproduction of the historical screenshot'}
    (dest/'checks.json').write_text(json.dumps(stats,indent=2)+'\n')
    np.savez_compressed(dest/'source_data.npz',kt_axis=r['kt_axis'],kt_edges=edges,matrix_element_weight=r['kt_matrix_weight'],
                        physical_density=density,psi=psi,azimuth_density=r['kt_azimuth_density'],stokes=r['fixed_b_integrated_stokes'])
    print(json.dumps(stats,indent=2))


if __name__=='__main__':main()
