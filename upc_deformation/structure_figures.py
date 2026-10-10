"""Two-row structure/noise figure with honest shape ratios."""
from __future__ import annotations
import math
import numpy as np
from .structure_noise import masked_ratio


def render_structure_noise(results,config,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize,LogNorm
    from matplotlib.ticker import FixedLocator,FuncFormatter
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':10,'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':.7})
    cfg=config['comparison'];axis=results[0]['axis'];extent=[axis[0],axis[-1],axis[0],axis[-1]]
    limit=float(np.max(np.abs(axis)))
    ticks=[-limit,0,limit] if limit>=.2-1e-6 else [-.15,0,.15]
    images=np.stack([r['normalized_images'] for r in results],axis=1) # row,case,x,y
    vmax=images.max();ratios=[];masks=[];factors=[]
    for col in [1,2]:
        rr=[];mm=[]
        for row in range(2):
            ratio,keep=masked_ratio(images[row,col],images[row,0],axis,cfg['normalization_radius_GeV'],cfg['ratio_floor_fraction'])
            rr.append(ratio);mm.append(keep)
        rr=np.stack(rr);mm=np.stack(mm);ratios.append(rr);masks.append(mm)
        factors.append(max(1.05,float(np.nanmax(rr)),1/float(np.nanmin(rr))))
    data={'shape_ratios':np.stack(ratios,axis=1),'ratio_valid_mask':np.stack(masks,axis=1),
          'ratio_color_factors':np.array(factors)}

    def save(fig,stem):
        for ext in ['png','pdf','svg']:fig.savefig(destination/f'{stem}.{ext}',dpi=200,bbox_inches='tight')
        plt.close(fig)

    names=['Spherical','Large deformation','Neutron skin (skin type)','Deformed / spherical','Skin / spherical']
    beta=results[1]['case']['nuclear_overrides']
    labels=[r'$\beta_2=\beta_3=0$',rf'$\beta_2={beta["beta2"]:g},\ \beta_3={beta["beta3"]:g}$; {results[1]["orientations"]} orientations',
            rf'$\Delta r_{{np}}={config["structure_study"]["skin_rms_fm"]:.2f}$ fm; $a_n=a_p$','Shape ratio','Shape ratio']
    for mode in ['log']:
        fig=plt.figure(figsize=(18,9.5));artists={}
        for row,ypos in enumerate([.565,.165]):
            for col,xpos in enumerate([.055,.237,.419,.631,.813]):
                ax=fig.add_axes([xpos,ypos,.153,.29])
                if col<3:
                    norm=LogNorm(vmax*1e-4,vmax)
                    im=ax.imshow(images[row,col].T,origin='lower',extent=extent,cmap='magma',norm=norm,interpolation='nearest')
                    if row==1:
                        ax.text(.97,.95,rf'$f_{{inc}}={100*results[col]["incoherent_fraction_in_disk"]:.1f}\%$',
                                transform=ax.transAxes,ha='right',va='top',color='white',fontsize=9)
                else:
                    i=col-3
                    im=ax.imshow(np.ma.masked_invalid(ratios[i][row].T),origin='lower',extent=extent,
                        cmap='RdBu_r',norm=LogNorm(1/factors[i],factors[i]),interpolation='nearest')
                    ax.set_facecolor('#e8e8e8')
                artists[col]=im
                ax.set_xticks(ticks);ax.set_yticks(ticks)
                if col!=0:ax.set_yticklabels([])
                else:ax.set_ylabel(r'$q_y$ (GeV/$c$)')
                if row==1:ax.set_xlabel(r'$q_x$ (GeV/$c$)')
                letter=chr(97+row*5+col)
                if row==0:ax.set_title(f'({letter})  {names[col]}\n{labels[col]}',fontsize=10,pad=9)
                else:ax.set_title(f'({letter})  {names[col]}',fontsize=10,pad=7)
        cax=fig.add_axes([.09,.072,.45,.018])
        fig.colorbar(artists[0],cax=cax,orientation='horizontal',label=r'Unit-yield density [(GeV/$c$)$^{-2}$]; shared logarithmic color scale')
        for i,x in enumerate([.631,.813]):
            cax=fig.add_axes([x,.072,.153,.018]);f=factors[i]
            cbar=fig.colorbar(artists[i+3],cax=cax,orientation='horizontal',label=names[i+3])
            cbar.locator=FixedLocator([1/f,1/math.sqrt(f),1,math.sqrt(f),f])
            cbar.formatter=FuncFormatter(lambda v,pos:f'{v:.2f}')
            cbar.update_ticks();cbar.minorticks_off();cbar.ax.tick_params(labelsize=8)
        fig.text(.055,.918,'Coherent only',fontsize=13,weight='bold')
        fig.text(.055,.510,'Coherent + isotropic incoherent',fontsize=13,weight='bold')
        fig.suptitle(r'$^{96}$Zr+$^{96}$Zr, 200 GeV, $y=0$ | '+rf'$b\in[{cfg["b_min_fm"]:g},{cfg["b_max_fm"]:g}]$ fm | $x\parallel k_T$',y=.982,fontsize=14)
        fig.text(.5,-.022,'Raw intensities are averaged before unit normalization in $q_T<0.17$ GeV/$c$. '
                 'Each ratio column has its own scale, shared between rows; white = 1.\n'
                 f'Gray ratio pixels: outside the normalization disk or spherical density < {100*cfg["ratio_floor_fraction"]:g}% of its peak. '
                 'Incoherent: source empirical yield, isotropic Gaussian; Born level, full polar acceptance.',ha='center',fontsize=8)
        # Canonical main output always uses logarithmic intensity coloring.
        for ext in ['png','pdf','svg']:
            fig.savefig(destination/f'structure_noise_main.{ext}',dpi=200,bbox_inches='tight')
        save(fig,f'structure_noise_{mode}')

    # All independently oriented raw images, displayed with one common yield
    # denominator so their arithmetic mean remains the main deformed image.
    result=results[1];ori=result['coherent_images_per_orientation']/result['yields'][0]
    n=len(ori);columns=8;rows=math.ceil((n+1)/columns)
    fig,axs=plt.subplots(rows,columns,figsize=(16,2.15*rows),squeeze=False)
    fig.subplots_adjust(left=.025,right=.94,bottom=.05,top=.945,wspace=.14,hspace=.29)
    norm=Normalize(0,max(float(ori.max()),float(images[0,1].max())))
    for i,ax in enumerate(axs.flat):
        if i>n:ax.axis('off');continue
        im=ori[i] if i<n else images[0,1]
        artist=ax.imshow(im.T,origin='lower',extent=extent,cmap='magma',norm=norm)
        ax.set_title(f'Orientation pair {i+1}' if i<n else f'Mean of {n}',fontsize=8)
        ax.set_xticks([]);ax.set_yticks([])
    cax=fig.add_axes([.955,.15,.012,.63]);fig.colorbar(artist,cax=cax,label='Density / common ensemble yield')
    fig.suptitle('Independent orientations of both deformed nuclei: intensities first, ensemble average last\n'
                 'Every panel uses the same b integration and lepton reference axis as the main intensity panels',fontsize=12)
    save(fig,'deformed_orientation_gallery')

    # Radial and angular summaries are obtained from the saved coefficients,
    # without fitting an image or using the logarithmic display values.
    fig,axs=plt.subplots(2,2,figsize=(10,7.4),layout='constrained')
    colors=['#444444','#2878a1','#c3662e']
    for result,color in zip(results,colors):
        r=result['radii'];a0,c,d=result['coefficients_per_orientation'].mean(0)
        inc_total=result['incoherent_yield_per_orientation'].mean()
        slope=result['physics']['output_settings']['b_slope_incoh']
        inc=inc_total*slope/np.pi*np.exp(-slope*r*r)
        keep=(r>0)&(r<=cfg['normalization_radius_GeV'])
        for row,base in enumerate([a0,a0+inc]):
            axs[row,0].plot(r[keep],(base/result['yields'][row])[keep],color=color,label=result['case']['label'])
            valid=keep&(base>1e-8*base.max())
            axs[row,1].plot(r[valid],(-c/(4*np.maximum(base,1e-30)))[valid],color=color)
        for row in [0,1]:
            axs[row,0].set_ylabel('Azimuth-averaged density')
            axs[row,1].set_ylabel(r'$v_2(q_T)=\langle\cos2\alpha\rangle$')
            for ax in axs[row]:ax.set_xlabel(r'$q_T$ (GeV/$c$)');ax.spines[['top','right']].set_visible(False)
    axs[0,0].set_title('Coherent: radial shape');axs[0,1].set_title('Coherent: angular modulation')
    axs[1,0].set_title('With isotropic incoherent');axs[1,1].set_title('With isotropic incoherent')
    axs[0,0].legend(fontsize=9);save(fig,'radial_and_v2')
    return data
