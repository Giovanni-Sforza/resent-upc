"""Four-panel UPC/decay comparison and angle-origin diagnostic."""
from __future__ import annotations
import numpy as np


def render_four_panel(results, settings, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.patches import Arc, Circle
    from .kt import polarization
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':10,'svg.fonttype':'none','pdf.fonttype':42,
                         'axes.linewidth':.7,'savefig.facecolor':'white'})
    maximum = max(float(r['normalized_images'].max()) for r in results)
    kmaximum = max(float(r['kt_density'].max()) for r in results)
    kminimum = min(float(r['kt_density'][r['kt_density']>0].min()) for r in results)
    # Keep a separate kT color scale, including the finite-bin edge maximum.
    kscale = settings.get('kt_color_scale', 'log')
    if kscale not in ('linear', 'log'):
        raise ValueError('kt_color_scale must be linear or log')
    knorm = Normalize(0, kmaximum) if kscale=='linear' else LogNorm(kminimum, kmaximum)

    def save(fig, stem):
        for ext in ('png','pdf','svg'):
            fig.savefig(destination/f'{stem}.{ext}',dpi=190,bbox_inches='tight')
        plt.close(fig)

    for result in results:
        a=result['axis']; extent=[a[0],a[-1],a[0],a[-1]]
        qlimit=float(np.max(np.abs(a)))
        qticks=[-qlimit,0,qlimit] if qlimit>=.2-1e-6 else [-.15,0,.15]
        edges=result['kt_edges']; kextent=[edges[0],edges[-1],edges[0],edges[-1]]
        name=result['case']['name']
        for mode in ('log','perpendicular'):
            images=result['perpendicular_images'] if mode=='perpendicular' else result['normalized_images']
            fig=plt.figure(figsize=(15.8,5.5))
            axs=[fig.add_axes([x,.31,.185,.532]) for x in [.055,.292,.530,.767]]
            norm=LogNorm(maximum*1e-4,maximum)
            for panel, imindex in [(0,0),(2,1),(3,2)]:
                ax=axs[panel]
                artist=ax.imshow(images[imindex].T,origin='lower',extent=extent,cmap='magma',norm=norm,interpolation='nearest')
                ax.set_xticks(qticks);ax.set_yticks(qticks)
                ax.set_xlabel(r'$q_x$ (GeV/$c$)')
                ax.set_ylabel(r'$q_y$ (GeV/$c$)')
            axs[0].set_title(f"(a)  J/$\\psi$ | known $b$ axis\n$b={settings['fixed_b_fm']:g}$ fm, $x\\parallel b$",fontsize=11)
            reference=r'$x\perp k_T$' if mode=='perpendicular' else r'$x\parallel k_T$'
            axs[2].set_title(f"(c)  J/$\\psi$ | decay axis\n$b={settings['fixed_b_fm']:g}$ fm, "+reference,fontsize=11)
            axs[3].set_title(f"(d)  J/$\\psi$ | decay axis\n${settings['b_min_fm']:g}\\leq b\\leq {settings['b_max_fm']:g}$ fm, "+reference,fontsize=11)
            density=np.ma.masked_where(result['kt_density']<=0,result['kt_density'])
            ka=axs[1].imshow(density.T,origin='lower',extent=kextent,cmap='magma',
                             norm=knorm,interpolation='nearest')
            axs[1].set_facecolor('#f1f1f1')
            axs[1].set_title(f"(b)  Lepton relative $k_T$\n$b={settings['fixed_b_fm']:g}$ fm, $x\\parallel b$",fontsize=11)
            axs[1].set_xlabel(r'$k_x$ (GeV/$c$)');axs[1].set_ylabel(r'$k_y$ (GeV/$c$)')
            axs[1].set_xticks([-1.5,0,1.5]);axs[1].set_yticks([-1.5,0,1.5])
            cbk=fig.add_axes([.292,.155,.185,.026])
            fig.colorbar(ka,cax=cbk,orientation='horizontal',label=r'(b): $dP/d^2k_T$ [(GeV/$c$)$^{-2}$], '+kscale+' scale')
            cbq=fig.add_axes([.550,.155,.385,.026])
            fig.colorbar(artist,cax=cbq,orientation='horizontal',label=r'(a,c,d): normalized $J/\psi$ density [(GeV/$c$)$^{-2}$], log scale')
            fig.text(.055,.12,r'$q_T=p_{T,+}+p_{T,-}$'+'\n'+r'$k_T=(p_{T,+}-p_{T,-})/2$',fontsize=11)
            case=result['case']
            fig.suptitle(r'$^{96}$Zr+$^{96}$Zr, 200 GeV, $y=0$ | '+case['label']+
                         rf" | $\beta_2={case['beta2']:g}$, $\beta_3={case['beta3']:g}$",y=.99,fontsize=13)
            fig.text(.5,.012,'Born level; massless leptons; full polar acceptance; negligible transverse boost. '
                     '(b) uses the same $q_T<0.17$ GeV/$c$ sample.\n'
                     '(a,c,d): common logarithmic color scale and unit yield in the same $q_T$ disk. '
                     f'(b): independent {kscale} color scale, unit total probability, finite-bin boundary density.',
                     ha='center',va='bottom',fontsize=8)
            if mode=='log':
                for ext in ('png','pdf','svg'):
                    fig.savefig(destination/f'{name}_main.{ext}',dpi=190,bbox_inches='tight')
            save(fig,f'{name}_{mode}')

        # Separately show the old square weight, geometric definitions and the
        # actual angular marginal. The middle panel is explicitly illustrative.
        fig,axs=plt.subplots(1,3,figsize=(12.6,4.9),gridspec_kw={'width_ratios':[1,1,1.15]})
        fig.subplots_adjust(left=.07,right=.98,bottom=.25,top=.77,wspace=.38)
        ax=axs[0]
        wi=ax.imshow(result['kt_matrix_weight'].T,origin='lower',extent=kextent,cmap='magma',vmin=0,vmax=1)
        ax.contour(result['kt_axis'],result['kt_axis'],result['kt_matrix_weight'].T,
                   levels=[.2,.4,.6,.8],colors='white',linewidths=.6,alpha=.7)
        ax.add_patch(Circle((0,0),edges[-1],fill=False,ls='--',lw=1,color='#24c3d6'))
        ax.set_xlabel(r'$k_x$ (GeV/$c$)');ax.set_ylabel(r'$k_y$ (GeV/$c$)')
        ax.set_xticks([-1.5,0,1.5]);ax.set_yticks([-1.5,0,1.5])
        ax.set_title('(a)  Old matrix-element weight\nNo phase-space Jacobian',fontsize=11)
        fig.colorbar(wi,ax=ax,location='bottom',fraction=.05,pad=.30,label=r'$W/(M^2\Sigma)$')
        ax=axs[1];ax.set_xlim(-.22,1.3);ax.set_ylim(-.25,1.2);ax.set_aspect('equal');ax.axis('off')
        for angle,length,color,label in [(0,1.15,'#333333',r'$b$'),(30,1.1,'#bb4f18',r'$q_T$'),(70,1.1,'#247ba0',r'$k_T$')]:
            x,y=length*np.cos(np.deg2rad(angle)),length*np.sin(np.deg2rad(angle))
            ax.annotate('',(x,y),(0,0),arrowprops={'arrowstyle':'->','color':color,'lw':2})
            ax.text(x+.02,y+.02,label,color=color,fontsize=13)
        for diameter,theta1,theta2,color in [(.75,0,30,'#bb4f18'),(1.1,0,70,'#247ba0'),(1.5,30,70,'#777777')]:
            ax.add_patch(Arc((0,0),diameter,diameter,theta1=theta1,theta2=theta2,color=color,lw=1.3))
        ax.text(.39,.04,r'$\phi_q$',color='#bb4f18',fontsize=12)
        ax.text(.50,.30,r'$\psi$',color='#247ba0',fontsize=12)
        ax.text(.39,.67,r'$|\alpha|$',color='#777777',fontsize=12)
        ax.text(.52,-.18,r'$\alpha=\phi_q-\psi$',ha='center',fontsize=12)
        ax.set_title('(b)  Angle definitions\nIllustrative directions',fontsize=11)
        ax=axs[2]
        degrees=np.rad2deg(result['kt_psi'])
        rel=2*np.pi*result['kt_azimuth_density']
        ax.plot(degrees,rel,lw=2,color='#247ba0',label='Computed decay axis')
        ax.axhline(1,color='#999999',lw=1,ls='--',label='Uniform direction')
        ax.set_xlim(0,180);ax.set_ylim(0,1.65);ax.set_xticks([0,45,90,135,180])
        ax.set_xlabel(r'$\psi=\angle(k_T,b)$ (degrees)')
        ax.set_ylabel('Angular probability / uniform')
        ax.set_title('(c)  Broad direction distribution\nFull polar-angle acceptance',fontsize=11)
        ax.spines[['top','right']].set_visible(False)
        ax.legend(loc='lower center',fontsize=8,frameon=False)
        p1,p2=polarization(result['fixed_b_integrated_stokes'])
        fig.suptitle(result['case']['label']+rf' | fixed $b={settings["fixed_b_fm"]:g}$ fm',fontsize=13)
        fig.text(.5,.014,'Left: cyan circle marks the physical decay boundary; square corners are outside on-shell phase space.\n'
                 +rf'Right: $P(\psi)\propto 1-\frac{{1}}{{2}}({p1:.3f}\cos2\psi{p2:+.3f}\sin2\psi)$; '
                 r'the peak is broad, not a sharp estimate of $b$.',ha='center',fontsize=9)
        save(fig,f'{name}_angle_origin')
