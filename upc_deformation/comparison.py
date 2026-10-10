"""Small, reproducible forward comparisons using the local Torch model."""
from __future__ import annotations

import copy
import math
import time
import numpy as np
import torch
from .forward import UPCForwardModel
from .decay import (stokes_maps, decay_coefficients, decay_axis_image,
                    b_quadrature, normalize_in_disk)


def calculate_case(config: dict, case: dict, log=print) -> dict:
    settings = config['comparison']
    physics = copy.deepcopy(config['physics'])
    physics['nuclear_parameters'].update(beta2=case['beta2'], beta3=case['beta3'])
    physics['collision_parameters']['b'] = settings['fixed_b_fm']
    nuc=physics['nuclear_parameters']
    source_support=max(nuc['R_ws_p'],nuc['R_ws_n'])*(1+abs(case['beta2'])+.75*abs(case['beta3']))+8*max(nuc['d_ws_p'],nuc['d_ws_n'])
    if physics['grid_parameters']['output']['L']/2 <= settings['b_max_fm']+source_support:
        raise ValueError('Field grid does not cover b_max plus the nuclear support; increase L and N together')
    device = torch.device(settings.get('device', 'cpu'))
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('This complex FFT pipeline supports CPU or CUDA')
    model = UPCForwardModel(physics, settings.get('longitudinal_units', 'upstream')).to(device).eval()
    axis = model._p_axis_q
    radius = torch.linspace(0, float(axis.abs().max()) * math.sqrt(2) * (1 + 1e-6),
                            settings['n_radial'], device=device)
    b = torch.linspace(settings['b_min_fm'], settings['b_max_fm'], settings['b_nodes'],
                       dtype=torch.float64, device=device)
    if not bool(torch.isclose(b, torch.tensor(settings['fixed_b_fm'], device=device,
                                            dtype=b.dtype)).any()):
        raise ValueError('fixed_b_fm must be included in the integration grid')
    fixed_index = int(torch.argmin((b - settings['fixed_b_fm']).abs()))
    averaged = case['orientation'] == 'independent_average'
    n = settings['n_orientations'] if averaged else 1
    if averaged:
        # Independent rotations for the two nuclei; the same bank across cases/b.
        gen = torch.Generator(device='cpu').manual_seed(settings['seed'])
        r1 = model._sample_uniform_rotations(1, n, torch.device('cpu'), torch.float32, gen)[0].to(device)
        r2 = model._sample_uniform_rotations(1, n, torch.device('cpu'), torch.float32, gen)[0].to(device)
    else:
        r1 = r2 = torch.eye(3, device=device)[None]
    coefficients = torch.zeros(len(b), 3, len(radius), dtype=torch.float64, device=device)
    first_half = torch.zeros_like(coefficients)
    fixed_image = torch.zeros(len(axis), len(axis), dtype=torch.float64, device=device)
    fixed_stokes = torch.zeros(3, dtype=torch.float64, device=device)
    qx, qy = torch.meshgrid(axis, axis, indexing='ij')
    fixed_disk = qx.square() + qy.square() <= settings['normalization_radius_GeV']**2
    fixed_half = torch.zeros_like(fixed_image)
    half_count = 0
    start_time = time.monotonic()
    orientation_chunk = min(settings['orientation_chunk'], n)
    if n > 1 and n % (2 * orientation_chunk):
        raise ValueError('n_orientations must be divisible by twice orientation_chunk')
    b_chunk = settings['b_chunk']
    forward_closure = None
    with torch.inference_mode():
        for start in range(0, n, orientation_chunk):
            stop = min(start + orientation_chunk, n)
            count = stop - start
            ys = torch.full((count, 1), settings['y'], device=device)
            for bi in range(0, len(b), b_chunk):
                bj = min(bi + b_chunk, len(b))
                mx, my, p_full = model.fixed_b_complex_amplitude_multi_y(
                    None, None, ys, b[bi:bj].tolist(), R_1=r1[start:stop], R_2=r2[start:stop])
                maps = stokes_maps(mx[:, :, 0], my[:, :, 0])
                # Keep ensemble weights; no per-orientation or per-b normalization.
                mean_maps = maps.mean(0)
                coeff = decay_coefficients(mean_maps, p_full, radius, settings['n_phi']).double()
                coefficients[bi:bj] += coeff * count
                in_half = n > 1 and stop <= n // 2
                if in_half:
                    first_half[bi:bj] += coeff * count
                if bi <= fixed_index < bj:
                    projected, _ = model.project_full_grid_intensity_to_output(mean_maps[fixed_index - bi])
                    image = projected[0]
                    fixed_stokes += (projected.double()*fixed_disk).sum((-2,-1))*(axis[1]-axis[0])**2*count
                    fixed_image += image.double() * count
                    if in_half:
                        fixed_half += image.double() * count
                    if start == 0:
                        original, _ = model.forward_multi_y(None, y_views=ys, R_1=r1[start:stop],
                                                            R_2=r2[start:stop], incoherent=False)
                        expected = original[:, 0].mean(0)
                        forward_closure = float((image - expected).abs().max() / expected.abs().max())
                del mx, my, maps, mean_maps
                if time.monotonic() - start_time > settings.get('timeout_seconds', 1800):
                    raise TimeoutError('Per-case simulation timeout exceeded')
            if n > 1 and stop <= n // 2:
                half_count += count
            log(f"{case['name']}: orientations {stop}/{n}, {time.monotonic()-start_time:.1f}s", flush=True)
    coefficients /= n
    fixed_image /= n
    weights = b_quadrature(b)
    integrated = (coefficients * weights[:, None, None]).sum(0)
    at_fixed = coefficients[fixed_index]
    lepton_fixed = decay_axis_image(at_fixed, radius, axis)
    lepton_integrated = decay_axis_image(integrated, radius, axis)
    raw = torch.stack((fixed_image, lepton_fixed, lepton_integrated))
    normalized, yields = normalize_in_disk(raw, axis, settings['normalization_radius_GeV'])
    perp = torch.stack((fixed_image, decay_axis_image(at_fixed, radius, axis, 'perpendicular'),
                         decay_axis_image(integrated, radius, axis, 'perpendicular')))
    perp, _ = normalize_in_disk(perp, axis, settings['normalization_radius_GeV'])
    x, y = torch.meshgrid(axis, axis, indexing='ij')
    angle = torch.atan2(y, x)
    disk = x.square() + y.square() <= settings['normalization_radius_GeV'] ** 2
    v2 = (normalized * disk * (2 * angle).cos()).sum((-2, -1)) / (normalized * disk).sum((-2, -1))
    radial_weights = radius.double()
    radial_yields = 2 * math.pi * torch.trapezoid(coefficients[:, 0] * radial_weights,
                                                radius.double(), dim=-1)
    checks = {'forward_projection_relative_max_error': forward_closure,
              'min_intensity': float(raw.min()),
              'max_decay_modulation_bound_ratio': float((coefficients[:, 1:].square().sum(1).sqrt()
                                                       / coefficients[:, 0].clamp_min(1e-30)).max()),
              'fixed_axis_vs_decay_disk_yield_relative_difference': float(yields[1] / yields[0] - 1)}
    coarse = torch.arange(0, len(b), 2, device=device)
    if len(b) % 2 and len(coarse) > 1:
        coarse_coeff = (coefficients[coarse] * b_quadrature(b[coarse])[:, None, None]).sum(0)
        coarse_img = decay_axis_image(coarse_coeff, radius, axis)
        coarse_norm, coarse_yield = normalize_in_disk(coarse_img, axis, settings['normalization_radius_GeV'])
        checks['b_coarse_vs_full_shape_relative_l1'] = float(((coarse_norm-normalized[2]).abs()*disk).sum()
                                                         / (normalized[2]*disk).sum())
        checks['b_coarse_vs_full_yield_relative_difference'] = float(coarse_yield / yields[2] - 1)
    tail_start = settings['b_max_fm'] - 20
    tail = b >= tail_start
    if int(tail.sum()) >= 2:
        tail_coeff = (coefficients[tail] * b_quadrature(b[tail])[:, None, None]).sum(0)
        _, tail_yield = normalize_in_disk(decay_axis_image(tail_coeff, radius, axis), axis,
                                         settings['normalization_radius_GeV'])
        checks['last_20fm_yield_fraction'] = float(tail_yield / yields[2])
    if half_count:
        half_coeff = first_half / half_count
        half_integrated = (half_coeff * weights[:, None, None]).sum(0)
        half_images = torch.stack((fixed_half / half_count,
                                  decay_axis_image(half_coeff[fixed_index], radius, axis),
                                  decay_axis_image(half_integrated, radius, axis)))
        half_normalized, _ = normalize_in_disk(half_images, axis, settings['normalization_radius_GeV'])
        checks['half_vs_full_orientation_shape_relative_l1'] = (
            ((half_normalized - normalized).abs()*disk).sum((-2,-1))
            / (normalized*disk).sum((-2,-1))).tolist()
    if checks['min_intensity'] < -1e-8 * float(raw.max()) or checks['max_decay_modulation_bound_ratio'] > 1.0001:
        raise FloatingPointError('Decay density failed positivity bound')
    if forward_closure > 1e-5:
        raise AssertionError('Local observable does not reproduce the copied Torch forward')
    def array(t): return t.detach().cpu().numpy()
    result = {'case': case, 'axis': array(axis), 'radii': array(radius), 'b_values': array(b),
            'b_weights': array(weights), 'coefficients_per_b': array(coefficients),
            'radial_yields_per_b': array(radial_yields), 'raw_images': array(raw),
            'normalized_images': array(normalized), 'perpendicular_images': array(perp),
            'yields': array(yields), 'v2': array(v2), 'checks': checks,
            'elapsed_seconds': time.monotonic()-start_time, 'orientations': n,
            'rotation_bank_1': array(r1), 'rotation_bank_2': array(r2),
            'fixed_b_integrated_stokes': array(fixed_stokes/n)}
    from .kt import add_kt_arrays
    result['checks']['kt'] = add_kt_arrays(result, physics['physical_constants']['Mjpsi'])
    return result


def render_comparisons(results: list[dict], settings: dict, destination):
    if all('kt_density' in result for result in results):
        from .kt_figures import render_four_panel
        return render_four_panel(results, settings, destination)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':10,'svg.fonttype':'none','pdf.fonttype':42,
                         'axes.linewidth':.7,'savefig.facecolor':'white'})
    maximum = max(float(r['normalized_images'].max()) for r in results)
    names = ['Known b axis', 'Lepton decay axis', 'Lepton decay axis']
    btext = [f"b = {settings['fixed_b_fm']:g} fm", f"b = {settings['fixed_b_fm']:g} fm",
             f"{settings['b_min_fm']:g} < b < {settings['b_max_fm']:g} fm"]
    def save(fig, stem):
        for extension in ('png', 'pdf', 'svg'):
            fig.savefig(destination / f'{stem}.{extension}', dpi=190, bbox_inches='tight')
        plt.close(fig)
    for result in results:
        a = result['axis']; extent=[a[0],a[-1],a[0],a[-1]]
        for mode in ('linear', 'log', 'perpendicular'):
            images = result['perpendicular_images'] if mode=='perpendicular' else result['normalized_images']
            fig, axs = plt.subplots(1,3,figsize=(11.3,4.9),sharey=True)
            fig.subplots_adjust(left=.065,right=.885,bottom=.21,top=.75,wspace=.16)
            norm = LogNorm(maximum*1e-4,maximum) if mode=='log' else Normalize(0,maximum)
            for i,(ax,im) in enumerate(zip(axs,images)):
                artist=ax.imshow(im.T,origin='lower',extent=extent,cmap='magma',norm=norm,interpolation='nearest')
                reference=names[i] if mode!='perpendicular' or i==0 else 'Axis perpendicular to leptons'
                ax.set_title(f'{chr(97+i)}   {reference}\n{btext[i]}',fontsize=10)
                ax.set_xlabel(r'$p_x$ (GeV/$c$)')
                ax.set_xticks([-.15,0,.15]); ax.set_yticks([-.15,0,.15])
                ax.set_aspect('equal')
            axs[0].set_ylabel(r'$p_y$ (GeV/$c$)')
            cax=fig.add_axes([.906,.255,.017,.445])
            fig.colorbar(artist,cax=cax,label=r'Normalized density [(GeV/$c$)$^{-2}$]')
            fig.suptitle(f"$^{{96}}$Zr+$^{{96}}$Zr, 200 GeV, $y=0$ | {result['case']['label']}\n"
                         +rf"$\beta_2={result['case']['beta2']:g}$, $\beta_3={result['case']['beta3']:g}$; coherent $J/\psi$",fontsize=12)
            units = 'qL units checked' if settings.get('longitudinal_units') == 'physical' else 'upstream qL convention'
            fig.text(.5,.028,'Full decay-angle acceptance; Born level; '+units+'\nCommon color scale; unit yield inside $p_T<0.17$ GeV/$c$',ha='center',va='bottom',fontsize=8)
            save(fig, f"{result['case']['name']}_{mode}")
    fig,axs=plt.subplots(1,len(results),figsize=(4*len(results),3.6),layout='constrained',squeeze=False)
    colors=['#222222','#2985b5','#d06936']
    for result,ax in zip(results,axs[0]):
        a=result['axis']; mid=len(a)//2
        for i,image in enumerate(result['normalized_images']):
            ax.plot(a,image[:,mid],color=colors[i],label=['Fixed b, known axis','Fixed b, decay axis','Integrated b, decay axis'][i])
            ax.plot(a,image[mid,:],color=colors[i],ls='--',alpha=.85)
        ax.set_title(result['case']['label'],fontsize=10)
        ax.set_xlabel(r'Momentum along slice (GeV/$c$)')
        ax.set_xlim(-.17,.17);ax.set_ylim(bottom=0)
        ax.spines[['top','right']].set_visible(False)
    axs[0,0].set_ylabel('Normalized density')
    axs[0,-1].legend(fontsize=8)
    fig.suptitle('Central slices: solid = $p_y=0$; dashed = $p_x=0$',fontsize=12)
    save(fig,'central_slices')
