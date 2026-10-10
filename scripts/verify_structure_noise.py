#!/usr/bin/env python3
"""Offline numerical and export checks for the structure/noise comparison."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import fitz
import numpy as np
from upc_deformation.structure_noise import masked_ratio


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='outputs/jpsi_structure_noise')
    args=parser.parse_args();root=Path(args.output).resolve()
    summary=json.loads((root/'summary.json').read_text())
    settings=summary['config']['comparison'];results=[];checks={}
    for name in ['spherical','deformed','skin']:
        with np.load(root/f'{name}.npz') as data:r={k:data[k] for k in data.files}
        results.append(r);raw=r['raw_images'];axis=r['axis']
        x,y=np.meshgrid(axis,axis,indexing='ij');disk=x*x+y*y<=settings['normalization_radius_GeV']**2
        assert np.isfinite(raw).all() and raw.min()>=0
        np.testing.assert_allclose(raw[1],raw[0]+r['incoherent_image'],rtol=2e-15)
        mean=r['coherent_images_per_orientation'].astype(float).mean(0)
        mean_error=np.abs(mean-raw[0]).max()/raw[0].max()
        assert mean_error<1e-6
        dx=float(axis[1]-axis[0]);yields=(r['normalized_images']*disk).sum((1,2))*dx*dx
        np.testing.assert_allclose(yields,1,atol=1e-6)
        inc=r['incoherent_image'];symmetry=np.abs(inc-inc.T).max()/inc.max()
        assert symmetry<1e-7
        for bank in ['rotation_bank_1','rotation_bank_2']:
            rotations=r[bank].astype(float)
            np.testing.assert_allclose(rotations@rotations.transpose(0,2,1),np.broadcast_to(np.eye(3),rotations.shape),atol=8e-7)
            np.testing.assert_allclose(np.linalg.det(rotations),1,atol=8e-7)
        checks[name]={'orientation_average_relative_max_error':float(mean_error),
                      'unit_disk_yields':yields.tolist(),'incoherent_transpose_relative_max_error':float(symmetry)}
    with np.load(root/'ratios.npz') as data:
        for row in range(2):
            for i,col in enumerate([1,2]):
                expected,keep=masked_ratio(results[col]['normalized_images'][row],results[0]['normalized_images'][row],
                    results[0]['axis'],settings['normalization_radius_GeV'],settings['ratio_floor_fraction'])
                np.testing.assert_array_equal(data['ratio_valid_mask'][row,i],keep)
                np.testing.assert_allclose(data['shape_ratios'][row,i],expected,equal_nan=True,rtol=1e-14)
                key=f'row{row+1}_ratio{i+1}'
                checks[key]={'range':[float(np.nanmin(expected)),float(np.nanmax(expected))],
                             'valid_pixel_count':int(keep.sum())}
    skin=summary['cases'][2]['density_rms_first_orientation']
    for nucleus in skin:assert abs(nucleus['skin_fm']-summary['config']['structure_study']['skin_rms_fm'])<2e-5
    exports=[]
    for stem in ['structure_noise_main','structure_noise_log','deformed_orientation_gallery','radial_and_v2']:
        for extension in ['png','pdf','svg']:assert (root/f'{stem}.{extension}').exists()
        with fitz.open(root/f'{stem}.pdf') as document:
            assert len(document)==1
            text=document[0].get_text()
            if stem.startswith('structure_noise'):
                for letter in 'abcdefghij':assert f'({letter})' in text
            elif stem=='deformed_orientation_gallery':
                assert f'Mean of {settings["n_orientations"]}' in text
        assert '<text' in (root/f'{stem}.svg').read_text()
        exports.append(stem)
    checks['exports']=exports
    checks['interpretation']='Numerical closure and finite-sample diagnostics; not detector calibration or training validation'
    (root/'qa.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
