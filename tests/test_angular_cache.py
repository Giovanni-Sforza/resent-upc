"""Raw-shard combination must preserve yield weighting and provenance."""
from copy import deepcopy
import numpy as np
import pytest
from upc_deformation.angular_cache import merge_orientation_shards


def shard(scale,angle):
    r=np.linspace(0,.3,7);axis=np.linspace(-.2,.2,9)
    orientations=np.zeros((2,3,2,3,len(r)))
    for i,weight in enumerate([1.,3.]):
        for stage in range(3):
            for branch in range(2):
                orientations[i,stage,branch,0]=scale*weight*np.exp(-r*(2+scale))
                orientations[i,stage,branch,1]=(.12 if branch==0 else -.03)*orientations[i,stage,branch,0]
    coeff=orientations.mean(0)
    x,y=np.meshgrid(axis,axis,indexing='ij')
    images=np.stack([np.stack([scale*(1+x*x+scale*y*y),scale*(1+scale*x*x+y*y)]) for _ in range(4)])
    disk=x*x+y*y<=.17**2;da=(axis[1]-axis[0])**2
    yields=(images*disk).sum((-2,-1))*da
    banks=[]
    for shift in [0.,.3]:
        mats=[]
        for a in [angle+shift,angle+shift+.1]:
            mats.append([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
        banks.append(np.array(mats))
    return {'case':{'name':'beta2','beta2':.3,'beta3':0.,'skin_fm':0.},
            'physics':{'unit':'fixture'},'density':{'R':5.},
            'stage_names':['b_axis','decay_fixed','decay_b','decay_by'],
            'axis':axis,'radii':r,'b_values':np.array([12.,60.]),'b_weights':np.ones(2),
            'y_values':np.array([0.]),'y_weights':np.array([2.]),
            'n_orientations':2,'orientation_coefficients':orientations,'coefficients':coeff,
            'node_coefficients':np.stack([coeff[0][None],coeff[0][None]]),
            'half_node_coefficients':np.zeros((2,1,2,3,len(r))),
            'coarse_b_coefficients':coeff[-1],
            'images':images,'yields':yields,'normalized_images':images/yields[...,None,None],
            'incoherent_probabilities':np.ones((2,1))*scale,
            'background_coefficients':np.zeros((3,len(r))),
            'rotation_bank_1':banks[0],'rotation_bank_2':banks[1],
            'elapsed_seconds':1.,'calculation_hash':'old','b_window_reuse':{'old':True}}


def test_combines_raw_intensity_not_normalized_images():
    a,b=shard(1.,0.),shard(4.,1.)
    old=deepcopy(a);merged=merge_orientation_shards(a,b)
    expected=.5*(a['images']+b['images'])
    np.testing.assert_array_equal(merged['images'],expected)
    np.testing.assert_allclose(merged['normalized_images'],expected/(.5*(a['yields']+b['yields']))[...,None,None])
    assert not np.allclose(merged['normalized_images'],.5*(a['normalized_images']+b['normalized_images']))
    np.testing.assert_array_equal(merged['half_node_coefficients'],a['node_coefficients'])
    np.testing.assert_allclose(merged['coefficients'],np.concatenate([a['orientation_coefficients'],b['orientation_coefficients']]).mean(0))
    np.testing.assert_array_equal(a['images'],old['images'])
    assert merged['n_orientations']==4 and 'calculation_hash' not in merged


@pytest.mark.parametrize('damage',['case','density','axis','count','mean','duplicate'])
def test_rejects_incompatible_or_duplicate_shards(damage):
    a,b=shard(1.,0.),shard(4.,1.)
    if damage=='case':b['case']['beta2']=.2
    elif damage=='density':b['density']['R']=6.
    elif damage=='axis':b['axis'][0]-=.001
    elif damage=='count':b['n_orientations']=3
    elif damage=='mean':b['coefficients']=b['coefficients']+1.
    elif damage=='duplicate':
        b['rotation_bank_1']=a['rotation_bank_1'].copy();b['rotation_bank_2']=a['rotation_bank_2'].copy()
    with pytest.raises(ValueError):merge_orientation_shards(a,b)
