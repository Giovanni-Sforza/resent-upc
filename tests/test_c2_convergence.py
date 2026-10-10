import numpy as np
import pytest

from upc_deformation.c2_convergence import (ratio_statistics,
    project_orientation_moments, summarize_orientation_moments)


def test_raw_means_are_divided_after_averaging_and_scale_cancels():
    den=np.array([[1.,2.],[9.,3.],[4.,8.]])
    num=np.array([[.4,.3],[-.6,.8],[.7,-.2]])
    result=ratio_statistics(num,den)
    np.testing.assert_allclose(result['value'],num.mean(0)/den.mean(0))
    assert not np.allclose(result['value'],(num/den).mean(0))
    scaled=ratio_statistics(12.3*num,12.3*den)
    np.testing.assert_allclose(result['value'],scaled['value'])
    np.testing.assert_allclose(result['covariance'],scaled['covariance'])


def test_joint_covariance_retains_numerator_denominator_and_momentum_correlations():
    den=np.array([[3.,2.],[2.,5.],[4.,3.],[5.,6.]])
    num=np.array([[.8,.4],[.3,.9],[-.2,.7],[.1,-.4]])
    result=ratio_statistics(num,den)
    d=den.mean(0);n=num.mean(0)
    samples=np.column_stack((num,den))
    jac=np.hstack((np.diag(1/d),np.diag(-n/d**2)))
    expected=jac@np.cov(samples,rowvar=False,ddof=1)@jac.T/len(num)
    np.testing.assert_allclose(result['covariance'],expected,rtol=1e-13,atol=1e-17)
    np.testing.assert_allclose(result['orientation_se']**2,np.diag(expected))
    assert abs(expected[0,1])>0
    single=ratio_statistics(num[:1],den[:1])
    np.testing.assert_array_equal(single['covariance'],np.zeros((2,2)))


def test_prefixes_use_nested_indices_but_halves_and_blocks_are_disjoint():
    den=np.arange(1.,17.).reshape(8,2)
    num=den*np.arange(8.)[:,None]/10
    moments={'orientation_denominator':den,'orientation_C2_numerator':num,
             'orientation_S2_numerator':num/2,
             'bin_orientation_denominator':np.column_stack((den,den[:,0])),
             'bin_orientation_C2_numerator':np.column_stack((num,num[:,0])),
             'bin_orientation_S2_numerator':np.column_stack((num/2,num[:,0]/2))}
    out=summarize_orientation_moments(moments,(2,4,8))
    np.testing.assert_allclose(out['nested']['4']['C2'],num[:4].mean(0)/den[:4].mean(0))
    np.testing.assert_allclose(out['halves'][1]['C2'],num[4:].mean(0)/den[4:].mean(0))
    assert [(e['sample_start'],e['sample_stop']) for e in out['halves']]==[(0,4),(4,8)]
    assert [(e['sample_start'],e['sample_stop']) for e in out['blocks']]==[(0,2),(2,4),(4,6),(6,8)]
    half_se=np.sqrt(out['halves'][0]['orientation_se']**2+out['halves'][1]['orientation_se']**2)
    np.testing.assert_allclose(out['half_difference_se'],half_se)
    with pytest.raises(ValueError):summarize_orientation_moments(moments,(4,2,8))


def test_isotropic_pixels_give_zero_angular_moment_and_normalized_mean_image():
    axis=np.linspace(-.2,.2,41);radii=np.linspace(0,.3,51)
    coeff=np.zeros((4,3,len(radii)));coeff[:,0]=np.arange(1.,5.)[:,None]
    q=np.linspace(0,.17,35);angles=np.arange(72)*2*np.pi/72
    out=project_orientation_moments(coeff,radii,axis,q,angles)
    np.testing.assert_allclose(out['orientation_C2_numerator'],0.,atol=1e-14)
    np.testing.assert_allclose(out['bin_orientation_C2_numerator'],0.,atol=1e-14)
    np.testing.assert_allclose(out['mean_raw_image'],2.5)
    x,y=np.meshgrid(axis,axis,indexing='ij');disk=x*x+y*y<=.17**2
    area=(axis[1]-axis[0])**2
    np.testing.assert_allclose(out['mean_normalized_image'][disk].sum()*area,1.)


def test_negative_images_and_nonpositive_ratio_denominators_are_rejected():
    axis=np.linspace(-.2,.2,41);radii=np.linspace(0,.3,51)
    coeff=np.ones((1,3,len(radii)));coeff[:,1]=4.;coeff[:,2]=0.
    with pytest.raises(ValueError,match='nonnegative'):
        project_orientation_moments(coeff,radii,axis,np.linspace(0,.17,35),np.arange(72)*2*np.pi/72)
    with pytest.raises(ValueError,match='positive denominators'):
        ratio_statistics([[1.,2.]],[[1.,0.]])
