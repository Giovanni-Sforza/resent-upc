"""Scalable evaluation of the existing Cartesian panel-d observable.

All pixel reconstruction, circular bilinear sampling and unnormalized moments
are linear in the saved radial coefficients. Sparse matrices compose those
same operations once; they do not replace panel d with a harmonic-ratio proxy.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from .c2_convergence import project_orientation_moments, ratio_statistics, summarize_orientation_moments
from .spectral_weights import validate_edges


class PixelMomentProjector:
    """Linear maps for the archived pixel, circle and three-annulus definitions."""

    def __init__(self, radii, axis, q, angles, edges=(.03, .08, .13, .17)):
        self.radii, self.axis, self.q, self.angles = [np.asarray(a, dtype=float) for a in (radii, axis, q, angles)]
        self.edges = validate_edges(edges)
        r, a, q, phi = self.radii, self.axis, self.q, self.angles
        if (r.ndim != 1 or len(r) < 2 or r[0] < 0 or np.any(np.diff(r) <= 0)
                or a.ndim != 1 or len(a) < 3 or np.any(np.diff(a) <= 0)
                or not np.allclose(np.diff(a), np.diff(a)[0], rtol=3e-5, atol=0)
                or q.ndim != 1 or len(q) < 2 or q[0] < 0 or np.any(np.diff(q) <= 0)
                or q[-1] > min(-a[0], a[-1]) or self.edges[-1] > q[-1]
                or phi.ndim != 1 or len(phi) < 4
                or not np.allclose(phi, np.arange(len(phi))*2*np.pi/len(phi), rtol=0, atol=1e-12)
                or any(not np.isfinite(v).all() for v in (r, a, q, phi))):
            raise ValueError('invalid radial, Cartesian, circle or angular grid')
        xx, yy = np.meshgrid(a, a, indexing='ij')
        rr = np.sqrt(xx*xx + yy*yy).ravel()
        if rr.min() < r[0] or rr.max() > r[-1]:
            raise ValueError('radial coefficients must cover every Cartesian corner')
        alpha = np.arctan2(yy, xx).ravel()
        n, nr = len(a), len(r)
        ri = np.searchsorted(r, rr, side='left').clip(1, nr-1)
        f = (rr-r[ri-1])/(r[ri]-r[ri-1])
        rows = np.arange(n*n)
        pr, pc, pv = [], [], []
        for channel, factor in enumerate((np.ones(n*n), -.5*np.cos(2*alpha), -.5*np.sin(2*alpha))):
            for offset, weight in ((-1, 1-f), (0, f)):
                pr.append(rows); pc.append(channel*nr+ri+offset); pv.append(factor*weight)
        self.pixel_map = sparse.csr_matrix((np.concatenate(pv), (np.concatenate(pr), np.concatenate(pc))), shape=(n*n, 3*nr))

        cx = (q[:, None]*np.cos(phi)).ravel()
        cy = (q[:, None]*np.sin(phi)).ravel()
        ix = np.searchsorted(a, cx, side='right').clip(1, n-1)-1
        iy = np.searchsorted(a, cy, side='right').clip(1, n-1)-1
        fx, fy = (cx-a[ix])/(a[ix+1]-a[ix]), (cy-a[iy])/(a[iy+1]-a[iy])
        sr, sc, sv = [], [], []
        for dx, dy, weight in ((0, 0, (1-fx)*(1-fy)), (0, 1, (1-fx)*fy), (1, 0, fx*(1-fy)), (1, 1, fx*fy)):
            sr.append(np.arange(len(cx))); sc.append((ix+dx)*n+iy+dy); sv.append(weight)
        sampling = sparse.csr_matrix((np.concatenate(sv), (np.concatenate(sr), np.concatenate(sc))), shape=(len(cx), n*n))
        circle_pixels = sampling @ self.pixel_map
        angle_index = np.tile(np.arange(len(phi)), len(q))
        circle_index = np.repeat(np.arange(len(q)), len(phi))
        reduce_maps = []
        for factor in (np.ones(len(phi)), np.cos(2*phi), np.sin(2*phi)):
            reduce_maps.append(sparse.csr_matrix((factor[angle_index]/len(phi), (circle_index, np.arange(len(cx)))), shape=(len(q), len(cx))))
        ring_map = sparse.vstack([reducer @ circle_pixels for reducer in reduce_maps], format='csr')
        q2 = xx*xx+yy*yy
        cos2 = np.divide(xx*xx-yy*yy, q2, out=np.zeros_like(q2), where=q2 > 0)
        sin2 = np.divide(2*xx*yy, q2, out=np.zeros_like(q2), where=q2 > 0)
        masks = [(q2 >= lo*lo) & ((q2 <= hi*hi) if i == 2 else q2 < hi*hi)
                 for i, (lo, hi) in enumerate(zip(self.edges[:-1], self.edges[1:]))]
        if any(not m.any() for m in masks):
            raise ValueError('every diagnostic annulus must contain pixels')
        area = float(a[1]-a[0])**2
        annuli = sparse.csr_matrix(np.asarray([(m*weight*area).ravel() for weight in (np.ones_like(xx), cos2, sin2) for m in masks]))
        self.moment_map = sparse.vstack((ring_map, annuli @ self.pixel_map), format='csr')
        self.disk = q2 <= q[-1]**2
        self.pixel_area = area

    def project(self, coefficients, chunk_size=256):
        c = np.asarray(coefficients, float)
        if (c.ndim != 3 or c.shape[1:] != (3, len(self.radii)) or not len(c)
                or not np.isfinite(c).all() or chunk_size < 1):
            raise ValueError('finite (orientation,3,radius) coefficients are required')
        # Positivity of the continuous kernel implies positivity after both
        # radial and Cartesian interpolation. Allow only roundoff-scale slack.
        lower = c[:, 0]-.5*np.hypot(c[:, 1], c[:, 2])
        if np.min(lower) < -1e-12*max(float(np.max(np.abs(c))), np.finfo(float).tiny):
            raise ValueError('coefficients imply a negative angular intensity')
        values = np.concatenate([(self.moment_map @ c[i:i+chunk_size].reshape(-1, 3*len(self.radii)).T).T
                                 for i in range(0, len(c), chunk_size)], axis=0)
        nq = len(self.q)
        result = {}
        for j, key in enumerate(('orientation_denominator', 'orientation_C2_numerator', 'orientation_S2_numerator')):
            result[key] = values[:, j*nq:(j+1)*nq]
        for j, key in enumerate(('bin_orientation_denominator', 'bin_orientation_C2_numerator', 'bin_orientation_S2_numerator')):
            result[key] = values[:, 3*nq+3*j:3*nq+3*j+3]
        if np.any(result['orientation_denominator'] <= 0) or np.any(result['bin_orientation_denominator'] <= 0):
            raise ValueError('nonpositive circle or annular yield')
        mean = c.mean(0)
        image = (self.pixel_map @ mean.ravel()).reshape(len(self.axis), len(self.axis))
        total = float(image[self.disk].sum()*self.pixel_area)
        result.update(mean_raw_image=image, mean_normalized_image=image/total,
                      normalization_yield=total, mean_coefficients=mean)
        return result

    def check_direct(self, coefficients, atol=2e-12):
        """Regression against the unchanged pixel-by-pixel implementation."""
        c = np.asarray(coefficients, float)[:2]
        fast = self.project(c)
        direct = project_orientation_moments(c, self.radii, self.axis, self.q, self.angles, self.edges)
        errors = {}
        for key in ('orientation_denominator', 'orientation_C2_numerator', 'orientation_S2_numerator',
                    'bin_orientation_denominator', 'bin_orientation_C2_numerator', 'bin_orientation_S2_numerator', 'mean_raw_image'):
            scale = max(float(np.max(np.abs(direct[key]))), np.finfo(float).tiny)
            error = float(np.max(np.abs(fast[key]-direct[key]))/scale)
            # Tiny sine moments are judged against the corresponding yield,
            # since their exact physical value can be zero.
            if 'numerator' in key:
                denkey = 'bin_orientation_denominator' if key.startswith('bin_') else 'orientation_denominator'
                error = float(np.max(np.abs(fast[key]-direct[key]))/np.max(direct[denkey]))
            errors[key] = error
        c_fast = fast['orientation_C2_numerator']/fast['orientation_denominator']
        c_direct = direct['orientation_C2_numerator']/direct['orientation_denominator']
        errors['C2_max_absolute'] = float(np.max(np.abs(c_fast-c_direct)))
        if max(errors.values()) > atol:
            raise ValueError(f'Cartesian projection closure failed: {errors}')
        return errors


def moment_estimate(moments, prefixes=(32, 128, 512, 2048, 8192, 32768)):
    n = len(moments['orientation_denominator'])
    prefixes = sorted(set([p for p in prefixes if 2 <= p <= n]+([n] if n > 1 else [])))
    result = summarize_orientation_moments(moments, prefixes)
    den = moments['bin_orientation_denominator']
    result['spectral'] = ratio_statistics(den[:, 1:], den[:, :-1])
    return result


def paired_ratio_difference(num_a, den_a, num_b, den_b):
    """Difference of mean ratios; paired influence covariance, never mean ratios."""
    num_a, den_a, num_b, den_b = [np.asarray(v, float) for v in (num_a, den_a, num_b, den_b)]
    a, b = ratio_statistics(num_a, den_a), ratio_statistics(num_b, den_b)
    if len(num_a) != len(num_b):
        raise ValueError('paired estimates require identical orientation counts')
    ua = (num_a-a['value']*den_a)/a['mean_denominator']
    ub = (num_b-b['value']*den_b)/b['mean_denominator']
    u = ua-ub
    n = len(u)
    cov = np.atleast_2d(np.cov(u, rowvar=False, ddof=1))/n if n > 1 else np.zeros((u.shape[1], u.shape[1]))
    return {'value': a['value']-b['value'], 'orientation_se': np.sqrt(np.maximum(0, np.diag(cov))), 'covariance': cov}
