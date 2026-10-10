"""Local unit bridge around the unchanged imported Torch physics model.

The upstream API computes qL=M/(2 gamma) exp(+-y) in GeV, whereas z and FFT
coordinates are fm and fm^-1. Physical mode converts qL only at those interfaces.
The original photon energy prefactor is retained; absolute-rate calibration is
outside this diagnostic's scope. Upstream mode is available for exact replay.
"""
import math
import torch
from .vendor.upc_forward import UPCForwardModel as ImportedUPCForwardModel


class UPCForwardModel(ImportedUPCForwardModel):
    def __init__(self, config, longitudinal_units='physical'):
        if longitudinal_units not in ('physical', 'upstream'):
            raise ValueError('longitudinal_units must be physical or upstream')
        self.longitudinal_units = longitudinal_units
        super().__init__(config)

    def _compute_T_prim_and_thickness(self, rho_total, qL, Z_nuc=None):
        inverse_length = qL / self.hbar_c if self.longitudinal_units == 'physical' else qL
        return super()._compute_T_prim_and_thickness(rho_total, inverse_length, Z_nuc=Z_nuc)

    def _compute_photon_field(self, F_3d_sorted, gamma_n, kz_value, Z_nuc=None):
        if self.longitudinal_units == 'upstream':
            return super()._compute_photon_field(F_3d_sorted, gamma_n, kz_value, Z_nuc=Z_nuc)
        vx, vy = super()._compute_photon_field(F_3d_sorted, gamma_n, kz_value / self.hbar_c, Z_nuc=Z_nuc)
        # Keep upstream 1/sqrt(qL_GeV) energy normalization, while the form factor,
        # propagator and Fourier phase use inverse-fm momenta consistently.
        return vx / math.sqrt(self.hbar_c), vy / math.sqrt(self.hbar_c)

    def fixed_b_empirical_incoherent_probability_multi_y(self, y_views, b_values,
                                                       *, R_1=None, R_2=None):
        """Legacy empirical incoherent yield, before b/orientation averaging.

        WS-only adapter, returning (batch,n_b,n_y). The same inherited density,
        field, thickness and quadrature routines as forward_multi_y are used;
        this avoids subtracting two nearly equal coherent+incoherent images.
        Isotropy in the decay-axis frame is imposed by the calling diagnostic.
        """
        if y_views.ndim != 2 or not b_values:
            raise ValueError('Expected y_views=(batch,n_y) and nonempty b nodes')
        if any(self._nuclear_configs[i]['density_model'] != 'woods_saxon' for i in (1,2)):
            raise ValueError('This diagnostic adapter supports Woods-Saxon densities only')
        batch, ny = y_views.shape
        rp1, rt1 = self._compute_density_ws(nuclear_ID=1, R=R_1, batch_size=batch)
        rp2, rt2 = self._compute_density_ws(nuclear_ID=2, R=R_2, batch_size=batch)
        f1 = self._expand_to_BVY(self._compute_F_3d(rp1), ny)
        f2 = self._expand_to_BVY(self._compute_F_3d(rp2), ny)
        rt1 = self._expand_to_BVY(rt1, ny); rt2 = self._expand_to_BVY(rt2, ny)
        ys = y_views.to(rt1).reshape(-1)
        q1 = self.Mjpsi*.5/self.gamma_1*torch.exp(ys)
        q2 = self.Mjpsi*.5/self.gamma_2*torch.exp(-ys)
        _, t1 = self._compute_T_prim_and_thickness(rt1, q1, Z_nuc=self.Z_1)
        _, t2 = self._compute_T_prim_and_thickness(rt2, q2, Z_nuc=self.Z_2)
        vx1, vy1 = self._compute_photon_field(f1, self.gamma_1, q1, Z_nuc=self.Z_1)
        vx2, vy2 = self._compute_photon_field(f2, self.gamma_2, q2, Z_nuc=self.Z_2)
        dummy = torch.zeros(batch*ny,1,3,device=rt1.device,dtype=rt1.dtype)
        width = torch.ones(batch*ny,device=rt1.device,dtype=rt1.dtype)
        old_b = self.b_imp; probabilities = []
        try:
            for b in b_values:
                if not math.isfinite(b) or b <= 0:raise ValueError('b must be finite and positive')
                self.b_imp = float(b)
                value = self._compute_incoherent_total_prob(dummy,width,ys,rt1,t1,t2,vx2,vy2,vx1,vy1)
                probabilities.append(value.reshape(batch,ny))
        finally:
            self.b_imp = old_b
        return torch.stack(probabilities,dim=1)
