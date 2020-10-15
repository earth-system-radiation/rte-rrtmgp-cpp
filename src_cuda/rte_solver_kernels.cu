#include <chrono>

#include "rte_kernel_launcher_cuda.h"
#include "tools_gpu.h"
#include "Array.h"
#include <iomanip>

namespace
{
    template<typename TF>__device__
    void lw_source_noscat_kernel(const int icol, const int igpt, const int ncol, const int nlay, const int ngpt, const TF eps,
                                 const TF* __restrict__ lay_source, const TF* __restrict__ lev_source_up, const TF* __restrict__ lev_source_dn,
                                 const TF* __restrict__ tau, const TF* __restrict__ trans, TF* __restrict__ source_dn, TF* __restrict__ source_up)
    {
        const TF tau_thres = sqrt(eps);
        for (int ilay=0; ilay<nlay; ++ilay)
        {
            const int idx = icol + ilay*ncol + igpt*ncol*nlay;
            const TF fact = (tau[idx]>tau_thres) ? (TF(1.) - trans[idx]) / tau[idx] - trans[idx] : tau[idx] * (TF(.5) - TF(1.)/TF(3.)*tau[idx]);
            source_dn[idx] = (TF(1.) - trans[idx]) * lev_source_dn[idx] + TF(2.) * fact * (lay_source[idx]-lev_source_dn[idx]);
            source_up[idx] = (TF(1.) - trans[idx]) * lev_source_up[idx] + TF(2.) * fact * (lay_source[idx]-lev_source_up[idx]);
        }
    }

    template<typename TF>__device__
    void lw_transport_noscat_kernel(const int icol, const int igpt, const int ncol, const int nlay, const int ngpt, const BOOL_TYPE top_at_1,
                                 const TF* __restrict__ tau, const TF* __restrict__ trans, const TF* __restrict__ sfc_albedo,
                                 const TF* __restrict__ source_dn, const TF* __restrict__ source_up, const TF* __restrict__ source_sfc,
                                 TF* __restrict__ radn_up, TF* __restrict__ radn_dn, const TF* __restrict__ source_sfc_jac, TF* __restrict__ radn_up_jac)
    {
        if (top_at_1)
        {
            for (int ilev=1; ilev<(nlay+1); ++ilev)
            {
                const int idx1 = icol + ilev*ncol + igpt*ncol*(nlay+1);
                const int idx2 = icol + (ilev-1)*ncol + igpt*ncol*(nlay+1);
                const int idx3 = icol + (ilev-1)*ncol + igpt*ncol*nlay;
                radn_dn[idx1] = trans[idx3] * radn_dn[idx2] + source_dn[idx3];
            }

            const int idx_bot = icol + nlay*ncol + igpt*ncol*(nlay+1);
            const int idx2d = icol + igpt*ncol;
            radn_up[idx_bot] = radn_dn[idx_bot] * sfc_albedo[idx2d] + source_sfc[idx2d];
            radn_up_jac[idx_bot] = source_sfc_jac[idx2d];

            for (int ilev=nlay-1; ilev>=0; --ilev)
            {
                const int idx1 = icol + ilev*ncol + igpt*ncol*(nlay+1);
                const int idx2 = icol + (ilev+1)*ncol + igpt*ncol*(nlay+1);
                const int idx3 = icol + ilev*ncol + igpt*ncol*nlay;
                radn_up[idx1] = trans[idx3] * radn_up[idx2] + source_up[idx3];
                radn_up_jac[idx1] = trans[idx3] * radn_up_jac[idx2];
            }
        }
        else
        {
            for (int ilev=(nlay-1); ilev>=0; --ilev)
            {
                const int idx1 = icol + ilev*ncol + igpt*ncol*(nlay+1);
                const int idx2 = icol + (ilev+1)*ncol + igpt*ncol*(nlay+1);
                const int idx3 = icol + ilev*ncol + igpt*ncol*nlay;
                radn_dn[idx1] = trans[idx3] * radn_dn[idx2] + source_dn[idx3];
            }

            const int idx_bot = icol + igpt*ncol*(nlay+1);
            const int idx2d = icol + igpt*ncol;
            radn_up[idx_bot] = radn_dn[idx_bot] * sfc_albedo[idx2d] + source_sfc[idx2d];
            radn_up_jac[idx_bot] = source_sfc_jac[idx2d];

            for (int ilev=1; ilev<(nlay+1); ++ilev)
            {
                const int idx1 = icol + ilev*ncol + igpt*ncol*(nlay+1);
                const int idx2 = icol + (ilev-1)*ncol + igpt*ncol*(nlay+1);
                const int idx3 = icol + (ilev-1)*ncol + igpt*ncol*nlay;
                radn_up[idx1] = trans[idx3] * radn_up[idx2] + source_up[idx3];
                radn_up_jac[idx1] = trans[idx3] * radn_up_jac[idx2];;
            }
        }

    }

    template<typename TF>__device__
    void lw_solver_noscat_kernel(const int icol, const int igpt, const int ncol, const int nlay, const int ngpt, const TF eps, const BOOL_TYPE top_at_1,
                                 const TF D, const TF weight, const TF* __restrict__ tau, const TF* __restrict__ lay_source,
                                 const TF* __restrict__ lev_source_inc, const TF* __restrict__ lev_source_dec, const TF* __restrict__ sfc_emis,
                                 const TF* __restrict__ sfc_src, TF* __restrict__ radn_up, TF* __restrict__ radn_dn,
                                 const TF* __restrict__ sfc_src_jac, TF* __restrict__ radn_up_jac, TF* __restrict__ tau_loc,
                                 TF* __restrict__ trans, TF* __restrict__ source_dn, TF* __restrict__ source_up,
                                 TF* __restrict__ source_sfc, TF* __restrict__ sfc_albedo, TF* __restrict__ source_sfc_jac)
    {
        const TF pi = acos(TF(-1.));
        const TF* lev_source_up;
        const TF* lev_source_dn;
        int top_level;
        if (top_at_1)
        {
            top_level = 0;
            lev_source_up = lev_source_dec;
            lev_source_dn = lev_source_inc;
        }
        else
        {
            top_level = nlay;
            lev_source_up = lev_source_inc;
            lev_source_dn = lev_source_dec;
        }
        const int idx_top = icol + top_level*ncol + igpt*ncol*(nlay+1);
        radn_dn[idx_top] = radn_dn[idx_top] / (TF(2.) * pi * weight);

        const int idx2d = icol + igpt*ncol;

        for (int ilay=0; ilay<nlay; ++ilay)
        {
            const int idx3d = icol + ilay*ncol + igpt*ncol*nlay;
            tau_loc[idx3d] = tau[idx3d] * D;
            trans[idx3d]   = exp(-tau_loc[idx3d]);
        }

        lw_source_noscat_kernel(icol, igpt, ncol, nlay, ngpt, eps, lay_source, lev_source_up, lev_source_dn,
                         tau_loc, trans, source_dn, source_up);

        sfc_albedo[idx2d] = TF(1.) - sfc_emis[idx2d];
        source_sfc[idx2d] = sfc_emis[idx2d] * sfc_src[idx2d];
        source_sfc_jac[idx2d] = sfc_emis[idx2d] * sfc_src_jac[idx2d];

        lw_transport_noscat_kernel(icol, igpt, ncol, nlay, ngpt, top_at_1, tau, trans, sfc_albedo, source_dn,
                                   source_up, source_sfc, radn_up, radn_dn, source_sfc_jac, radn_up_jac);

        for (int ilev=0; ilev<(nlay+1); ++ilev)
        {
            const int idx = icol + ilev*ncol + igpt*ncol*(nlay+1);
            radn_up[idx] *= TF(2.) * pi * weight;
            radn_dn[idx] *= TF(2.) * pi * weight;
            radn_up_jac[idx] *= TF(2.) * pi * weight;
        }
    }

    template<typename TF>__device__
    void sw_adding_kernel(const int icol, const int igpt,
                          const int ncol, const int nlay, const BOOL_TYPE top_at_1,
                          const TF* __restrict__ sfc_alb_dif, const TF* __restrict__ r_dif, const TF* __restrict__ t_dif,
                          const TF* __restrict__ source_dn, const TF* __restrict__ source_up, const TF* __restrict__ source_sfc,
                          TF* __restrict__ flux_up, TF* __restrict__ flux_dn, const TF* __restrict__ flux_dir,
                          TF* __restrict__ albedo, TF* __restrict__ src, TF* __restrict__ denom)
    {
        if (top_at_1)
        {
            const int sfc_idx_3d = icol + nlay*ncol + igpt*(nlay+1)*ncol;
            const int sfc_idx_2d = icol + igpt*ncol;
            albedo[sfc_idx_3d] = sfc_alb_dif[sfc_idx_2d];
            src[sfc_idx_3d] = source_sfc[sfc_idx_2d];

            for (int ilay=nlay-1; ilay >= 0; --ilay)
            {
                const int lay_idx  = icol + ilay*ncol + igpt*ncol*nlay;
                const int lev_idx1 = icol + ilay*ncol + igpt*ncol*(nlay+1);
                const int lev_idx2 = icol + (ilay+1)*ncol + igpt*ncol*(nlay+1);
                denom[lay_idx] = TF(1.)/(TF(1.) - r_dif[lay_idx] * albedo[lev_idx2]);
                albedo[lev_idx1] = r_dif[lay_idx] + t_dif[lay_idx] * t_dif[lay_idx]
                                                  * albedo[lev_idx2] * denom[lay_idx];
                src[lev_idx1] = source_up[lay_idx] + t_dif[lay_idx] * denom[lay_idx] *
                                (src[lev_idx2] + albedo[lev_idx2] * source_dn[lay_idx]);
            }
            const int top_idx = icol + igpt*(nlay+1)*ncol;
            flux_up[top_idx] = flux_dn[top_idx]*albedo[top_idx] + src[top_idx];

            for (int ilay=1; ilay < (nlay+1); ++ilay)
            {
                const int lev_idx1 = icol + ilay*ncol + igpt*(nlay+1)*ncol;
                const int lev_idx2 = icol + (ilay-1)*ncol + igpt*(nlay+1)*ncol;
                const int lay_idx = icol + (ilay-1)*ncol + igpt*(nlay)*ncol;
                flux_dn[lev_idx1] = (t_dif[lay_idx]*flux_dn[lev_idx2] +
                                     r_dif[lay_idx]*src[lev_idx1] +
                                     source_dn[lay_idx]) * denom[lay_idx];
                flux_up[lev_idx1] = flux_dn[lev_idx1] * albedo[lev_idx1] + src[lev_idx1];
            }

            for (int ilay=0; ilay < (nlay+1); ++ilay)
            {
                const int idx = icol + ilay*ncol + igpt*(nlay+1)*ncol;
                flux_dn[idx] += flux_dir[idx];
            }
        }
        else
        {
            const int sfc_idx_3d = icol + igpt*(nlay+1)*ncol;
            const int sfc_idx_2d = icol + igpt*ncol;
            albedo[sfc_idx_3d] = sfc_alb_dif[sfc_idx_2d];
            src[sfc_idx_3d] = source_sfc[sfc_idx_2d];

            for (int ilay=0; ilay<nlay; ++ilay)
            {
                const int lay_idx  = icol + ilay*ncol + igpt*ncol*nlay;
                const int lev_idx1 = icol + ilay*ncol + igpt*ncol*(nlay+1);
                const int lev_idx2 = icol + (ilay+1)*ncol + igpt*ncol*(nlay+1);
                denom[lay_idx] = TF(1.)/(TF(1.) - r_dif[lay_idx] * albedo[lev_idx1]);
                albedo[lev_idx2] = r_dif[lay_idx] + (t_dif[lay_idx] * t_dif[lay_idx] *
                                                     albedo[lev_idx1] * denom[lay_idx]);
                src[lev_idx2] = source_up[lay_idx] + t_dif[lay_idx]*denom[lay_idx]*
                                                     (src[lev_idx1]+albedo[lev_idx1]*source_dn[lay_idx]);
            }
            const int top_idx = icol + nlay*ncol + igpt*(nlay+1)*ncol;
            flux_up[top_idx] = flux_dn[top_idx] *albedo[top_idx] + src[top_idx];

            for (int ilay=nlay-1; ilay >= 0; --ilay)
            {
                    const int lev_idx1 = icol + ilay*ncol + igpt*(nlay+1)*ncol;
                    const int lev_idx2 = icol + (ilay+1)*ncol + igpt*(nlay+1)*ncol;
                    const int lay_idx = icol + ilay*ncol + igpt*nlay*ncol;
                    flux_dn[lev_idx1] = (t_dif[lay_idx]*flux_dn[lev_idx2] +
                                         r_dif[lay_idx]*src[lev_idx1] +
                                         source_dn[lay_idx]) * denom[lay_idx];
                    flux_up[lev_idx1] = flux_dn[lev_idx1] * albedo[lev_idx1] + src[lev_idx1];
            }
            for (int ilay=nlay; ilay >= 0; --ilay)
            {
                const int idx = icol + ilay*ncol + igpt*(nlay+1)*ncol;
                flux_dn[idx] += flux_dir[idx];
            }
        }
    }

    template<typename TF>__device__
    void sw_source_kernel(const int icol, const int igpt,
                          const int ncol, const int nlay, const BOOL_TYPE top_at_1,
                          TF* __restrict__ r_dir, TF* __restrict__ t_dir, TF* __restrict__ t_noscat,
                          const TF* __restrict__ sfc_alb_dir, TF* __restrict__ source_up, TF* __restrict__ source_dn,
                          TF* __restrict__ source_sfc, TF* __restrict__ flux_dir)
    {

        if (top_at_1)
        {
            for (int ilay=0; ilay<nlay; ++ilay)
            {
                const int idx_lay  = icol + ilay*ncol + igpt*nlay*ncol;
                const int idx_lev1 = icol + ilay*ncol + igpt*(nlay+1)*ncol;
                const int idx_lev2 = icol + (ilay+1)*ncol + igpt*(nlay+1)*ncol;
                source_up[idx_lay] = r_dir[idx_lay] * flux_dir[idx_lev1];
                source_dn[idx_lay] = t_dir[idx_lay] * flux_dir[idx_lev1];
                flux_dir[idx_lev2] = t_noscat[idx_lay] * flux_dir[idx_lev1];

            }
            const int sfc_idx = icol + igpt*ncol;
            const int flx_idx = icol + nlay*ncol + igpt*(nlay+1)*ncol;
            source_sfc[sfc_idx] = flux_dir[flx_idx] * sfc_alb_dir[icol];
        }
        else
        {
            for (int ilay=nlay-1; ilay>=0; --ilay)
            {
                const int idx_lay  = icol + ilay*ncol + igpt*nlay*ncol;
                const int idx_lev1 = icol + (ilay)*ncol + igpt*(nlay+1)*ncol;
                const int idx_lev2 = icol + (ilay+1)*ncol + igpt*(nlay+1)*ncol;
                source_up[idx_lay] = r_dir[idx_lay] * flux_dir[idx_lev2];
                source_dn[idx_lay] = t_dir[idx_lay] * flux_dir[idx_lev2];
                flux_dir[idx_lev1] = t_noscat[idx_lay] * flux_dir[idx_lev2];

            }
            const int sfc_idx = icol + igpt*ncol;
            const int flx_idx = icol + igpt*(nlay+1)*ncol;
            source_sfc[sfc_idx] = flux_dir[flx_idx] * sfc_alb_dir[icol];
        }

    }

    template<typename TF>__device__
    void apply_BC_kernel_lw(const int icol, const int igpt, const int isfc, int ncol, const int nlay, const int ngpt, const BOOL_TYPE top_at_1, const TF* __restrict__ inc_flux, TF* __restrict__ flux_dn)
    {
        const int idx_in  = icol + isfc*ncol + igpt*ncol*(nlay+1);
        const int idx_out = (top_at_1) ? icol + igpt*ncol*(nlay+1) : icol + nlay*ncol + igpt*ncol*(nlay+1);
        flux_dn[idx_out] = inc_flux[idx_in];
    }

    template<typename TF>__global__ //apply_BC_gpt
    void apply_BC_kernel(const int ncol, const int nlay, const int ngpt, const BOOL_TYPE top_at_1, const TF* __restrict__ inc_flux, TF* __restrict__ flux_dn)
    {
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;
        const int igpt = blockIdx.y*blockDim.y + threadIdx.y;
        if ( (icol < ncol) && (igpt < ngpt) )
        {
            if (top_at_1)
            {
                const int idx_out = icol + igpt*ncol*(nlay+1);
                const int idx_in  = icol + igpt*ncol;
                flux_dn[idx_out] = inc_flux[idx_in];
            }
            else
            {
                const int idx_out = icol + nlay*ncol + igpt*ncol*(nlay+1);
                const int idx_in  = icol + igpt*ncol;
                flux_dn[idx_out] = inc_flux[idx_in];
            }
        }
    }

    template<typename TF>__global__ //apply_BC_factor
    void apply_BC_kernel(const int ncol, const int nlay, const int ngpt, const BOOL_TYPE top_at_1, const TF* __restrict__ inc_flux, const TF* __restrict__ factor, TF* __restrict__ flux_dn)
    {
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;
        const int igpt = blockIdx.y*blockDim.y + threadIdx.y;
        if ( (icol < ncol) && (igpt < ngpt) )
        {
            if (top_at_1)
            {
                const int idx_out = icol + igpt*ncol*(nlay+1);
                const int idx_in  = icol + igpt*ncol;
                flux_dn[idx_out] = inc_flux[idx_in] * factor[icol];
            }
            else
            {
                const int idx_out = icol + nlay*ncol + igpt*ncol*(nlay+1);
                const int idx_in  = icol + igpt*ncol;
                flux_dn[idx_out] = inc_flux[idx_in] * factor[icol];
            }
        }
    }

    template<typename TF>__global__ //apply_BC_0
    void apply_BC_kernel(const int ncol, const int nlay, const int ngpt, const BOOL_TYPE top_at_1, TF* __restrict__ flux_dn)
    {
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;
        const int igpt = blockIdx.y*blockDim.y + threadIdx.y;
        if ( (icol < ncol) && (igpt < ngpt) )
        {
            if (top_at_1)
            {
                const int idx_out = icol + igpt*ncol*(nlay+1);
                flux_dn[idx_out] = TF(0.);
            }
            else
            {
                const int idx_out = icol + nlay*ncol + igpt*ncol*(nlay+1);
                flux_dn[idx_out] = TF(0.);
            }
        }
    }

    template<typename TF>__global__
    void sw_2stream_kernel(const int ncol, const int nlay, const int ngpt, const TF tmin,
            const TF* __restrict__ tau, const TF* __restrict__ ssa, const TF* __restrict__ g, const TF* __restrict__ mu0,
            TF* __restrict__ r_dif, TF* __restrict__ t_dif,
            TF* __restrict__ r_dir, TF* __restrict__ t_dir, TF* __restrict__ t_noscat)
    {
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;
        const int ilay = blockIdx.y*blockDim.y + threadIdx.y;
        const int igpt = blockIdx.z*blockDim.z + threadIdx.z;

        if ( (icol < ncol) && (ilay < nlay) && (igpt < ngpt) )
        {
            const int idx = icol + ilay*ncol + igpt*nlay*ncol;
            const TF mu0_inv = TF(1.)/mu0[icol];
            const TF gamma1 = (TF(8.) - ssa[idx] * (TF(5.) + TF(3.) * g[idx])) * TF(.25);
            const TF gamma2 = TF(3.) * (ssa[idx] * (TF(1.) -          g[idx])) * TF(.25);
            const TF gamma3 = (TF(2.) - TF(3.) * mu0[icol] *          g[idx]) * TF(.25);
            const TF gamma4 = TF(1.) - gamma3;

            const TF alpha1 = gamma1 * gamma4 + gamma2 * gamma3;
            const TF alpha2 = gamma1 * gamma3 + gamma2 * gamma4;

            const TF k = sqrt(max((gamma1 - gamma2) * (gamma1 + gamma2), TF(1e-12)));
            const TF exp_minusktau = exp(-tau[idx] * k);
            const TF exp_minus2ktau = exp_minusktau * exp_minusktau;

            const TF rt_term = TF(1.) / (k      * (TF(1.) + exp_minus2ktau) +
                                         gamma1 * (TF(1.) - exp_minus2ktau));
            r_dif[idx] = rt_term * gamma2 * (TF(1.) - exp_minus2ktau);
            t_dif[idx] = rt_term * TF(2.) * k * exp_minusktau;
            t_noscat[idx] = exp(-tau[idx] * mu0_inv);

            const TF k_mu     = k * mu0[icol];
            const TF k_gamma3 = k * gamma3;
            const TF k_gamma4 = k * gamma4;

            const TF fact = (abs(TF(1.) - k_mu*k_mu) > tmin) ? TF(1.) - k_mu*k_mu : tmin;
            const TF rt_term2 = ssa[idx] * rt_term / fact;

            r_dir[idx] = rt_term2  * ((TF(1.) - k_mu) * (alpha2 + k_gamma3)   -
                                      (TF(1.) + k_mu) * (alpha2 - k_gamma3) * exp_minus2ktau -
                                      TF(2.) * (k_gamma3 - alpha2 * k_mu)  * exp_minusktau * t_noscat[idx]);

            t_dir[idx] = -rt_term2 * ((TF(1.) + k_mu) * (alpha1 + k_gamma4) * t_noscat[idx]   -
                                      (TF(1.) - k_mu) * (alpha2 - k_gamma4) * exp_minus2ktau * t_noscat[idx] -
                                       TF(2.) * (k_gamma4 + alpha1 * k_mu) * exp_minusktau);
        }
    }

    template<typename TF>__global__
    void sw_source_adding_kernel(const int ncol, const int nlay, const int ngpt, const BOOL_TYPE top_at_1,
                                 const TF* __restrict__ sfc_alb_dir, const TF* __restrict__ sfc_alb_dif,
                                 TF* __restrict__ r_dif, TF* __restrict__ t_dif,
                                 TF* __restrict__ r_dir, TF* __restrict__ t_dir, TF* __restrict__ t_noscat,
                                 TF* __restrict__ flux_up, TF* __restrict__ flux_dn, TF* __restrict__ flux_dir,
                                 TF* __restrict__ source_up, TF* __restrict__ source_dn, TF* __restrict__ source_sfc,
                                 TF* __restrict__ albedo, TF* __restrict__ src, TF* __restrict__ denom)
    {
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;
        const int igpt = blockIdx.y*blockDim.y + threadIdx.y;

        if ( (icol < ncol) && (igpt < ngpt) )
        {
            sw_source_kernel(icol, igpt, ncol, nlay, top_at_1, r_dir, t_dir,
                             t_noscat, sfc_alb_dir, source_up, source_dn, source_sfc, flux_dir);

            sw_adding_kernel(icol, igpt, ncol, nlay, top_at_1, sfc_alb_dif,
                             r_dif, t_dif, source_dn, source_up, source_sfc,
                             flux_up, flux_dn, flux_dir, albedo, src, denom);
        }
    }

    template<typename TF>__global__
    void lw_solver_noscat_gaussquad_kernel(const int ncol, const int nlay, const int ngpt, const TF eps,
                                           const BOOL_TYPE top_at_1, const int nmus, const TF* __restrict__ ds, const TF* __restrict__ weights,
                                           const TF* __restrict__ tau, const TF* __restrict__ lay_source,
                                           const TF* __restrict__ lev_source_inc, const TF* __restrict__ lev_source_dec, const TF* __restrict__ sfc_emis,
                                           const TF* __restrict__ sfc_src, TF* __restrict__ radn_up, TF* __restrict__ radn_dn,
                                           const TF* __restrict__ sfc_src_jac, TF* __restrict__ radn_up_jac, TF* __restrict__ tau_loc,
                                           TF* __restrict__ trans, TF* __restrict__ source_dn, TF* __restrict__ source_up,
                                           TF* __restrict__ source_sfc, TF* __restrict__ sfc_albedo, TF* __restrict__ source_sfc_jac,
                                           TF* __restrict__ flux_up, TF* __restrict__ flux_dn, TF* __restrict__ flux_up_jac)
    {
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;
        const int igpt = blockIdx.y*blockDim.y + threadIdx.y;

        if ( (icol < ncol) && (igpt < ngpt) )
        {
            lw_solver_noscat_kernel(icol, igpt, ncol, nlay, ngpt, eps, top_at_1, ds[0], weights[0], tau, lay_source,
                             lev_source_inc, lev_source_dec, sfc_emis, sfc_src, flux_up, flux_dn, sfc_src_jac,
                             flux_up_jac, tau_loc, trans, source_dn, source_up, source_sfc, sfc_albedo, source_sfc_jac);
            const int top_level = top_at_1 ? 0 : nlay;
            apply_BC_kernel_lw(icol, igpt, top_level, ncol, nlay, ngpt, top_at_1, flux_dn, radn_dn);

            if (nmus > 1)
            {
                for (int imu=1; imu<nmus; ++imu)
                {
                    lw_solver_noscat_kernel(icol, igpt, ncol, nlay, ngpt, eps, top_at_1, ds[imu], weights[imu], tau, lay_source,
                                     lev_source_inc, lev_source_dec, sfc_emis, sfc_src, radn_up, radn_dn, sfc_src_jac,
                                     radn_up_jac, tau_loc, trans, source_dn, source_up, source_sfc, sfc_albedo, source_sfc_jac);

                    for (int ilev=0; ilev<(nlay+1); ++ilev)
                    {
                        const int idx = icol + ilev*ncol + igpt*ncol*(nlay+1);
                        flux_up[idx] += radn_up[idx];
                        flux_dn[idx] += radn_dn[idx];
                        flux_up_jac[idx] += radn_up_jac[idx];
                    }
                }
            }
        }
    }
}

namespace rte_kernel_launcher_cuda
{
    template<typename TF>
    void apply_BC(const int ncol, const int nlay, const int ngpt, const BOOL_TYPE top_at_1,
                  const Array_gpu<TF,2>& inc_flux_dir, const Array_gpu<TF,1>& mu0, Array_gpu<TF,3>& gpt_flux_dir)
    {
        const int block_col = 32;
        const int block_gpt = 32;

        const int grid_col  = ncol/block_col + (ncol%block_col > 0);
        const int grid_gpt  = ngpt/block_gpt + (ngpt%block_gpt > 0);

        dim3 grid_gpu(grid_col, grid_gpt);
        dim3 block_gpu(block_col, block_gpt);
        apply_BC_kernel<<<grid_gpu, block_gpu>>>(ncol, nlay, ngpt, top_at_1, inc_flux_dir.ptr(), mu0.ptr(), gpt_flux_dir.ptr());

    }
    template<typename TF>
    void apply_BC(const int ncol, const int nlay, const int ngpt, const BOOL_TYPE top_at_1, Array_gpu<TF,3>& gpt_flux_dn)
    {
        const int block_col = 32;
        const int block_gpt = 32;

        const int grid_col  = ncol/block_col + (ncol%block_col > 0);
        const int grid_gpt  = ngpt/block_gpt + (ngpt%block_gpt > 0);

        dim3 grid_gpu(grid_col, grid_gpt);
        dim3 block_gpu(block_col, block_gpt);
        apply_BC_kernel<<<grid_gpu, block_gpu>>>(ncol, nlay, ngpt, top_at_1, gpt_flux_dn.ptr());
    }
    template<typename TF>
    void apply_BC(const int ncol, const int nlay, const int ngpt, const BOOL_TYPE top_at_1, const Array_gpu<TF,2>& inc_flux_dif, Array_gpu<TF,3>& gpt_flux_dn)
    {
        const int block_col = 32;
        const int block_gpt = 32;

        const int grid_col  = ncol/block_col + (ncol%block_col > 0);
        const int grid_gpt  = ngpt/block_gpt + (ngpt%block_gpt > 0);

        dim3 grid_gpu(grid_col, grid_gpt);
        dim3 block_gpu(block_col, block_gpt);
        apply_BC_kernel<<<grid_gpu, block_gpu>>>(ncol, nlay, ngpt, top_at_1, inc_flux_dif.ptr(), gpt_flux_dn.ptr());
    }

    template<typename TF>
    void lw_solver_noscat_gaussquad(const int ncol, const int nlay, const int ngpt, const BOOL_TYPE top_at_1, const int nmus,
                                    const Array_gpu<TF,2>& ds, const Array_gpu<TF,2>& weights, const Array_gpu<TF,3>& tau, const Array_gpu<TF,3> lay_source,
                                    const Array_gpu<TF,3>& lev_source_inc, const Array_gpu<TF,3>& lev_source_dec, const Array_gpu<TF,2>& sfc_emis,
                                    const Array_gpu<TF,2>& sfc_src, Array_gpu<TF,3>& flux_up, Array_gpu<TF,3>& flux_dn,
                                    const Array_gpu<TF,2>& sfc_src_jac, Array_gpu<TF,3>& flux_up_jac)
    {
        float elapsedtime;
        TF eps = std::numeric_limits<TF>::epsilon();
        const int flx_size = flux_dn.size() * sizeof(TF);
        const int opt_size = tau.size() * sizeof(TF);
        const int mus_size = nmus * sizeof(TF);
        const int sfc_size = sfc_src.size() * sizeof(TF);

        TF* tau_loc;
        TF* radn_up;
        TF* radn_up_jac;
        TF* radn_dn;
        TF* trans;
        TF* source_dn;
        TF* source_up;
        TF* source_sfc;
        TF* source_sfc_jac;
        TF* sfc_albedo;

        cuda_safe_call(cudaMalloc((void **) &source_sfc, sfc_size));
        cuda_safe_call(cudaMalloc((void **) &source_sfc_jac, sfc_size));
        cuda_safe_call(cudaMalloc((void **) &sfc_albedo, sfc_size));
        cuda_safe_call(cudaMalloc((void **) &tau_loc, opt_size));
        cuda_safe_call(cudaMalloc((void **) &trans, opt_size));
        cuda_safe_call(cudaMalloc((void **) &source_dn, opt_size));
        cuda_safe_call(cudaMalloc((void **) &source_up, opt_size));
        cuda_safe_call(cudaMalloc((void **) &radn_dn, flx_size));
        cuda_safe_call(cudaMalloc((void **) &radn_up, flx_size));
        cuda_safe_call(cudaMalloc((void **) &radn_up_jac, flx_size));

        const int block_col2d = 32;
        const int block_gpt2d = 32;

        const int grid_col2d  = ncol/block_col2d + (ncol%block_col2d > 0);
        const int grid_gpt2d  = ngpt/block_gpt2d + (ngpt%block_gpt2d > 0);

        dim3 grid_gpu2d(grid_col2d, grid_gpt2d);
        dim3 block_gpu2d(block_col2d, block_gpt2d);
        lw_solver_noscat_gaussquad_kernel<<<grid_gpu2d, block_gpu2d>>>(
                ncol, nlay, ngpt, eps, top_at_1, nmus, ds.ptr(), weights.ptr(), tau.ptr(), lay_source.ptr(),
                lev_source_inc.ptr(), lev_source_dec.ptr(), sfc_emis.ptr(), sfc_src.ptr(), radn_up,
                radn_dn, sfc_src_jac.ptr(), radn_up_jac, tau_loc, trans, source_dn, source_up,
                source_sfc, sfc_albedo, source_sfc_jac, flux_up.ptr(), flux_dn.ptr(), flux_up_jac.ptr());

        cuda_safe_call(cudaFree(tau_loc));
        cuda_safe_call(cudaFree(radn_up));
        cuda_safe_call(cudaFree(radn_up_jac));
        cuda_safe_call(cudaFree(radn_dn));
        cuda_safe_call(cudaFree(trans));
        cuda_safe_call(cudaFree(source_dn));
        cuda_safe_call(cudaFree(source_up));
        cuda_safe_call(cudaFree(source_sfc));
        cuda_safe_call(cudaFree(source_sfc_jac));
        cuda_safe_call(cudaFree(sfc_albedo));
    }

    template<typename TF>
    void sw_solver_2stream(const int ncol, const int nlay, const int ngpt, const BOOL_TYPE top_at_1,
                           const Array_gpu<TF,3>& tau, const Array_gpu<TF,3>& ssa, const Array_gpu<TF,3>& g,
                           const Array_gpu<TF,1>& mu0, const Array_gpu<TF,2>& sfc_alb_dir, const Array_gpu<TF,2>& sfc_alb_dif,
                           Array_gpu<TF,3>& flux_up, Array_gpu<TF,3>& flux_dn, Array_gpu<TF,3>& flux_dir)
    {
        const int opt_size = tau.size() * sizeof(TF);
        const int alb_size  = sfc_alb_dir.size() * sizeof(TF);
        const int flx_size  = flux_up.size() * sizeof(TF);
        TF* r_dif;
        TF* t_dif;
        TF* r_dir;
        TF* t_dir;
        TF* t_noscat;
        TF* source_up;
        TF* source_dn;
        TF* source_sfc;
        TF* albedo;
        TF* src;
        TF* denom;

        cuda_safe_call(cudaMalloc((void **) &r_dif, opt_size));
        cuda_safe_call(cudaMalloc((void **) &t_dif, opt_size));
        cuda_safe_call(cudaMalloc((void **) &r_dir, opt_size));
        cuda_safe_call(cudaMalloc((void **) &t_dir, opt_size));
        cuda_safe_call(cudaMalloc((void **) &t_noscat, opt_size));
        cuda_safe_call(cudaMalloc((void **) &source_up, opt_size));
        cuda_safe_call(cudaMalloc((void **) &source_dn, opt_size));
        cuda_safe_call(cudaMalloc((void **) &source_sfc, alb_size));
        cuda_safe_call(cudaMalloc((void **) &albedo, flx_size));
        cuda_safe_call(cudaMalloc((void **) &src, flx_size));
        cuda_safe_call(cudaMalloc((void **) &denom, opt_size));
        const int block_col3d = 32;
        const int block_lay3d = 16;
        const int block_gpt3d = 1;

        const int grid_col3d  = ncol/block_col3d + (ncol%block_col3d > 0);
        const int grid_lay3d  = nlay/block_lay3d + (nlay%block_lay3d > 0);
        const int grid_gpt3d  = ngpt/block_gpt3d + (ngpt%block_gpt3d > 0);

        dim3 grid_gpu3d(grid_col3d, grid_lay3d, grid_gpt3d);
        dim3 block_gpu3d(block_col3d, block_lay3d, block_gpt3d);

        TF tmin = std::numeric_limits<TF>::epsilon();
        sw_2stream_kernel<<<grid_gpu3d, block_gpu3d>>>(
                ncol, nlay, ngpt, tmin, tau.ptr(), ssa.ptr(), g.ptr(), mu0.ptr(), r_dif, t_dif, r_dir, t_dir, t_noscat);

//        cuda_safe_call(cudaDeviceSynchronize());
        const int block_col2d = 32;
        const int block_gpt2d = 32;

        const int grid_col2d  = ncol/block_col2d + (ncol%block_col2d > 0);
        const int grid_gpt2d  = ngpt/block_gpt2d + (ngpt%block_gpt2d > 0);

        dim3 grid_gpu2d(grid_col2d, grid_gpt2d);
        dim3 block_gpu2d(block_col2d, block_gpt2d);
        sw_source_adding_kernel<<<grid_gpu2d, block_gpu2d>>>(
                ncol, nlay, ngpt, top_at_1, sfc_alb_dir.ptr(), sfc_alb_dif.ptr(), r_dif, t_dif, r_dir, t_dir, t_noscat,
                flux_up.ptr(), flux_dn.ptr(), flux_dir.ptr(), source_up, source_dn, source_sfc, albedo, src, denom);

        cuda_safe_call(cudaFree(r_dif));
        cuda_safe_call(cudaFree(t_dif));
        cuda_safe_call(cudaFree(r_dir));
        cuda_safe_call(cudaFree(t_dir));
        cuda_safe_call(cudaFree(t_noscat));
        cuda_safe_call(cudaFree(source_up));
        cuda_safe_call(cudaFree(source_dn));
        cuda_safe_call(cudaFree(source_sfc));
        cuda_safe_call(cudaFree(albedo));
        cuda_safe_call(cudaFree(src));
        cuda_safe_call(cudaFree(denom));
    }
}

#ifdef FLOAT_SINGLE_RRTMGP
template void rte_kernel_launcher_cuda::sw_solver_2stream<float>(
            const int, const int, const int, const BOOL_TYPE,
            const Array_gpu<float,3>&, const Array_gpu<float,3>&, const Array_gpu<float,3>&,
            const Array_gpu<float,1>&, const Array_gpu<float,2>&, const Array_gpu<float,2>&,
            Array_gpu<float,3>&, Array_gpu<float,3>&, Array_gpu<float,3>&);

template void rte_kernel_launcher_cuda::lw_solver_noscat_gaussquad<float>(
            const int ncol, const int nlay, const int ngpt, const BOOL_TYPE top_at_1, const int nmus,
            const Array_gpu<float,2>& ds, const Array_gpu<float,2>& weights, const Array_gpu<float,3>& tau, const Array_gpu<float,3> lay_source,
            const Array_gpu<float,3>& lev_source_inc, const Array_gpu<float,3>& lev_source_dec, const Array_gpu<float,2>& sfc_emis,
            const Array_gpu<float,2>& sfc_src, Array_gpu<float,3>& flux_dn, Array_gpu<float,3>& flux_up,
            const Array_gpu<float,2>& sfc_src_jac, Array_gpu<float,3>& flux_up_jac);

#else
template void rte_kernel_launcher_cuda::apply_BC(const int, const int, const int, const BOOL_TYPE,
                  const Array_gpu<double,2>&, const Array_gpu<double,1>&, Array_gpu<double,3>&);
template void rte_kernel_launcher_cuda::apply_BC(const int, const int, const int, const BOOL_TYPE, Array_gpu<double,3>&);
template void rte_kernel_launcher_cuda::apply_BC(const int, const int, const int, const BOOL_TYPE,
                  const Array_gpu<double,2>&, Array_gpu<double,3>&);


template void rte_kernel_launcher_cuda::sw_solver_2stream<double>(
            const int, const int, const int, const BOOL_TYPE,
            const Array_gpu<double,3>&, const Array_gpu<double,3>&, const Array_gpu<double,3>&,
            const Array_gpu<double,1>&, const Array_gpu<double,2>&, const Array_gpu<double,2>&,
            Array_gpu<double,3>&, Array_gpu<double,3>&, Array_gpu<double,3>&);

template void rte_kernel_launcher_cuda::lw_solver_noscat_gaussquad<double>(
            const int ncol, const int nlay, const int ngpt, const BOOL_TYPE top_at_1, const int nmus,
            const Array_gpu<double,2>& ds, const Array_gpu<double,2>& weights, const Array_gpu<double,3>& tau, const Array_gpu<double,3> lay_source,
            const Array_gpu<double,3>& lev_source_inc, const Array_gpu<double,3>& lev_source_dec, const Array_gpu<double,2>& sfc_emis,
            const Array_gpu<double,2>& sfc_src, Array_gpu<double,3>& flux_up, Array_gpu<double,3>& flux_dn,
            const Array_gpu<double,2>& sfc_src_jac,Array_gpu<double,3>& flux_up_jac);
#endif
