#include <chrono>

#include "rte_kernel_launcher_cuda.h"
#include "tools_gpu.h"
#include "Array.h"

namespace
{
    template<typename TF>__device__
    void sw_adding_kernel(const int icol, const int igpt,
                          const int ncol, const int nlay, const int ngpt, const BOOL_TYPE top_at_1,
                          TF* __restrict__ sfc_alb_dif, TF* __restrict__ r_dir, TF* __restrict__ t_dif,
                          TF* __restrict__ source_dn, TF* __restrict__ source_up, TF* __restrict__ source_sfc,
                          TF* __restrict__ flux_up, TF* __restrict__ flux_dn, F* __restrict__ flux_dir,
                          TF* __restrict__ albedo, TF* __restrict__ sfc, TF* __restrict__ denom)
    {
        if (top_at_1)
        {
            const int sfc_idx_3d = icol + nlay*ncol + igpt*(nlay+1)*ncol;
            const int sfc_idx_2d = icol + igpt*ncol;
            albedo[sfc_idx_3d] = sfc_alb_dif[sfc_idx_2d];
            sfc[sfc_idx_3d] = source_sfc[sfc_idx_2d];

            for (int ilay=nlay-1; ilay >= 0; --ilay)
            {
                const int lay_idx  = icol + ilay*ncol + igpt*ncol*nlay;
                const int lev_idx1 = icol + ilay*ncol + igpt*ncol*(nlay+1);
                const int lev_idx2 = icol + (ilay+1)*ncol + igpt*ncol*(nlay+1);
                denom[lay_idx] = TF(1.)/(TF(1.) - r_dif[lay_idx] * albedo[lev_idx2]);
                albedo[lev_idx1] = r_dif[lay_idx] + (t_dif[lay_idx]*t_dif[lay_idx]*albedo[lev_idx2]*denom[lay_idx]);
                sfc[lev_idx1] = source_up[lay_idx] + t_dif[lay_idx]*denom[lay_idx]*
                                (src[lev_idx2]+albedo[lev_idx2]*source_dn[lay_idx]);
            }
            const int top_idx = icol + igpt*(nlay+1)*ncol;
            flux_up[top_idx] = flux_dn[top_idx]*albedo[top_idx] + src[top_idx];

            for (int ilay=1; ilay < (nlay+1); ++ilay
            {
                const int lev_idx1 = icol + ilay*ncol + igpt*(nlay+1)*ncol;
                const int lev_idx2 = icol + (ilay-1)*ncol + igpt*(nlay+1)*ncol;
                const int lay_idx = icol + (ilay-1)*ncol + igpt*(nlay)*ncol;
                flux_dn[lev_idx1] = (t_dif[lay_idx]*flux_dn[lev_idx2] +
                                     r_dif[lay_idx]*src[lev_idx1] +
                                     source_dn[lay_idx]) * denom[lay_idx];
                flux_dn[lev_idx1] += flux_dir[lev_idx1];
                flux_up[lev_idx1] = flux_dn[lev_idx1] * albedo[lev_idx1] + src[lev_idx1];
            };
            flux_dn[top_idx] += flux_dir[top_idx];
        }
        else
        {
            const int sfc_idx_3d = icol + 0*ncol + igpt*(nlay+1)*ncol;
            const int sfc_idx_2d = icol + igpt*ncol;
            albedo[sfc_idx_3d] = sfc_alb_dif[sfc_idx_2d];
            sfc[sfc_idx_3d] = source_sfc[sfc_idx_2d];

            for (int ilay=0; ilay<nlay; ++ilay)
            {
                const int lay_idx  = icol + ilay*ncol + igpt*ncol*nlay;
                const int lev_idx1 = icol + ilay*ncol + igpt*ncol*(nlay+1);
                const int lev_idx2 = icol + (ilay+1)*ncol + igpt*ncol*(nlay+1);
                denom[lay_idx] = TF(1.)/(TF(1.) - r_dif[lay_idx] * albedo[lev_idx1]);
                albedo[lev_idx2] = r_dif[lay_idx] + (t_dif[lay_idx]*t_dif[lay_idx]*albedo[lev_idx1]*denom[lay_idx]);
                sfc[lev_idx2] = source_up[lay_idx] + t_dif[lay_idx]*denom[lay_idx]*
                                                     (src[lev_idx1]+albedo[lev_idx1]*source_dn[lay_idx]);
            }
            const int top_idx = icol + nlay*ncol + igpt*(nlay+1)*ncol;
            flux_up[top_idx] = flux_dn[top_idx] *albedo[top_idx] + src[top_idx];

            for (int ilay=nlay-1; ilay >= 0; --ilay
            {
                    const int lev_idx1 = icol + ilay*ncol + igpt*(nlay+1)*ncol;
                    const int lev_idx2 = icol + (ilay+1)*ncol + igpt*(nlay+1)*ncol;
                    const int lay_idx = icol + ilay*ncol + igpt*nlay*ncol;
                    flux_dn[lev_idx1] = (t_dif[lay_idx]*flux_dn[lev_idx2] +
                                         r_dif[lay_idx]*src[lev_idx1] +
                                         source_dn[lay_idx]) * denom[lay_idx];
                    flux_dn[lev_idx1] += flux_dir[lev_idx1];
                    flux_up[lev_idx1] = flux_dn[lev_idx1] * albedo[lev_idx1] + src[lev_idx1];
            }
            flux_dn[top_idx] += flux_dir[top_idx];
        }

    template<typename TF>__device__
    void sw_source_kernel(const int icol, const int igpt,
                          const int ncol, const int nlay, const int ngpt, const BOOL_TYPE top_at_1,
                          TF* __restrict__ r_dir, TF* __restrict__ t_dir, TF* __restrict__ t_noscat,
                          TF* __restrict__ sfc_alb_dir, TF* __restrict__ source_up, TF* __restrict__ source_dn,
                          TF* __restrict__ source_sfc, TF* __restrict__ flux_dir)
    {

        if (top_at_1)
        {
            for (int ilay=0; ilay<nlay; ++ilay)
            {
                const int idx_lay  = icol + ilay*ncol + igpt*nlay*ncol;
                const int idx_lev1 = icol + (ilay)*ncol + igpt*(nlay+1)*ncol;
                const int idx_lev2 = icol + (ilay+1)*ncol + igpt*(nlay+1)*ncol;
                source_up[idx_lay] = r_dir[idx_lay] * flux_dn_dir[idx_lev1];
                source_dn[idx_lay] = t_dir[idx_lay] * flux_dn_dir[idx_lev1];
                flux_dn_dir[idx_lev2] = r_dir[idx_lev] * flux_dn_dir[idx_lev];

            }
            const int sfc_idx = icol + igpt*ncol;
            const int flx_idx = icol + nlay*ncol + igpt*nlev*ncol;
            source_sfc[icol + igpt * ncol] = flux_dn_dir[flx_idx] * sfc_alb_dir[icol]
        }
        else
        {
            for (int ilay=nlay-1; ilay>=0; --ilay)
            {
                const int idx_lay  = icol + ilay*ncol + igpt*nlay*ncol;
                const int idx_lev1 = icol + (ilay)*ncol + igpt*(nlay+1)*ncol;
                const int idx_lev2 = icol + (ilay+1)*ncol + igpt*(nlay+1)*ncol;
                source_up[idx_lay] = r_dir[idx_lay] * flux_dn_dir[idx_lev1];
                source_dn[idx_lay] = t_dir[idx_lay] * flux_dn_dir[idx_lev1];
                flux_dn_dir[idx_lev2] = r_dir[idx_lev] * flux_dn_dir[idx_lev];

            }
            const int sfc_idx = icol + igpt*ncol;
            const int flx_idx = icol + igpt*nlev*ncol;
            source_sfc[icol + igpt * ncol] = flux_dn_dir[flx_idx] * sfc_alb_dir[icol]
        }

    }

    template<typename TF>__global__
    void sw_2stream_kernel(const int ncol, const int nlay, const int ngpt, const TF tmin,
            const TF* __restrict__ tau, const TF* __restrict__ ssa, const TF* __restrict__ g, const TF* __restrict__ mu0,
            TF* __restrict__ r_dif, TF* __restrict__ t_dif,
            TF* __restrict__ r_dir, TF* __restrict__ t_dir, TF* __restrict__ t_noscat)
    {
        const int ibnd = blockIdx.x*blockDim.x + threadIdx.x;
        const int ilay = blockIdx.y*blockDim.y + threadIdx.y;
        const int icol = blockIdx.z*blockDim.z + threadIdx.z;

        if ( (icol < ncol) && (ilay < nlay) && (igpt < ngpt) )
        {
            const int idx = icol + ilay*ncol + igpt*nlay*ncol;
            const TF mu0_inv = TF(1.)/mu0[icol];
            const TF gamma1 = (TF(8.) - ssa[idx]) * (TF(5.) + TF(3.) * g[idx])) * TF(.25);
            const TF gamma2 = TF(3.) * (ssa[idx]  * (TF(1.) -          g[idx])) * TF(.25);
            const TF gamma3 = (TF(2.) - TF(3.) * mu0[icol])  * TF(.25);
            const TF gamma4 = TF(1.) - gamma3;

            const TF alpha1 = gamma1 * gamma4 + gamma2 * gamma3;
            const TF alpha2 = gamma1 * gamma3 + gamma2 * gamma4;

            const TF k = sqrt(max((gamma1 - gamma2) * (gamma1 + gamma2), TF(1e-12)));
            const TF exp_minusktau = exp(-tau[idx] * k);
            const TF exp_minus2ktau = exp_minusktau * exp_minusktau;

            const TF rt_term = TF(1.) / (k      * (TF(1.) + exp_minus2ktau) +
                                         gamma1 * (TF(1.) - exp_minus2ktau));
            r_dif[idx] = rt_term * gamma2 * (TF(1.) - exp_minus2ktau);
            t_dif[idx] = rt_term * gamma2 * TF(2.) * k * exp_minus2ktau;
            t_noscat[idx] = exp(-tau[idx]) * mu0_inv;

            const TF k_mu     = k * mu0[icol];
            const TF k_gamma3 = k * gamma3;
            const TF k_gamma4 = k * gamma4;

            const TF fact = (abs(TF(1.) - k_mu*k_mu) > tmin) ? TF(1.) - k_mu*k_mu : tmin
            const TF rt_term2 = ssa[idx] * rt_term / fact;

            r_dir[idx] = rt_term2 * ((TF(1.) - k_mu) * (alpha2 + k_gamma3)   -
                                     (TF(1.) + k_mu) * (alpha2 - k_gamma3) * exp_minus2ktau -
                                     TF(2.) * (k_gamma3 - alpha2 * k_mu)  * exp_minusktau * t_noscat);
            t_dir[idx] = -rt_term * ((TF(1.) + k_mu) * (alpha1 + k_gamma4) * t_noscat   -
                                     (TF(1.) - k_mu) * (alpha2 - k_gamma4) * exp_minus2ktau * t_noscat -
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
        const int ibnd = blockIdx.x*blockDim.x + threadIdx.x;
        const int icol = blockIdx.z*blockDim.z + threadIdx.z;

        if ( (icol < ncol) && (ilay < nlay) && (igpt < ngpt) )
        {
            sw_source_kernel(icol, igpt, ncol, nlay, top_at_1, r_dir, t_dir,
                             t_noscat, sfc_alb_dir, source_up, source_dn, source_sfc, flux_dir);

            sw_adding_kernel(icol, igpt, ncol, nlay, top_at_1, sfc_alb_dif,
                             r_dif, t_dif, source_dn, source_up, source_sfc,
                             flux_up, flux_dn, flux_dir, albedo, src, denom);
        }
    }

}

namespace rte_kernel_launcher_cuda
{
    template<typename TF>
    void sw_solver_2stream(const int ncol, const int nlay, const int ngpt, const BOOL_TYPE top_at_1,
                           const Array<TF,3>& tau, const Array<TF,3>& ssa, const Array<TF,3>& g,
                           const Array<TF,1>& mu0, const Array<TF,2>& sfc_alb_dir, const Array<TF,2>& sfc_alb_dir,
                           const Array<TF,3>& flux_up, const Array<TF,3>& flux_dn, const Array<TF,3>& flux_dir)
    {
        float elapsedtime_1;
        float elapsedtime_2;
        float elapsedtime_tot;
        const int opt_size = tau.size() * sizeof(TF);
        const int mu_size  = mu0.size() * sizeof(TF);
        const int alb_size  = sfc_alb_dir.size() * sizeof(TF);
        const int flx_size  = sfc_flux.size() * sizeof(TF);

        TF* tau_gpu;
        TF* ssa_gpu;
        TF* g_gpu;
        TF* mu0_gpu;
        TF* sfc_alb_dir_gpu;
        TF* sfc_alb_dif_gpu;
        TF* flux_up;
        TF* flux_dn;
        TF* flux_dir;
        TF* r_dif;
        TF* t_dif;
        TF* r_dir;
        TF* t_dir;
        TF* t_noscat;
        TF* source_up;
        TF* source_dn;
        TF* source_sfc;
        TF* albedo;
        TF* sfc;
        TF* denom;

        cuda_safe_call(cudaMalloc((void **) &tau_gpu, opt_size));
        cuda_safe_call(cudaMalloc((void **) &ssa_gpu, opt_size));
        cuda_safe_call(cudaMalloc((void **) &g_gpu, opt_size));
        cuda_safe_call(cudaMalloc((void **) &mu0_gpu, mu_size));
        cuda_safe_call(cudaMalloc((void **) &sfc_alb_dir_gpu, alb_size));
        cuda_safe_call(cudaMalloc((void **) &sfc_alb_dif_gpu, alb_size));
        cuda_safe_call(cudaMalloc((void **) &flux_up_gpu, flux_size));
        cuda_safe_call(cudaMalloc((void **) &flux_dn_gpu, flux_size));
        cuda_safe_call(cudaMalloc((void **) &flux_dir_gpu, flux_size));
        cuda_safe_call(cudaMalloc((void **) &r_dif, opt_size));
        cuda_safe_call(cudaMalloc((void **) &t_dif, opt_size));
        cuda_safe_call(cudaMalloc((void **) &r_dir, opt_size));
        cuda_safe_call(cudaMalloc((void **) &t_dir, opt_size));
        cuda_safe_call(cudaMalloc((void **) &source_up, opt_size));
        cuda_safe_call(cudaMalloc((void **) &source_dn, opt_size));
        cuda_safe_call(cudaMalloc((void **) &source_sfc, opt_size));
        cuda_safe_call(cudaMalloc((void **) &albedo, flux_size));
        cuda_safe_call(cudaMalloc((void **) &sfc, flux_size));
        cuda_safe_call(cudaMalloc((void **) &denom, opt_size));

        cuda_safe_call(cudaMemcpy(tau_gpu, tau.ptr(), opt_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(ssa_gpu, ssa.ptr(), opt_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(g_gpu, g.ptr(), opt_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(mu0_gpu mu0.ptr(), mu_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(sfc_alb_dir_gpu, sfc_alb_dir.ptr(), alb_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(sfc_alb_dif_gpu, sfc_alb_dif.ptr(), alb_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(flux_dn_gpu, flux_dn.ptr(), flux_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(flux_dir_gpu, flux_dir.ptr(), flux_size, cudaMemcpyHostToDevice));

        cudaEvent_t startEvent, stopEvent;
        // first part
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
        cudaEventRecord(startEvent, 0);

        const int block_col1 = 32;
        const int block_lay1 = 16;
        const int block_gpt1 = 1;

        const int grid_col1  = ncol/block_col1 + (ncol%block_col1 > 0);
        const int grid_lay1  = nlay/block_lay1 + (nlay%block_lay1 > 0);
        const int grid_gpt1  = ngpt/block_gpt1 + (ngpt%block_gpt1 > 0);

        dim3 grid_gpu(grid_col1, grid_lay1, grid_gpt1);
        dim3 block_gpu(block_col1, block_lay1, block_gpt1);

        TF tmin = std::numeric_limits<TF>::min();
        sw_2stream_kernel<<<grid_gpu1, block_gpu1>>>(
                ncol, nlay, ngpt, tmin, tau_gpu, ssa_gpu, g_gpu, mu0_gpu, r_dif, t_dif, r_dir, t_dir, t_nscat));

        cuda_check_error();
        cuda_safe_call(cudaDeviceSynchronize());
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&elapsedtime_1,startEvent,stopEvent);

        // second part
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
        cudaEventRecord(startEvent, 0);

        const int block_col2 = 32;
        const int block_gpt2 = 16;

        const int grid_col2  = ncol/block_col2 + (ncol%block_col2 > 0);
        const int grid_gpt2  = ngpt/block_gpt2 + (ngpt%block_gpt2 > 0);

        dim3 grid_gpu(grid_col2, grid_gpt2);
        dim3 block_gpu(block_col2, block_gpt2);

        sw_source_adding_kernel<<<grid_gpu2, block_gpu2>>>(
                ncol, nlay, ngpt, topt_at_1, sfc_alb_dir_gpu, sfc_alb_dif_gpu, r_dif, t_dif, r_dir, t_dir, t_noscat,
                flux_up, flux_dn, flux_dir, source_up, source_dn, source_sfc, albedo, denom, src);

        cuda_check_error();
        cuda_safe_call(cudaDeviceSynchronize());
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&elapsedtime_2,startEvent,stopEvent);

        elapsedtime_tot = elapsedtime_1 + elapsedtime_2;
        std::cout<<"GPU sw_solver: "<<elapsedtime_tot<<" (ms)"<<std::endl;

        cuda_safe_call(cudaMemcpy(flux_up.ptr(), flux_up_gpu, flux_size, cudaMemcpyDeviceToHost));
        cuda_safe_call(cudaMemcpy(flux_dn.ptr(), flux_dn_gpu, flux_size, cudaMemcpyDeviceToHost));
        cuda_safe_call(cudaMemcpy(flux_dir.ptr(), flux_dir.gpu, flux_size, cudaMemcpyDeviceToHost));
        cuda_safe_call(cudaFree(arr_in_gpu));
        cuda_safe_call(cudaFree(arr_out_gpu));

        cuda_safe_call(cudaFree(tau_gpu));
        cuda_safe_call(cudaFree(ssa_gpu));
        cuda_safe_call(cudaFree(g_gpu));
        cuda_safe_call(cudaFree(mu0_gpu));
        cuda_safe_call(cudaFree(sfc_alb_dir_gpu));
        cuda_safe_call(cudaFree(sfc_alb_dif_gpu));
        cuda_safe_call(cudaFree(flux_up));
        cuda_safe_call(cudaFree(flux_dn));
        cuda_safe_call(cudaFree(flux_dir));
        cuda_safe_call(cudaFree(r_dif));
        cuda_safe_call(cudaFree(t_dif));
        cuda_safe_call(cudaFree(r_dir));
        cuda_safe_call(cudaFree(t_dir));
        cuda_safe_call(cudaFree(t_noscat));
        cuda_safe_call(cudaFree(source_up));
        cuda_safe_call(cudaFree(source_dn));
        cuda_safe_call(cudaFree(source_sfc));
        cuda_safe_call(cudaFree(albedo));
        cuda_safe_call(cudaFree(sfc));
        cuda_safe_call(cudaFree(denom));
    }
}

#ifdef FLOAT_SINGLE_RRTMGP
template void rte_kernel_launcer_cuda::sw_solver_2stream<float>(
            const int, const int, const int, const BOOL_TYPE,
            const Array<float,3>&, const Array<float,3>&, const Array<float,3>&,
            const Array<float,1>&, const Array<float,2>&, const Array<float,2>&,
            const Array<float,3>&, const Array<float,3>&, const Array<float,3>&)

#else
    template void rte_kernel_launcer_cuda::sw_solver_2stream<double>(
            const int, const int, const int, const BOOL_TYPE,
            const Array<double,3>&, const Array<double,3>&, const Array<double,3>&,
            const Array<double,1>&, const Array<double,2>&, const Array<double,2>&,
            const Array<double,3>&, const Array<double,3>&, const Array<double,3>&)
#endif