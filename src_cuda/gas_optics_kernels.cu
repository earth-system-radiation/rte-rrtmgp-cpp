#include <chrono>

#include "rrtmgp_kernel_launcher_cuda.h"
#include "tools_gpu.h"
#include "Array.h"

namespace
{
    // Add the kernels here.
    template<typename TF>__device__
    void interpolate2D_byflav_kernel(const TF* __restrict__ fminor,
                                     const TF* __restrict__ kin,
                                     const int gpt_start, const int gpt_end,
                                     TF* __restrict__ k,
                                     const int* __restrict__ jeta,
                                     const int jtemp,
                                     const int ngpt,
                                     const int neta)
    {
        const int band_gpt = gpt_end-gpt_start;
        const int j0 = jeta[0];
        const int j1 = jeta[1];
        for (int igpt=0; igpt<band_gpt; ++igpt)
        {
            k[igpt] = fminor[0] * kin[igpt + (j0-1)*ngpt + (jtemp-1)*neta*ngpt] +
                      fminor[1] * kin[igpt +  j0   *ngpt + (jtemp-1)*neta*ngpt] +
                      fminor[2] * kin[igpt + (j1-1)*ngpt + jtemp    *neta*ngpt] +
                      fminor[3] * kin[igpt +  j1   *ngpt + jtemp    *neta*ngpt];
        }
    }

    template<typename TF>__device__
    void interpolate3D_byflav_kernel(const TF* __restrict__ scaling,
                                     const TF* __restrict__ fmajor,
                                     const TF* __restrict__ k,
                                     const int gpt_start, const int gpt_end,
                                     const int* __restrict__ jeta,
                                     const int jtemp,
                                     const int jpress,
                                     const int ngpt,
                                     const int neta,
                                     const int npress,
                                     TF* __restrict__ tau_major)
    {
        const int band_gpt = gpt_end-gpt_start;
        const int j0 = jeta[0];
        const int j1 = jeta[1];
        for (int igpt=0; igpt<band_gpt; ++igpt)
        {
            tau_major[igpt] = scaling[0]*
                              (fmajor[0] * k[igpt + (j0-1)*ngpt + (jpress-1)*neta*ngpt + (jtemp-1)*neta*ngpt*npress] +
                               fmajor[1] * k[igpt +  j0   *ngpt + (jpress-1)*neta*ngpt + (jtemp-1)*neta*ngpt*npress] +
                               fmajor[2] * k[igpt + (j0-1)*ngpt + jpress*neta*ngpt     + (jtemp-1)*neta*ngpt*npress] +
                               fmajor[3] * k[igpt +  j0   *ngpt + jpress*neta*ngpt     + (jtemp-1)*neta*ngpt*npress])
                            + scaling[1]*
                              (fmajor[4] * k[igpt + (j1-1)*ngpt + (jpress-1)*neta*ngpt + jtemp*neta*ngpt*npress] +
                               fmajor[5] * k[igpt +  j1   *ngpt + (jpress-1)*neta*ngpt + jtemp*neta*ngpt*npress] +
                               fmajor[6] * k[igpt + (j1-1)*ngpt + jpress*neta*ngpt     + jtemp*neta*ngpt*npress] +
                               fmajor[7] * k[igpt +  j1   *ngpt + jpress*neta*ngpt     + jtemp*neta*ngpt*npress]);
        }
    }

    template<typename TF>__device__
    int locate_val(const TF* __restrict__ arr,
                   const int ncol,
                   const int nlay,
                   const BOOL_TYPE maxmin, //False: find minimum
                   const BOOL_TYPE* __restrict__ mask,
                   const BOOL_TYPE maskval)
    {
        TF temp = arr[0];
        for (int i=0; i<nlay; ++i)
        {
            const int ii = i*ncol;
            if (mask[ii]==maskval)
            {
                temp = arr[ii];
                break;
            }
        }
        int idx = 0;
        for (int i=0; i<nlay; ++i)
        {
            const int ii = i*ncol;
            if ((arr[ii]>temp) == maxmin and mask[ii]==maskval)
            {
                idx = i;
                temp = arr[ii];
            }
        }
        return idx;
    }

    template<typename TF>__global__
    void interpolation_kernel(
            const int ncol, const int nlay, const int ngas, const int nflav,
            const int neta, const int npres, const int ntemp, const TF tmin,
            const int* __restrict__ flavor,
            const TF* __restrict__ press_ref_log,
            const TF* __restrict__ temp_ref,
            TF press_ref_log_delta,
            TF temp_ref_min,
            TF temp_ref_delta,
            TF press_ref_trop_log,
            const TF* __restrict__ vmr_ref,
            const TF* __restrict__ play,
            const TF* __restrict__ tlay,
            TF* __restrict__ col_gas,
            int* __restrict__ jtemp,
            TF* __restrict__ fmajor, TF* __restrict__ fminor,
            TF* __restrict__ col_mix,
            BOOL_TYPE* __restrict__ tropo,
            int* __restrict__ jeta,
            int* __restrict__ jpress)
    {
        const int ilay = blockIdx.x*blockDim.x + threadIdx.x;
        const int icol = blockIdx.y*blockDim.y + threadIdx.y;

        if ( (icol < ncol) && (ilay < nlay) )
        {
            const int idx = icol + ilay*ncol;

            jtemp[idx] = int((tlay[idx] - (temp_ref_min-temp_ref_delta)) / temp_ref_delta);
            jtemp[idx] = min(ntemp-1, max(1, jtemp[idx]));
            const TF ftemp = (tlay[idx] - temp_ref[jtemp[idx]-1]) / temp_ref_delta;

            const TF locpress = TF(1.) + (log(play[idx]) - press_ref_log[0]) / press_ref_log_delta;
            jpress[idx] = min(npres-1, max(1, int(locpress)));
            const TF fpress = locpress - TF(jpress[idx]);

            tropo[idx] = log(play[idx]) > press_ref_trop_log;
            const int itropo = !tropo[idx];

            for (int iflav=0; iflav<nflav; ++iflav)
            {
                const int gas1 = flavor[2*iflav];
                const int gas2 = flavor[2*iflav+1];
                for (int itemp=0; itemp<2; ++itemp)
                {
                    const int vmr_base_idx = itropo + (jtemp[idx]+itemp-1) * (ngas+1) * 2;
                    const int colmix_idx = itemp + 2*(iflav + nflav*icol + nflav*ncol*ilay);
                    const int colgas1_idx = icol + ilay*ncol + gas1*nlay*ncol;
                    const int colgas2_idx = icol + ilay*ncol + gas2*nlay*ncol;
                    TF eta;
                    const TF ratio_eta_half = vmr_ref[vmr_base_idx + 2 * gas1] /
                                              vmr_ref[vmr_base_idx + 2 * gas2];
                    col_mix[colmix_idx] = col_gas[colgas1_idx] + ratio_eta_half * col_gas[colgas2_idx];
                    if (col_mix[colmix_idx] > TF(2.)*tmin)
                    {
                        eta = col_gas[colgas1_idx] / col_mix[colmix_idx];
                    } else
                    {
                        eta = TF(0.5);
                    }
                    const TF loceta = eta * TF(neta-1);
                    jeta[colmix_idx] = min(int(loceta)+1, neta-1);
                    const TF feta = fmod(loceta, TF(1.));
                    const TF ftemp_term  = TF(1-itemp) + TF(2*itemp-1)*ftemp;
                    // compute interpolation fractions needed for minot species
                    const int fminor_idx = 2*(itemp + 2*(iflav + icol*nflav + ilay*ncol*nflav));
                    fminor[fminor_idx] = (TF(1.0)-feta) * ftemp_term;
                    fminor[fminor_idx+1] = feta * ftemp_term;
                    // compute interpolation fractions needed for major species
                    const int fmajor_idx = 2*2*(itemp + 2*(iflav + icol*nflav + ilay*ncol*nflav));
                    fmajor[fmajor_idx] = (TF(1.0)-fpress) * fminor[fminor_idx];
                    fmajor[fmajor_idx+1] = (TF(1.0)-fpress) * fminor[fminor_idx+1];
                    fmajor[fmajor_idx+2] = fpress * fminor[fminor_idx];
                    fmajor[fmajor_idx+3] = fpress * fminor[fminor_idx+1];

                }
            }
        }
    }

    template<typename TF>__global__
    void compute_tau_major_absorption_kernel(
            const int ncol, const int nlay, const int nband, const int ngpt,
            const int nflav, const int neta, const int npres, const int ntemp,
            const int* __restrict__ gpoint_flavor,
            const int* __restrict__ band_lims_gpt,
            const TF* __restrict__ kmajor,
            const TF* __restrict__ col_mix, const TF* __restrict__ fmajor,
            const int* __restrict__ jeta, const BOOL_TYPE* __restrict__ tropo,
            const int* __restrict__ jtemp, const int* __restrict__ jpress,
            TF* __restrict__ tau, TF* __restrict__ tau_major)
    {
        // Fetch the three coordinates.
        const int ibnd = blockIdx.x*blockDim.x + threadIdx.x;
        const int ilay = blockIdx.y*blockDim.y + threadIdx.y;
        const int icol = blockIdx.z*blockDim.z + threadIdx.z;

        if ( (icol < ncol) && (ilay < nlay) && (ibnd < nband) ) {
            const int idx_collay = icol + ilay * ncol;
            const int itropo = !tropo[idx_collay];
            const int gpt_start = band_lims_gpt[2 * ibnd] - 1;
            const int gpt_end = band_lims_gpt[2 * ibnd + 1];
            const int iflav = gpoint_flavor[itropo + 2 * gpt_start] - 1;
            const int idx_fcl3 = 2 * 2 * 2* (iflav + icol * nflav + ilay * ncol * nflav);
            const int idx_fcl1 = 2 * (iflav + icol * nflav + ilay * ncol * nflav);
            const int idx_tau = gpt_start + ilay * ngpt + icol * nlay * ngpt;

            //major gases//
            interpolate3D_byflav_kernel(&col_mix[idx_fcl1], &fmajor[idx_fcl3],
                                        &kmajor[gpt_start], gpt_start, gpt_end,
                                        &jeta[idx_fcl1], jtemp[idx_collay], 
                                        jpress[idx_collay]+itropo, ngpt, neta, npres+1,
                                        &tau_major[idx_tau]);
            
            for (int igpt=gpt_start; igpt<gpt_end; ++igpt)
            {
                const int idx_out = igpt + ilay*ngpt + icol*nlay*ngpt;
                tau[idx_out] = tau_major[idx_out];
                //should be '+=' later on, but we first need the zero_arrays for that
            }
        }
    }

    template<typename TF>__global__
    void compute_tau_minor_absorption_kernel(
            const int ncol, const int nlay, const int ngpt,
            const int ngas, const int nflav, const int ntemp, const int neta,
            const int nscale_lower,
            const int nscale_upper,
            const int nminor_lower,
            const int nminor_upper,
            const int nminork_lower,
            const int nminork_upper,
            const int idx_h2o,
            const int* __restrict__ gpoint_flavor,
            const TF* __restrict__ kminor_lower,
            const TF* __restrict__ kminor_upper,
            const int* __restrict__ minor_limits_gpt_lower,
            const int* __restrict__ minor_limits_gpt_upper,
            const BOOL_TYPE* __restrict__ minor_scales_with_density_lower,
            const BOOL_TYPE* __restrict__ minor_scales_with_density_upper,
            const BOOL_TYPE* __restrict__ scale_by_complement_lower,
            const BOOL_TYPE* __restrict__ scale_by_complement_upper,
            const int* __restrict__ idx_minor_lower,
            const int* __restrict__ idx_minor_upper,
            const int* __restrict__ idx_minor_scaling_lower,
            const int* __restrict__ idx_minor_scaling_upper,
            const int* __restrict__ kminor_start_lower,
            const int* __restrict__ kminor_start_upper,
            const TF* __restrict__ play,
            const TF* __restrict__ tlay,
            const TF* __restrict__ col_gas,
            const TF* __restrict__ fminor,
            const int* __restrict__ jeta,
            const int* __restrict__ jtemp,
            const BOOL_TYPE* __restrict__ tropo,
            TF* __restrict__ tau,
            TF* __restrict__ tau_minor) 
    {
        // Fetch the three coordinates.
        const int ilay = blockIdx.x * blockDim.x + threadIdx.x;
        const int icol = blockIdx.y * blockDim.y + threadIdx.y;
        const TF PaTohPa = 0.01;
        const int ncl = ncol * nlay;
        if ((icol < ncol) && (ilay < nlay)) 
        {
            //kernel implementation
            const int idx_collay = icol + ilay * ncol;
            const int idx_collaywv = icol + ilay * ncol + idx_h2o * ncl;

            if (tropo[idx_collay] == 1) 
            {
                for (int imnr = 0; imnr < nscale_lower; ++imnr)
                {
                    TF scaling = col_gas[idx_collay + idx_minor[imnr] * ncl];
                    if (minor_scales_with_density_lower[imnr])
                    {
                        scaling *= PaTohPa * play[idx_collay] / tlay[idx_collay];
                        if (idx_minor_scaling_lower[imnr] > 0)
                        {
                            TF vmr_fact = TF(1.) / col_gas[idx_collay];
                            TF dry_fact = TF(1.) / (TF(1.) + col_gas[idx_collaywv] * vmr_fact);
                            if (scale_by_complement_lower[imnr])
                            {
                                scaling *= (TF(1.) - col_gas[idx_collay + idx_minor_scaling_lower[imnr] * ncl] * vmr_fact * dry_fact);
                            } 
                            else
                            {
                                scaling *= col_gas[idx_collay + idx_minor_scaling_lower[imnr] * ncl] * vmr_fact * dry_fact;
                            }
                        }
                    }
                    const int gpt_start = minor_limits_gpt_lower[2*imnr]-1;
                    const int gpt_end = minor_limits_gpt_lower[2*imnr+1];
                    const int iflav = gpoint_flavor[2*gpt_start]-1;
                    const int idx_fcl2 = 2 * 2 * (iflav + icol * nflav + ilay * ncol * nflav);
                    const int idx_fcl1 = 2 * (iflav + icol * nflav + ilay * ncol * nflav);
                    const int idx_tau = gpt_start + ilay*ngpt + icol*nlay*ngpt;

                    interpolate2D_byflav_kernel(&fminor[idx_fcl2], &kminor[kminor_start_lower[imnr]-1],
                                                kminor_start_lower[imnr]-1, kminor_start_lower[imnr]-1 + (gpt_end - gpt_start),
                                                &tau_minor[idx_tau], &jeta[idx_fcl1],
                                                jtemp[idx_collay], nminork_lower, neta);

                    for (int igpt = gpt_start; igpt < gpt_end; ++igpt)
                    {
                        const int idx_out = igpt + ilay * ngpt + icol * nlay * ngpt;
                        tau[idx_out] += tau_minor[idx_out] * scaling;
                    }
                }
            }
            else
            {
                for (int imnr = 0; imnr < nscale_upper; ++imnr)
                {
                    TF scaling = col_gas[idx_collay + idx_minor[imnr] * ncl];
                    if (minor_scales_with_density_upper[imnr])
                    {
                        scaling *= PaTohPa * play[idx_collay] / tlay[idx_collay];
                        if (idx_minor_scaling_upper[imnr] > 0)
                        {
                            TF vmr_fact = TF(1.) / col_gas[idx_collay];
                            TF dry_fact = TF(1.) / (TF(1.) + col_gas[idx_collaywv] * vmr_fact);
                            if (scale_by_complement_upper[imnr])
                            {
                                scaling *= (TF(1.) - col_gas[idx_collay + idx_minor_scaling_upper[imnr] * ncl] * vmr_fact * dry_fact);
                            }
                            else
                            {
                                scaling *= col_gas[idx_collay + idx_minor_scaling_upper[imnr] * ncl] * vmr_fact * dry_fact;
                            }
                        }
                    }
                    const int gpt_start = minor_limits_gpt_upper[2*imnr]-1;
                    const int gpt_end = minor_limits_gpt_upper[2*imnr+1];
                    const int iflav = gpoint_flavor[2*gpt_start+1]-1;
                    const int idx_fcl2 = 2 * 2 * (iflav + icol * nflav + ilay * ncol * nflav);
                    const int idx_fcl1 = 2 * (iflav + icol * nflav + ilay * ncol * nflav);
                    const int idx_tau = gpt_start + ilay*ngpt + icol*nlay*ngpt;

                    interpolate2D_byflav_kernel(&fminor[idx_fcl2], &kminor[kminor_start_upper[imnr]-1],
                                                kminor_start_upper[imnr]-1, kminor_start_upper[imnr]-1 + (gpt_end - gpt_start),
                                                &tau_minor[idx_tau], &jeta[idx_fcl1],
                                                jtemp[idx_collay], nminork_upper, neta);

                    for (int igpt = gpt_start; igpt < gpt_end; ++igpt)
                    {
                        const int idx_out = igpt + ilay * ngpt + icol * nlay * ngpt;
                        tau[idx_out] += tau_minor[idx_out] * scaling;
                    }
                }
            }
        }
    }

    template<typename TF>__global__
    void compute_tau_rayleigh_kernel(
            const int ncol, const int nlay, const int nbnd, const int ngpt,
            const int ngas, const int nflav, const int neta, const int npres, const int ntemp,
            const int* __restrict__ gpoint_flavor,
            const int* __restrict__ band_lims_gpt,
            const TF* __restrict__ krayl,
            int idx_h2o, const TF* __restrict__ col_dry, const TF* __restrict__ col_gas,
            const TF* __restrict__ fminor, const int* __restrict__ jeta,
            const BOOL_TYPE* __restrict__ tropo, const int* __restrict__ jtemp,
            TF* __restrict__ tau_rayleigh, TF* __restrict__ k)
    {
        // Fetch the three coordinates.
        const int ibnd = blockIdx.x*blockDim.x + threadIdx.x;
        const int ilay = blockIdx.y*blockDim.y + threadIdx.y;
        const int icol = blockIdx.z*blockDim.z + threadIdx.z;

        if ( (icol < ncol) && (ilay < nlay) && (ibnd < nbnd) )
        {
            //kernel implementation
            const int idx_collay = icol + ilay*ncol;
            const int idx_collaywv = icol + ilay*ncol + idx_h2o*nlay*ncol;
            const int itropo = !tropo[idx_collay];
            const int gpt_start = band_lims_gpt[2*ibnd]-1;
            const int gpt_end = band_lims_gpt[2*ibnd+1];
            const int iflav = gpoint_flavor[itropo+2*gpt_start]-1;
            const int idx_fcl2 = 2*2*(iflav + icol*nflav + ilay*ncol*nflav);
            const int idx_fcl1   = 2*(iflav + icol*nflav + ilay*ncol*nflav);
            const int idx_krayl  = gpt_start + ngpt*neta*ntemp*itropo;
            const int idx_k = gpt_start + ilay*ngpt + icol*nlay*ngpt;
            interpolate2D_byflav_kernel(&fminor[idx_fcl2],
                                        &krayl[idx_krayl],
                                        gpt_start, gpt_end, &k[idx_k],
                                        &jeta[idx_fcl1],
                                        jtemp[idx_collay],
                                        ngpt, neta);

            for (int igpt=gpt_start; igpt<gpt_end; ++igpt)
            {
                const int idx_out = igpt + ilay*ngpt + icol*nlay*ngpt;
                tau_rayleigh[idx_out] = k[idx_k+igpt-gpt_start]*(col_gas[idx_collaywv]+col_dry[idx_collay]);
            }
        }
    }

    
    template<typename TF>__global__
    void combine_and_reorder_2str_kernel(
            const int ncol, const int nlay, const int ngpt, const TF tmin,
            const TF* __restrict__ tau_abs, const TF* __restrict__ tau_rayleigh,
            TF* __restrict__ tau, TF* __restrict__ ssa, TF* __restrict__ g)
    {
        // Fetch the three coordinates.
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;
        const int igpt = blockIdx.y*blockDim.y + threadIdx.y;
        const int ilay = blockIdx.z*blockDim.z + threadIdx.z;

        if ( (icol < ncol) && (ilay < nlay) && (igpt < ngpt) )
        {
            const int idx_in  = igpt + ilay*ngpt + icol*(ngpt*nlay);
            const int idx_out = icol + ilay*ncol + igpt*(ncol*nlay);
	   
            const TF tau_tot = tau_abs[idx_in] + tau_rayleigh[idx_in];
            tau[idx_out] = tau_tot;
            g  [idx_out] = TF(0.);
            if (tau_tot>(TF(2.)*tmin))
                ssa[idx_out] = tau_rayleigh[idx_in]/tau_tot;
            else
                ssa[idx_out] = 0.;
        }
    }
    
    
    
}

namespace rrtmgp_kernel_launcher_cuda
{
    template<typename TF>
    void interpolation(
            const int ncol, const int nlay,
            const int ngas, const int nflav, const int neta, const int npres, const int ntemp,
            const Array<int,2>& flavor,
            const Array<TF,1>& press_ref_log,
            const Array<TF,1>& temp_ref,
            TF press_ref_log_delta,
            TF temp_ref_min,
            TF temp_ref_delta,
            TF press_ref_trop_log,
            const Array<TF,3>& vmr_ref,
            const Array<TF,2>& play,
            const Array<TF,2>& tlay,
            Array<TF,3>& col_gas,
            Array<int,2>& jtemp,
            Array<TF,6>& fmajor, Array<TF,5>& fminor,
            Array<TF,4>& col_mix,
            Array<BOOL_TYPE,2>& tropo,
            Array<int,4>& jeta,
            Array<int,2>& jpress)
    {
        const int flavor_size = flavor.size() * sizeof(int);
        const int press_ref_log_size = press_ref_log.size() * sizeof(TF);
        const int temp_ref_size = temp_ref.size() * sizeof(TF);
        const int vmr_ref_size = vmr_ref.size() * sizeof(TF);
        const int collay_tf_size = ncol * nlay * sizeof(TF);
        const int collay_int_size = ncol * nlay * sizeof(int);
        const int col_gas_size = col_gas.size() * sizeof(TF);
        const int jtemp_size = jtemp.size() * sizeof(int);
        const int fmajor_size = fmajor.size() * sizeof(TF);
        const int fminor_size = fminor.size() * sizeof(TF);
        const int col_mix_size = col_mix.size() * sizeof(TF);
        const int tropo_size = tropo.size() * sizeof(BOOL_TYPE);
        const int jeta_size = jeta.size() * sizeof(int);
        const int jpress_size = jpress.size() * sizeof(int);

        int* flavor_gpu;
        TF* press_ref_log_gpu;
        TF* temp_ref_gpu;
        TF* vmr_ref_gpu;
        TF* play_gpu;
        TF* tlay_gpu;
        TF* col_gas_gpu;
        int* jtemp_gpu;
        TF* fmajor_gpu;
        TF* fminor_gpu;
        TF* col_mix_gpu;
        BOOL_TYPE* tropo_gpu;
        int* jeta_gpu;
        int* jpress_gpu;

        cuda_safe_call(cudaMalloc((void **) &flavor_gpu, flavor_size));
        cuda_safe_call(cudaMalloc((void **) &press_ref_log_gpu, press_ref_log_size));
        cuda_safe_call(cudaMalloc((void **) &temp_ref_gpu, temp_ref_size));
        cuda_safe_call(cudaMalloc((void **) &vmr_ref_gpu, vmr_ref_size));
        cuda_safe_call(cudaMalloc((void **) &play_gpu, collay_tf_size));
        cuda_safe_call(cudaMalloc((void **) &tlay_gpu, collay_tf_size));
        cuda_safe_call(cudaMalloc((void **) &col_gas_gpu, col_gas_size));
        cuda_safe_call(cudaMalloc((void **) &jtemp_gpu, collay_int_size));
        cuda_safe_call(cudaMalloc((void **) &fmajor_gpu, fmajor_size));
        cuda_safe_call(cudaMalloc((void **) &fminor_gpu, fminor_size));
        cuda_safe_call(cudaMalloc((void **) &col_mix_gpu, col_mix_size));
        cuda_safe_call(cudaMalloc((void **) &tropo_gpu, tropo_size));
        cuda_safe_call(cudaMalloc((void **) &jeta_gpu, jeta_size));
        cuda_safe_call(cudaMalloc((void **) &jpress_gpu, collay_int_size));

        cuda_safe_call(cudaMemcpy(flavor_gpu, flavor.ptr(), flavor_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(press_ref_log_gpu, press_ref_log.ptr(), press_ref_log_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(temp_ref_gpu, temp_ref.ptr(), temp_ref_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(vmr_ref_gpu, vmr_ref.ptr(), vmr_ref_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(play_gpu, play.ptr(), collay_tf_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(tlay_gpu, tlay.ptr(), collay_tf_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(col_gas_gpu, col_gas.ptr(), col_gas_size, cudaMemcpyHostToDevice));

        cudaEvent_t startEvent, stopEvent;
        float elapsedtime;
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
        cudaEventRecord(startEvent, 0);

        const int block_lay = 16;
        const int block_col = 32;

        const int grid_lay  = nlay/block_lay + (nlay%block_lay > 0);
        const int grid_col  = ncol/block_col + (ncol%block_col > 0);

        dim3 grid_gpu(grid_lay, grid_col);
        dim3 block_gpu(block_lay, block_col);

        TF tmin = std::numeric_limits<TF>::min();
        interpolation_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, ngas, nflav, neta, npres, ntemp, tmin,
                flavor_gpu, press_ref_log_gpu, temp_ref_gpu,
                press_ref_log_delta, temp_ref_min,
                temp_ref_delta, press_ref_trop_log,
                vmr_ref_gpu, play_gpu, tlay_gpu,
                col_gas_gpu, jtemp_gpu, fmajor_gpu,
                fminor_gpu, col_mix_gpu, tropo_gpu,
                jeta_gpu, jpress_gpu);

        cuda_check_error();
        cuda_safe_call(cudaDeviceSynchronize());
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&elapsedtime,startEvent,stopEvent);
        std::cout<<"GPU interpolation: "<<elapsedtime<<" (ms)"<<std::endl;

        // Copy back the results.
        cuda_safe_call(cudaMemcpy(jtemp.ptr(), jtemp_gpu, jtemp_size, cudaMemcpyDeviceToHost));
        cuda_safe_call(cudaMemcpy(fmajor.ptr(), fmajor_gpu, fmajor_size, cudaMemcpyDeviceToHost));
        cuda_safe_call(cudaMemcpy(fminor.ptr(), fminor_gpu, fminor_size, cudaMemcpyDeviceToHost));
        cuda_safe_call(cudaMemcpy(col_mix.ptr(), col_mix_gpu, col_mix_size, cudaMemcpyDeviceToHost));
        cuda_safe_call(cudaMemcpy(tropo.ptr(), tropo_gpu, tropo_size, cudaMemcpyDeviceToHost));
        cuda_safe_call(cudaMemcpy(jeta.ptr(), jeta_gpu, jeta_size, cudaMemcpyDeviceToHost));
        cuda_safe_call(cudaMemcpy(jpress.ptr(), jpress_gpu, jpress_size, cudaMemcpyDeviceToHost));

        // Deallocate a CUDA array.
        cuda_safe_call(cudaFree(flavor_gpu));
        cuda_safe_call(cudaFree(press_ref_log_gpu));
        cuda_safe_call(cudaFree(temp_ref_gpu));
        cuda_safe_call(cudaFree(vmr_ref_gpu));
        cuda_safe_call(cudaFree(play_gpu));
        cuda_safe_call(cudaFree(tlay_gpu));
        cuda_safe_call(cudaFree(col_gas_gpu));
        cuda_safe_call(cudaFree(jtemp_gpu));
        cuda_safe_call(cudaFree(fmajor_gpu));
        cuda_safe_call(cudaFree(fminor_gpu));
        cuda_safe_call(cudaFree(col_mix_gpu));
        cuda_safe_call(cudaFree(tropo_gpu));
        cuda_safe_call(cudaFree(jeta_gpu));
        cuda_safe_call(cudaFree(jpress_gpu));
    }

    template<typename TF>
    void combine_and_reorder_2str(
            const int ncol, const int nlay, const int ngpt,
            const Array<TF,3>& tau_abs, const Array<TF,3>& tau_rayleigh,
            Array<TF,3>& tau, Array<TF,3>& ssa, Array<TF,3>& g)
    {
        // Store the sizes, all the same
        const int array_size = tau_abs.size()*sizeof(TF);

        TF* tau_abs_gpu;
        TF* tau_rayleigh_gpu;
        TF* tau_gpu;
        TF* ssa_gpu;
        TF* g_gpu;

        TF tmin = std::numeric_limits<TF>::min();
        // Allocate a CUDA array.
        cuda_safe_call(cudaMalloc((void**)&tau_abs_gpu, array_size));
        cuda_safe_call(cudaMalloc((void**)&tau_rayleigh_gpu, array_size));
        cuda_safe_call(cudaMalloc((void**)&tau_gpu, array_size));
        cuda_safe_call(cudaMalloc((void**)&ssa_gpu, array_size));
        cuda_safe_call(cudaMalloc((void**)&g_gpu, array_size));

        // Copy the data to the GPU.
        cuda_safe_call(cudaMemcpy(tau_abs_gpu, tau_abs.ptr(), array_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(tau_rayleigh_gpu, tau_rayleigh.ptr(), array_size, cudaMemcpyHostToDevice));
        cudaEvent_t startEvent, stopEvent;
        float elapsedtime;
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);

        cudaEventRecord(startEvent, 0);
        // Call the kernel.
        const int block_col = 32;
        const int block_gpt = 32;
        const int block_lay = 1;

        const int grid_col  = ncol/block_col + (ncol%block_col > 0);
        const int grid_gpt  = ngpt/block_gpt + (ngpt%block_gpt > 0);
        const int grid_lay  = nlay/block_lay + (nlay%block_lay > 0);

        dim3 grid_gpu(grid_col, grid_gpt, grid_lay);
        dim3 block_gpu(block_col, block_gpt, block_lay);

        combine_and_reorder_2str_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, ngpt, tmin,
                tau_abs_gpu, tau_rayleigh_gpu,
                tau_gpu, ssa_gpu, g_gpu);

        cuda_check_error();
        cuda_safe_call(cudaDeviceSynchronize());
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&elapsedtime,startEvent,stopEvent);
        std::cout<<"GPU combine_and_reorder_2str: "<<elapsedtime<<" (ms)"<<std::endl;

        // Copy back the results.
        cuda_safe_call(cudaMemcpy(tau.ptr(), tau_gpu, array_size, cudaMemcpyDeviceToHost));
        cuda_safe_call(cudaMemcpy(ssa.ptr(), ssa_gpu, array_size, cudaMemcpyDeviceToHost));
        cuda_safe_call(cudaMemcpy(g.ptr(), g_gpu, array_size, cudaMemcpyDeviceToHost));

        // Deallocate a CUDA array.
        cuda_safe_call(cudaFree(tau_abs_gpu));
        cuda_safe_call(cudaFree(tau_rayleigh_gpu));
        cuda_safe_call(cudaFree(tau_gpu));
        cuda_safe_call(cudaFree(ssa_gpu));
        cuda_safe_call(cudaFree(g_gpu));
    }
    
    template<typename TF>
    void compute_tau_rayleigh(
            const int ncol, const int nlay, const int nbnd, const int ngpt,
            const int ngas, const int nflav, const int neta, const int npres, const int ntemp,
            const Array<int,2>& gpoint_flavor,
            const Array<int,2>& band_lims_gpt,
            const Array<TF,4>& krayl,
            int idx_h2o, const Array<TF,2>& col_dry, const Array<TF,3>& col_gas,
            const Array<TF,5>& fminor, const Array<int,4>& jeta,
            const Array<BOOL_TYPE,2>& tropo, const Array<int,2>& jtemp,
            Array<TF,3>& tau_rayleigh)
    {
        float elapsedtime;
        const int gpoint_flavor_size = gpoint_flavor.size()*sizeof(int);
        const int band_lims_gpt_size = band_lims_gpt.size()*sizeof(int);
        const int krayl_size = krayl.size()*sizeof(TF);
        const int col_dry_size = col_dry.size()*sizeof(TF);
        const int col_gas_size = col_gas.size()*sizeof(TF);
        const int fminor_size = fminor.size()*sizeof(TF);
        const int jeta_size = jeta.size()*sizeof(int);
        const int tropo_size = tropo.size()*sizeof(BOOL_TYPE);
        const int jtemp_size = jtemp.size()*sizeof(int);
        const int tau_rayleigh_size = tau_rayleigh.size()*sizeof(TF);

        int* gpoint_flavor_gpu;
        int* band_lims_gpt_gpu;
        int* jeta_gpu;
        int* jtemp_gpu;
        BOOL_TYPE* tropo_gpu;
        TF* krayl_gpu;
        TF* col_dry_gpu;
        TF* col_gas_gpu;
        TF* fminor_gpu;
        TF* tau_rayleigh_gpu;
        TF* k_gpu;

        // Allocate a CUDA array.
        cuda_safe_call(cudaMalloc((void**)&gpoint_flavor_gpu, gpoint_flavor_size));
        cuda_safe_call(cudaMalloc((void**)&band_lims_gpt_gpu, band_lims_gpt_size));
        cuda_safe_call(cudaMalloc((void**)&krayl_gpu, krayl_size));
        cuda_safe_call(cudaMalloc((void**)&col_dry_gpu, col_dry_size));
        cuda_safe_call(cudaMalloc((void**)&col_gas_gpu, col_gas_size));
        cuda_safe_call(cudaMalloc((void**)&fminor_gpu, fminor_size));
        cuda_safe_call(cudaMalloc((void**)&jeta_gpu, jeta_size));
        cuda_safe_call(cudaMalloc((void**)&tropo_gpu, tropo_size));
        cuda_safe_call(cudaMalloc((void**)&jtemp_gpu, jtemp_size));
        cuda_safe_call(cudaMalloc((void**)&tau_rayleigh_gpu, tau_rayleigh_size));
        cuda_safe_call(cudaMalloc((void**)&k_gpu, tau_rayleigh_size));

        // Copy the data to the GPU.
        cuda_safe_call(cudaMemcpy(gpoint_flavor_gpu, gpoint_flavor.ptr(), gpoint_flavor_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(band_lims_gpt_gpu, band_lims_gpt.ptr(), band_lims_gpt_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(krayl_gpu, krayl.ptr(), krayl_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(col_dry_gpu, col_dry.ptr(), col_dry_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(col_gas_gpu, col_gas.ptr(), col_gas_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(fminor_gpu, fminor.ptr(), fminor_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(jeta_gpu, jeta.ptr(), jeta_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(tropo_gpu, tropo.ptr(), tropo_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(jtemp_gpu, jtemp.ptr(), jtemp_size, cudaMemcpyHostToDevice));

        cudaEvent_t startEvent, stopEvent;
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
        cudaEventRecord(startEvent, 0);

        // Call the kernel.
        const int block_bnd = 14;
        const int block_lay = 1;
        const int block_col = 32;

        const int grid_bnd  = nbnd/block_bnd + (nbnd%block_bnd > 0);
        const int grid_lay  = nlay/block_lay + (nlay%block_lay > 0);
        const int grid_col  = ncol/block_col + (ncol%block_col > 0);

        dim3 grid_gpu(grid_bnd, grid_lay, grid_col);
        dim3 block_gpu(block_bnd, block_lay, block_col);

        compute_tau_rayleigh_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, nbnd, ngpt,
                ngas, nflav, neta, npres, ntemp,
                gpoint_flavor_gpu,
                band_lims_gpt_gpu,
                krayl_gpu,
                idx_h2o, col_dry_gpu, col_gas_gpu,
                fminor_gpu, jeta_gpu,
                tropo_gpu, jtemp_gpu,
                tau_rayleigh_gpu, k_gpu);

        cuda_check_error();
        cuda_safe_call(cudaDeviceSynchronize());
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&elapsedtime,startEvent,stopEvent);
        std::cout<<"GPU compute_tau_rayleigh: "<<elapsedtime<<" (ms)"<<std::endl;

        // Copy back the results.
        cuda_safe_call(cudaMemcpy(tau_rayleigh.ptr(), tau_rayleigh_gpu, tau_rayleigh_size, cudaMemcpyDeviceToHost));
        
        // Deallocate a CUDA array.
        cuda_safe_call(cudaFree(gpoint_flavor_gpu));
        cuda_safe_call(cudaFree(band_lims_gpt_gpu));
        cuda_safe_call(cudaFree(krayl_gpu));
        cuda_safe_call(cudaFree(col_dry_gpu));
        cuda_safe_call(cudaFree(col_gas_gpu));
        cuda_safe_call(cudaFree(fminor_gpu));
        cuda_safe_call(cudaFree(jeta_gpu));
        cuda_safe_call(cudaFree(tropo_gpu));
        cuda_safe_call(cudaFree(jtemp_gpu));
        cuda_safe_call(cudaFree(tau_rayleigh_gpu));
        cuda_safe_call(cudaFree(k_gpu));
    }

    template<typename TF>
    void compute_tau_absorption(
            const int ncol, const int nlay, const int nband, const int ngpt,
            const int ngas, const int nflav, const int neta, const int npres, const int ntemp,
            const int nminorlower, const int nminorklower,
            const int nminorupper, const int nminorkupper,
            const int idx_h2o,
            const Array<int,2>& gpoint_flavor,
            const Array<int,2>& band_lims_gpt,
            const Array<TF,4>& kmajor,
            const Array<TF,3>& kminor_lower,
            const Array<TF,3>& kminor_upper,
            const Array<int,2>& minor_limits_gpt_lower,
            const Array<int,2>& minor_limits_gpt_upper,
            const Array<BOOL_TYPE,1>& minor_scales_with_density_lower,
            const Array<BOOL_TYPE,1>& minor_scales_with_density_upper,
            const Array<BOOL_TYPE,1>& scale_by_complement_lower,
            const Array<BOOL_TYPE,1>& scale_by_complement_upper,
            const Array<int,1>& idx_minor_lower,
            const Array<int,1>& idx_minor_upper,
            const Array<int,1>& idx_minor_scaling_lower,
            const Array<int,1>& idx_minor_scaling_upper,
            const Array<int,1>& kminor_start_lower,
            const Array<int,1>& kminor_start_upper,
            const Array<BOOL_TYPE,2>& tropo,
            const Array<TF,4>& col_mix, const Array<TF,6>& fmajor,
            const Array<TF,5>& fminor, const Array<TF,2>& play,
            const Array<TF,2>& tlay, const Array<TF,3>& col_gas,
            const Array<int,4>& jeta, const Array<int,2>& jtemp,
            const Array<int,2>& jpress, Array<TF,3>& tau)
    {
        float elapsedtime;
        const int gpoint_flavor_size = gpoint_flavor.size()*sizeof(int);
        const int band_lims_gpt_size = band_lims_gpt.size()*sizeof(int);
        const int kmajor_size = kmajor.size()*sizeof(TF);
        const int kminor_lower_size = kminor_lower.size()*sizeof(TF);
        const int kminor_upper_size = kminor_upper.size()*sizeof(TF);
        const int minor_limits_gpt_lower_size = minor_limits_gpt_lower.size()*sizeof(int);
        const int minor_limits_gpt_upper_size = minor_limits_gpt_upper.size()*sizeof(int);
        const int nminorlower_bool_size = nminorlower*sizeof(BOOL_TYPE); //minor scales with/scale by complement
        const int nminorupper_bool_size = nminorlower*sizeof(BOOL_TYPE); //minor scales with/scale by complement
        const int nminorlower_int_size = nminorlower*sizeof(int); //idx_minor(scaling) kminor
        const int nminorupper_int_size = nminorlower*sizeof(int);
        const int tropo_size = tropo.size()*sizeof(BOOL_TYPE);
        const int col_mix_size = col_mix.size()*sizeof(TF);
        const int fmajor_size = fmajor.size()*sizeof(TF);
        const int fminor_size = fminor.size()*sizeof(TF);
        const int collay_tf_size = ncol*nlay*sizeof(TF); //tlay,play
        const int col_gas_size = col_gas.size()*sizeof(TF);
        const int jeta_size =  jeta.size()*sizeof(int);
        const int collay_int_size = ncol*nlay*sizeof(int);
        const int itropo_size = 2*ncol*sizeof(int);
        const int tau_size = tau.size()*sizeof(TF);
        
        int* gpoint_flavor_gpu;
        int* band_lims_gpt_gpu;
        TF* kmajor_gpu;
        TF* kminor_lower_gpu;
        TF* kminor_upper_gpu;
        int* minor_limits_gpt_lower_gpu;
        int* minor_limits_gpt_upper_gpu;
        BOOL_TYPE* minor_scales_with_density_lower_gpu;
        BOOL_TYPE* minor_scales_with_density_upper_gpu;
        BOOL_TYPE* scale_by_complement_lower_gpu;
        BOOL_TYPE* scale_by_complement_upper_gpu;
        int* idx_minor_lower_gpu;
        int* idx_minor_upper_gpu;
        int* idx_minor_scaling_lower_gpu;
        int* idx_minor_scaling_upper_gpu;
        int* kminor_start_lower_gpu;
        int* kminor_start_upper_gpu;
        BOOL_TYPE* tropo_gpu;
        TF* col_mix_gpu;
        TF* fmajor_gpu;
        TF* fminor_gpu;
        TF* play_gpu;
        TF* tlay_gpu;
        TF* col_gas_gpu;
        int* jeta_gpu;
        int* jtemp_gpu;
        int* jpress_gpu;
        int* itropo_lower_gpu;
        int* itropo_upper_gpu;
        TF* tau_gpu;
        TF* tau_major_gpu;
        TF* tau_minor_gpu;

        // Allocate a CUDA array.
        cuda_safe_call(cudaMalloc((void**)& gpoint_flavor_gpu, gpoint_flavor_size));
        cuda_safe_call(cudaMalloc((void**)& band_lims_gpt_gpu, band_lims_gpt_size));
        cuda_safe_call(cudaMalloc((void**)& kmajor_gpu, kmajor_size));
        cuda_safe_call(cudaMalloc((void**)& kminor_lower_gpu, kminor_lower_size));
        cuda_safe_call(cudaMalloc((void**)& kminor_upper_gpu, kminor_upper_size));
        cuda_safe_call(cudaMalloc((void**)& minor_limits_gpt_lower_gpu, minor_limits_gpt_lower_size));
        cuda_safe_call(cudaMalloc((void**)& minor_limits_gpt_upper_gpu, minor_limits_gpt_upper_size));
        cuda_safe_call(cudaMalloc((void**)& minor_scales_with_density_lower_gpu, nminorlower_bool_size));
        cuda_safe_call(cudaMalloc((void**)& minor_scales_with_density_upper_gpu, nminorupper_bool_size));
        cuda_safe_call(cudaMalloc((void**)& scale_by_complement_lower_gpu, nminorlower_bool_size));
        cuda_safe_call(cudaMalloc((void**)& scale_by_complement_upper_gpu, nminorupper_bool_size));
        cuda_safe_call(cudaMalloc((void**)& idx_minor_lower_gpu, nminorlower_int_size));
        cuda_safe_call(cudaMalloc((void**)& idx_minor_upper_gpu, nminorupper_int_size));
        cuda_safe_call(cudaMalloc((void**)& idx_minor_scaling_lower_gpu, nminorlower_int_size));
        cuda_safe_call(cudaMalloc((void**)& idx_minor_scaling_upper_gpu, nminorupper_int_size));
        cuda_safe_call(cudaMalloc((void**)& kminor_start_lower_gpu, nminorlower_int_size));
        cuda_safe_call(cudaMalloc((void**)& kminor_start_upper_gpu, nminorupper_int_size));
        cuda_safe_call(cudaMalloc((void**)& tropo_gpu, tropo_size));
        cuda_safe_call(cudaMalloc((void**)& col_mix_gpu, col_mix_size));
        cuda_safe_call(cudaMalloc((void**)& fmajor_gpu, fmajor_size));
        cuda_safe_call(cudaMalloc((void**)& fminor_gpu, fminor_size));
        cuda_safe_call(cudaMalloc((void**)& play_gpu, collay_tf_size));
        cuda_safe_call(cudaMalloc((void**)& tlay_gpu, collay_tf_size));
        cuda_safe_call(cudaMalloc((void**)& col_gas_gpu, col_gas_size));
        cuda_safe_call(cudaMalloc((void**)& jeta_gpu, jeta_size));
        cuda_safe_call(cudaMalloc((void**)& jtemp_gpu, collay_int_size));
        cuda_safe_call(cudaMalloc((void**)& jpress_gpu, collay_int_size));
        cuda_safe_call(cudaMalloc((void**)& itropo_lower_gpu, itropo_size));
        cuda_safe_call(cudaMalloc((void**)& itropo_upper_gpu, itropo_size));
        cuda_safe_call(cudaMalloc((void**)& tau_gpu, tau_size));
        cuda_safe_call(cudaMalloc((void**)& tau_major_gpu, tau_size));
        cuda_safe_call(cudaMalloc((void**)& tau_minor_gpu, tau_size));

        // Copy the data to the GPU.
        cuda_safe_call(cudaMemcpy(gpoint_flavor_gpu, gpoint_flavor.ptr(), gpoint_flavor_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(band_lims_gpt_gpu, band_lims_gpt.ptr(), band_lims_gpt_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(kmajor_gpu, kmajor.ptr(), kmajor_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(kminor_lower_gpu, kminor_lower.ptr(), kminor_lower_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(kminor_upper_gpu, kminor_upper.ptr(), kminor_upper_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(minor_limits_gpt_lower_gpu, minor_limits_gpt_lower.ptr(), minor_limits_gpt_lower_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(minor_limits_gpt_upper_gpu, minor_limits_gpt_upper.ptr(), minor_limits_gpt_upper_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(minor_scales_with_density_lower_gpu, minor_scales_with_density_lower.ptr(), nminorlower_bool_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(minor_scales_with_density_upper_gpu, minor_scales_with_density_upper.ptr(), nminorupper_bool_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(scale_by_complement_lower_gpu, scale_by_complement_lower.ptr(), nminorlower_bool_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(scale_by_complement_upper_gpu, scale_by_complement_upper.ptr(), nminorupper_bool_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(idx_minor_lower_gpu, idx_minor_lower.ptr(), nminorlower_int_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(idx_minor_upper_gpu, idx_minor_upper.ptr(), nminorupper_int_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(idx_minor_scaling_lower_gpu, idx_minor_scaling_lower.ptr(), nminorlower_int_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(idx_minor_scaling_upper_gpu, idx_minor_scaling_upper.ptr(), nminorupper_int_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(kminor_start_lower_gpu, kminor_start_lower.ptr(), nminorlower_int_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(kminor_start_upper_gpu, kminor_start_upper.ptr(), nminorupper_int_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(tropo_gpu, tropo.ptr(), tropo_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(col_mix_gpu, col_mix.ptr(), col_mix_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(fmajor_gpu, fmajor.ptr(), fmajor_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(fminor_gpu, fminor.ptr(), fminor_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(play_gpu, play.ptr(), collay_tf_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(tlay_gpu, tlay.ptr(), collay_tf_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(col_gas_gpu, col_gas.ptr(), col_gas_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(jeta_gpu, jeta.ptr(), jeta_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(jtemp_gpu, jtemp.ptr(), collay_int_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(jpress_gpu, jpress.ptr(), collay_int_size, cudaMemcpyHostToDevice));

        cudaEvent_t startEvent, stopEvent;
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
        cudaEventRecord(startEvent, 0);

        // Call the kernel.
        const int block_bnd_maj = 14;
        const int block_lay_maj = 1;
        const int block_col_maj = 32;

        const int grid_bnd_maj  = nband/block_bnd_maj + (nband%block_bnd_maj > 0);
        const int grid_lay_maj  = nlay/block_lay_maj + (nlay%block_lay_maj > 0);
        const int grid_col_maj  = ncol/block_col_maj + (ncol%block_col_maj > 0);

        dim3 grid_gpu_maj(grid_bnd_maj, grid_lay_maj, grid_col_maj);
        dim3 block_gpu_maj(block_bnd_maj, block_lay_maj, block_col_maj);

        compute_tau_major_absorption_kernel<<<grid_gpu_maj, block_gpu_maj>>>(
                ncol, nlay, nband, ngpt,
                nflav, neta, npres, ntemp,
                gpoint_flavor_gpu, band_lims_gpt_gpu,
                kmajor_gpu, col_mix_gpu, fmajor_gpu, jeta_gpu,
                tropo_gpu, jtemp_gpu, jpress_gpu,
                tau_gpu, tau_major_gpu);

        const int nscale_lower = scale_by_complement_lower.dim(1);
        const int nscale_upper = scale_by_complement_upper.dim(1);
        const int block_lay_min = 14;
        const int block_col_min = 32;

        const int grid_lay_min  = nlay/block_lay_min + (nlay%block_lay_min > 0);
        const int grid_col_min  = ncol/block_col_min + (ncol%block_col_min > 0);

        dim3 grid_gpu_min(grid_lay_min, grid_col_min);
        dim3 block_gpu_min(block_lay_min, block_col_min);

        compute_tau_minor_absorption_kernel<<<grid_gpu_min, block_gpu_min>>>(
                ncol, nlay, ngpt,
                ngas, nflav, ntemp, neta,
                nscale_lower, nscale_upper
                nminorlower, nminorupper,
                nminorklower,nminorkupper,
                idx_h2o,
                gpoint_flavor_gpu,
                kminor_lower_gpu, kminor_upper_gpu,
                minor_limits_gpt_lower_gpu, minor_limits_gpt_upper_gpu,
                minor_scales_with_density_lower_gpu, inor_scales_with_density_upper_gpu,
                scale_by_complement_lower_gpu, scale_by_complement_upper_gpu,
                idx_minor_lower_gpu, idx_minor_upper_gpu,
                idx_minor_scaling_lower_gpu, idx_minor_scaling_upper_gpu,
                kminor_start_lower_gpu, kminor_start_upper_gpu,
                play_gpu, tlay_gpu, col_gas_gpu,
                fminor_gpu, jeta_gpu, jtemp_gpu,
                tropo_gpu, tau_gpu, tau_minor_gpu);

        cuda_check_error();
        cuda_safe_call(cudaDeviceSynchronize());
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&elapsedtime,startEvent,stopEvent);
        std::cout<<"GPU compute_tau_abs: "<<elapsedtime<<" (ms)"<<std::endl;

        // Copy back the results.
        cuda_safe_call(cudaMemcpy(tau.ptr(), tau_gpu, tau_size, cudaMemcpyDeviceToHost));

        // Deallocate a CUDA array.
        cuda_safe_call(cudaFree(gpoint_flavor_gpu));
        cuda_safe_call(cudaFree(band_lims_gpt_gpu));
        cuda_safe_call(cudaFree(kmajor_gpu));
        cuda_safe_call(cudaFree(kminor_lower_gpu));
        cuda_safe_call(cudaFree(kminor_upper_gpu));
        cuda_safe_call(cudaFree(minor_limits_gpt_lower_gpu));
        cuda_safe_call(cudaFree(minor_limits_gpt_upper_gpu));
        cuda_safe_call(cudaFree(minor_scales_with_density_lower_gpu));
        cuda_safe_call(cudaFree(minor_scales_with_density_upper_gpu));
        cuda_safe_call(cudaFree(scale_by_complement_lower_gpu));
        cuda_safe_call(cudaFree(scale_by_complement_upper_gpu));
        cuda_safe_call(cudaFree(idx_minor_lower_gpu));
        cuda_safe_call(cudaFree(idx_minor_upper_gpu));
        cuda_safe_call(cudaFree(idx_minor_scaling_lower_gpu));
        cuda_safe_call(cudaFree(idx_minor_scaling_upper_gpu));
        cuda_safe_call(cudaFree(kminor_start_lower_gpu));
        cuda_safe_call(cudaFree(kminor_start_upper_gpu));
        cuda_safe_call(cudaFree(tropo_gpu));
        cuda_safe_call(cudaFree(col_mix_gpu));
        cuda_safe_call(cudaFree(fmajor_gpu));
        cuda_safe_call(cudaFree(fminor_gpu));
        cuda_safe_call(cudaFree(play_gpu));
        cuda_safe_call(cudaFree(tlay_gpu));
        cuda_safe_call(cudaFree(col_gas_gpu));
        cuda_safe_call(cudaFree(jeta_gpu));
        cuda_safe_call(cudaFree(jtemp_gpu));
        cuda_safe_call(cudaFree(jpress_gpu));
        cuda_safe_call(cudaFree(itropo_lower_gpu));
        cuda_safe_call(cudaFree(itropo_upper_gpu));
        cuda_safe_call(cudaFree(tau_major_gpu));
        cuda_safe_call(cudaFree(tau_minor_gpu));
        cuda_safe_call(cudaFree(tau_gpu));
    }
}


#ifdef FLOAT_SINGLE_RRTMGP
template void rrtmgp_kernel_launcher_cuda::interpolation(
        const int, const int, const int, const int, const int, const int, const int,
        const Array<int,2>&, const Array<single,1>&, const Array<single,1>&,
        single, single, single, single, const Array<single,3>&, const Array<single,2>&,
        const Array<single,2>&, Array<single,3>&, Array<int,2>&, Array<single,6>&, Array<single,5>&,
        Array<single,4>&, Array<BOOL_TYPE,2>&, Array<int,4>&, Array<int,2>&);

template void rrtmgp_kernel_launcher_cuda::combine_and_reorder_2str<float>(
        const int, const int, const int, const Array<float,3>&, const Array<float,3>&, Array<float,3>&, Array<float,3>&, Array<float,3>&);

template void rrtmgp_kernel_launcher_cuda::compute_tau_rayleigh<float>(
        const int, const int, const int, const int, const int, const int, const int, const int, const int,
        const Array<int,2>&, const Array<int,2>&, const Array<float,4>&, int, const Array<float,2>&, 
        const Array<float,3>&, const Array<float,5>&, const Array<int,4>&, const Array<BOOL_TYPE,2>&, 
        const Array<int,2>&, Array<float,3>&);

template void rrtmgp_kernel_launcher_cuda::compute_tau_absorption<float>(const int, const int, const int, const int, const int, const int, 
	const int, const int, const int, const int, const int, const int, const int, const int,
        const Array<int,2>&, const Array<int,2>&, const Array<float,4>&, const Array<float,3>&, const Array<float,3>&,
        const Array<int,2>&, const Array<int,2>&, const Array<BOOL_TYPE,1>&, const Array<BOOL_TYPE,1>&,
        const Array<BOOL_TYPE,1>&, const Array<BOOL_TYPE,1>&, const Array<int,1>&, const Array<int,1>&,
        const Array<int,1>&, const Array<int,1>&, const Array<int,1>&, const Array<int,1>&, const Array<BOOL_TYPE,2>& tropo,
        const Array<float,4>&, const Array<float,6>&, const Array<float,5>&, const Array<float,2>&, const Array<float,2>&, const Array<float,3>&,
        const Array<int,4>&, const Array<int,2>&, const Array<int,2>&, Array<float,3>&);

#else
template void rrtmgp_kernel_launcher_cuda::interpolation(
        const int, const int, const int, const int, const int, const int, const int,
        const Array<int,2>&, const Array<double,1>&, const Array<double,1>&,
        double, double, double, double, const Array<double,3>&, const Array<double,2>&,
        const Array<double,2>&, Array<double,3>&, Array<int,2>&, Array<double,6>&, Array<double,5>&,
        Array<double,4>&, Array<BOOL_TYPE,2>&, Array<int,4>&, Array<int,2>&);

template void rrtmgp_kernel_launcher_cuda::combine_and_reorder_2str<double>(
        const int, const int, const int, const Array<double,3>&, const Array<double,3>&, Array<double,3>&, Array<double,3>&, Array<double,3>&);

template void rrtmgp_kernel_launcher_cuda::compute_tau_rayleigh<double>(
        const int, const int, const int, const int, const int, const int, const int, const int, const int,
        const Array<int,2>&, const Array<int,2>&, const Array<double,4>&, int, const Array<double,2>&, 
        const Array<double,3>&, const Array<double,5>&, const Array<int,4>&, const Array<BOOL_TYPE,2>&, 
        const Array<int,2>&, Array<double,3>&);

template void rrtmgp_kernel_launcher_cuda::compute_tau_absorption<double>(const int, const int, const int, const int, const int, const int, 
	const int, const int, const int, const int, const int, const int, const int, const int,
        const Array<int,2>&, const Array<int,2>&, const Array<double,4>&, const Array<double,3>&, const Array<double,3>&,
        const Array<int,2>&, const Array<int,2>&, const Array<BOOL_TYPE,1>&, const Array<BOOL_TYPE,1>&,
        const Array<BOOL_TYPE,1>&, const Array<BOOL_TYPE,1>&, const Array<int,1>&, const Array<int,1>&,
        const Array<int,1>&, const Array<int,1>&, const Array<int,1>&, const Array<int,1>&, const Array<BOOL_TYPE,2>& tropo,
        const Array<double,4>&, const Array<double,6>&, const Array<double,5>&, const Array<double,2>&, const Array<double,2>&, const Array<double,3>&,
        const Array<int,4>&, const Array<int,2>&, const Array<int,2>&, Array<double,3>&);
#endif


