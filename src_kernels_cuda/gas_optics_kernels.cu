#include "Types.h"

template<typename TF> __device__
TF interpolate1D(const TF val,
                 const TF offset,
                 const TF delta,
                 const int len,
                 const TF* __restrict__ table)
{
    TF val0 = (val - offset)/delta;
    TF frac = val0 - int(val0);
    int idx = min(len-1, max(1, int(val0)+1));
    return table[idx-1] + frac * (table[idx] - table[idx-1]);
}


template<typename TF> __device__
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

    #pragma unroll
    for (int igpt=0; igpt<band_gpt; ++igpt)
    {
        k[igpt] = fminor[0] * kin[igpt + (j0-1)*ngpt + (jtemp-1)*neta*ngpt] +
                  fminor[1] * kin[igpt +  j0   *ngpt + (jtemp-1)*neta*ngpt] +
                  fminor[2] * kin[igpt + (j1-1)*ngpt + jtemp    *neta*ngpt] +
                  fminor[3] * kin[igpt +  j1   *ngpt + jtemp    *neta*ngpt];
    }
}


template<typename TF> __device__
void interpolate3D_byflav_kernel(
        const TF* __restrict__ scaling,
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

    #pragma unroll
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


template<typename TF> __global__
void reorder12x21_kernel(
        const int ni, const int nj,
        const TF* __restrict__ arr_in, TF* __restrict__ arr_out)
{
    const int ii = blockIdx.x*blockDim.x + threadIdx.x;
    const int ij = blockIdx.y*blockDim.y + threadIdx.y;

    if ( (ii < ni) && (ij < nj) )
    {
        const int idx_out = ii + ij*ni;
        const int idx_in = ij + ii*nj;

        arr_out[idx_out] = arr_in[idx_in];
    }
}


template<typename TF> __global__
void reorder123x321_kernel(
        const int ni, const int nj, const int nk,
        const TF* __restrict__ arr_in, TF* __restrict__ arr_out)
{
    const int ii = blockIdx.x*blockDim.x + threadIdx.x;
    const int ij = blockIdx.y*blockDim.y + threadIdx.y;
    const int ik = blockIdx.z*blockDim.z + threadIdx.z;

    if ( (ii < ni) && (ij < nj) && (ik < nk))
    {
        const int idx_out = ii + ij*ni + ik*nj*ni;
        const int idx_in  = ik + ij*nk + ii*nj*nk;

        arr_out[idx_out] = arr_in[idx_in];
    }
}


template<typename TF> __global__
void zero_array_kernel(
        const int ni, const int nj, const int nk,
        TF* __restrict__ arr)
{
    const int ii = blockIdx.x*blockDim.x + threadIdx.x;
    const int ij = blockIdx.y*blockDim.y + threadIdx.y;
    const int ik = blockIdx.z*blockDim.z + threadIdx.z;
    if ( (ii < ni) && (ij < nj) && (ik < nk))
    {
        const int idx = ii + ij*ni + ik*nj*ni;
        arr[idx] = TF(0.);
    }
}


template<typename TF> __global__
void Planck_source_kernel(
        const int ncol, const int nlay, const int nband, const int ngpt,
        const int nflav, const int neta, const int npres, const int ntemp,
        const int nPlanckTemp,
        const TF* __restrict__ tlay, const TF* __restrict__ tlev,
        const TF* __restrict__ tsfc,
        const int sfc_lay,
        const TF* __restrict__ fmajor, const int* __restrict__ jeta,
        const BOOL_TYPE* __restrict__ tropo, const int* __restrict__ jtemp,
        const int* __restrict__ jpress, const int* __restrict__ gpoint_bands,
        const int* __restrict__ band_lims_gpt, const TF* __restrict__ pfracin,
        const TF temp_ref_min, const TF totplnk_delta,
        const TF* __restrict__ totplnk, const int* __restrict__ gpoint_flavor,
        const TF* __restrict__ ones, const TF delta_Tsurf,
        TF* __restrict__ sfc_src, TF* __restrict__ lay_src,
        TF* __restrict__ lev_src_inc, TF* __restrict__ lev_src_dec,
        TF* __restrict__ sfc_src_jac, TF* __restrict__ pfrac)
{
    const int ilay = blockIdx.x*blockDim.x + threadIdx.x;
    const int ibnd = blockIdx.y*blockDim.y + threadIdx.y;
    const int icol = blockIdx.z*blockDim.z + threadIdx.z;

    if ( (icol < ncol) && (ilay < nlay) && (ibnd < nband))
    {
        const int idx_collay = icol + ilay * ncol;
        const int itropo = !tropo[idx_collay];
        const int gpt_start = band_lims_gpt[2*ibnd] - 1;
        const int gpt_end = band_lims_gpt[2*ibnd + 1];
        const int iflav = gpoint_flavor[itropo + 2 * gpt_start] - 1;
        const int idx_fcl3 = 2 * 2 * 2 * (iflav + icol*nflav + ilay*ncol*nflav);
        const int idx_fcl1 = 2 * (iflav + icol*nflav + ilay*ncol*nflav);

        const int j0 = jeta[idx_fcl1+0];
        const int j1 = jeta[idx_fcl1+1];
        const int npress1 = npres+1;

        const int jtemp_idx = jtemp[idx_collay];
        const int jpress_idx = jpress[idx_collay]+itropo;

        // compute layer source irradiances.
        const int idx_tmp = icol + ilay*ncol;
        const TF planck_function_lay = interpolate1D(tlay[idx_tmp], temp_ref_min, totplnk_delta, nPlanckTemp, &totplnk[ibnd * nPlanckTemp]);

        // compute level source irradiances.
        const int idx_tmp1 = icol + (ilay+1)*ncol;
        const int idx_tmp2 = icol + ilay*ncol;
        const TF planck_function_lev1 = interpolate1D(tlev[idx_tmp1], temp_ref_min, totplnk_delta, nPlanckTemp, &totplnk[ibnd * nPlanckTemp]);
        const TF planck_function_lev2 = interpolate1D(tlev[idx_tmp2], temp_ref_min, totplnk_delta, nPlanckTemp, &totplnk[ibnd * nPlanckTemp]);

        for (int igpt=gpt_start; igpt<gpt_end; ++igpt)
        {
            const int iigpt = igpt-gpt_start;
            const int idx = igpt + ilay*ngpt + icol*nlay*ngpt;
            const int idx_sfc = igpt + icol*ngpt;

            const TF pfrac_loc =
                  (fmajor[idx_fcl3+0] * pfracin[gpt_start + iigpt + (j0-1)*ngpt + (jpress_idx-1)*neta*ngpt + (jtemp_idx-1)*neta*ngpt*npress1] +
                   fmajor[idx_fcl3+1] * pfracin[gpt_start + iigpt +  j0   *ngpt + (jpress_idx-1)*neta*ngpt + (jtemp_idx-1)*neta*ngpt*npress1] +
                   fmajor[idx_fcl3+2] * pfracin[gpt_start + iigpt + (j0-1)*ngpt + jpress_idx*neta*ngpt     + (jtemp_idx-1)*neta*ngpt*npress1] +
                   fmajor[idx_fcl3+3] * pfracin[gpt_start + iigpt +  j0   *ngpt + jpress_idx*neta*ngpt     + (jtemp_idx-1)*neta*ngpt*npress1])
                + 
                  (fmajor[idx_fcl3+4] * pfracin[gpt_start + iigpt + (j1-1)*ngpt + (jpress_idx-1)*neta*ngpt + jtemp_idx*neta*ngpt*npress1] +
                   fmajor[idx_fcl3+5] * pfracin[gpt_start + iigpt +  j1   *ngpt + (jpress_idx-1)*neta*ngpt + jtemp_idx*neta*ngpt*npress1] +
                   fmajor[idx_fcl3+6] * pfracin[gpt_start + iigpt + (j1-1)*ngpt + jpress_idx*neta*ngpt     + jtemp_idx*neta*ngpt*npress1] +
                   fmajor[idx_fcl3+7] * pfracin[gpt_start + iigpt +  j1   *ngpt + jpress_idx*neta*ngpt     + jtemp_idx*neta*ngpt*npress1]);

            // Layer source
            lay_src[idx] = pfrac_loc * planck_function_lay;

            // Level source
            lev_src_inc[idx] = pfrac_loc * planck_function_lev1;
            lev_src_dec[idx] = pfrac_loc * planck_function_lev2;

            // Surface
            if (ilay == sfc_lay - 1) // Subtract one to correct for fortran indexing.
            {
                const TF planck_function_sfc1 = interpolate1D(
                        tsfc[icol], temp_ref_min, totplnk_delta, nPlanckTemp, &totplnk[ibnd * nPlanckTemp]);
                const TF planck_function_sfc2 = interpolate1D(
                        tsfc[icol] + delta_Tsurf, temp_ref_min, totplnk_delta, nPlanckTemp, &totplnk[ibnd * nPlanckTemp]);

                sfc_src[idx_sfc] = pfrac_loc * planck_function_sfc1;
                sfc_src_jac[idx_sfc] = pfrac_loc * (planck_function_sfc2 - planck_function_sfc1);
            }
        }
    }
}


template<typename TF> __global__
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
    const int iflav = blockIdx.x*blockDim.x + threadIdx.x;
    const int icol  = blockIdx.y*blockDim.y + threadIdx.y;
    const int ilay  = blockIdx.z*blockDim.z + threadIdx.z;

    if ( (icol < ncol) && (ilay < nlay) && (iflav < nflav) )
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

        const int gas1 = flavor[2*iflav];
        const int gas2 = flavor[2*iflav+1];

        for (int itemp=0; itemp<2; ++itemp)
        {
            const int vmr_base_idx = itropo + (jtemp[idx]+itemp-1) * (ngas+1) * 2;
            const int colmix_idx = itemp + 2*(iflav + nflav*icol + nflav*ncol*ilay);
            const int colgas1_idx = icol + ilay*ncol + gas1*nlay*ncol;
            const int colgas2_idx = icol + ilay*ncol + gas2*nlay*ncol;
            const TF ratio_eta_half = vmr_ref[vmr_base_idx + 2*gas1] /
                                      vmr_ref[vmr_base_idx + 2*gas2];
            col_mix[colmix_idx] = col_gas[colgas1_idx] + ratio_eta_half * col_gas[colgas2_idx];

            TF eta;
            if (col_mix[colmix_idx] > TF(2.)*tmin)
                eta = col_gas[colgas1_idx] / col_mix[colmix_idx];
            else
                eta = TF(0.5);

            const TF loceta = eta * TF(neta-1);
            jeta[colmix_idx] = min(int(loceta)+1, neta-1);
            const TF feta = fmod(loceta, TF(1.));
            const TF ftemp_term  = TF(1-itemp) + TF(2*itemp-1)*ftemp;

            // Compute interpolation fractions needed for minot species.
            const int fminor_idx = 2*(itemp + 2*(iflav + icol*nflav + ilay*ncol*nflav));
            fminor[fminor_idx] = (TF(1.0)-feta) * ftemp_term;
            fminor[fminor_idx+1] = feta * ftemp_term;

            // Compute interpolation fractions needed for major species.
            const int fmajor_idx = 2*2*(itemp + 2*(iflav + icol*nflav + ilay*ncol*nflav));
            fmajor[fmajor_idx] = (TF(1.0)-fpress) * fminor[fminor_idx];
            fmajor[fmajor_idx+1] = (TF(1.0)-fpress) * fminor[fminor_idx+1];
            fmajor[fmajor_idx+2] = fpress * fminor[fminor_idx];
            fmajor[fmajor_idx+3] = fpress * fminor[fminor_idx+1];
        }
    }
}


template<typename TF> __global__
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
    const int ibnd = (blockIdx.z * blockDim.z) + threadIdx.z;
    const int ilay = (blockIdx.y * blockDim.y) + threadIdx.y;
    const int icol = (blockIdx.x * blockDim.x) + threadIdx.x;

    if ( (icol < ncol) && (ilay < nlay) && (ibnd < nband) )
    {
        const int idx_collay = icol + ilay * ncol;
        const int itropo = !tropo[idx_collay];
        const int gpt_start = band_lims_gpt[2*ibnd] - 1;
        const int gpt_end = band_lims_gpt[2*ibnd + 1];
        const int iflav = gpoint_flavor[itropo + 2*gpt_start] - 1;
        const int idx_fcl3 = 2 * 2 * 2 * (iflav + icol*nflav + ilay*ncol*nflav);
        const int idx_fcl1 = 2 * (iflav + icol*nflav + ilay*ncol*nflav);
        const int idx_tau = gpt_start + ilay*ngpt + icol*nlay*ngpt;

        //major gases//
        interpolate3D_byflav_kernel(
                &col_mix[idx_fcl1], &fmajor[idx_fcl3],
                &kmajor[gpt_start], gpt_start, gpt_end,
                &jeta[idx_fcl1], jtemp[idx_collay],
                jpress[idx_collay]+itropo, ngpt, neta, npres+1,
                &tau_major[idx_tau]);

        #pragma unroll
        for (int igpt=gpt_start; igpt<gpt_end; ++igpt)
        {
            const int idx_out = igpt + ilay*ngpt + icol*nlay*ngpt;
            tau[idx_out] += tau_major[idx_out];
        }
    }
}


template<typename TF> __global__
void compute_tau_minor_absorption_kernel(
        const int ncol, const int nlay, const int ngpt,
        const int ngas, const int nflav, const int ntemp, const int neta,
        const int nscale,
        const int nminor,
        const int nminork,
        const int idx_h2o, const int idx_tropo,
        const int* __restrict__ gpoint_flavor,
        const TF* __restrict__ kminor,
        const int* __restrict__ minor_limits_gpt,
        const BOOL_TYPE* __restrict__ minor_scales_with_density,
        const BOOL_TYPE* __restrict__ scale_by_complement,
        const int* __restrict__ idx_minor,
        const int* __restrict__ idx_minor_scaling,
        const int* __restrict__ kminor_start,
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
    const int icol = blockIdx.x * blockDim.x + threadIdx.x;
    const int ilay = blockIdx.y * blockDim.y + threadIdx.y;

    const TF PaTohPa = 0.01;
    const int ncl = ncol * nlay;

    if ((icol < ncol) && (ilay < nlay))
    {
        const int idx_collay = icol + ilay*ncol;
        const int idx_collaywv = icol + ilay*ncol + idx_h2o*ncl;

        if (tropo[idx_collay] == idx_tropo)
        {
            for (int imnr = 0; imnr < nscale; ++imnr)
            {
                TF scaling = col_gas[idx_collay + idx_minor[imnr] * ncl];

                if (minor_scales_with_density[imnr])
                {
                    scaling *= PaTohPa * play[idx_collay] / tlay[idx_collay];

                    if (idx_minor_scaling[imnr] > 0)
                    {
                        TF vmr_fact = TF(1.) / col_gas[idx_collay];
                        TF dry_fact = TF(1.) / (TF(1.) + col_gas[idx_collaywv] * vmr_fact);

                        if (scale_by_complement[imnr])
                            scaling *= (TF(1.) - col_gas[idx_collay + idx_minor_scaling[imnr] * ncl] * vmr_fact * dry_fact);
                        else
                            scaling *= col_gas[idx_collay + idx_minor_scaling[imnr] * ncl] * vmr_fact * dry_fact;
                    }
                }

                const int gpt_start = minor_limits_gpt[2*imnr]-1;
                const int gpt_end = minor_limits_gpt[2*imnr+1];
                const int gpt_offs = 1-idx_tropo;
                const int iflav = gpoint_flavor[2*gpt_start + gpt_offs]-1;
                const int idx_fcl2 = 2 * 2 * (iflav + icol*nflav + ilay*ncol*nflav);
                const int idx_fcl1 = 2 * (iflav + icol*nflav + ilay*ncol*nflav);
                const int idx_tau = gpt_start + ilay*ngpt + icol*nlay*ngpt;

                interpolate2D_byflav_kernel(
                        &fminor[idx_fcl2],
                        &kminor[kminor_start[imnr]-1],
                        kminor_start[imnr]-1,
                        kminor_start[imnr]-1 + (gpt_end - gpt_start),
                        &tau_minor[idx_tau],
                        &jeta[idx_fcl1],
                        jtemp[idx_collay],
                        nminork, neta);

                #pragma unroll
                for (int igpt = gpt_start; igpt < gpt_end; ++igpt)
                {
                    const int idx_out = igpt + ilay*ngpt + icol*nlay*ngpt;
                    tau[idx_out] += tau_minor[idx_out] * scaling;
                }
            }
        }
    }
}


template<typename TF> __global__
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


template<typename TF> __global__
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
            ssa[idx_out] = TF(0.);
    }
}

