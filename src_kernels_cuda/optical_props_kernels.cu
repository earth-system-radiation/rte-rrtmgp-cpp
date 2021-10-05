/*
 * This file is part of a C++ interface to the Radiative Transfer for Energetics (RTE)
 * and Rapid Radiative Transfer Model for GCM applications Parallel (RRTMGP).
 *
 * The original code is found at https://github.com/earth-system-radiation/rte-rrtmgp.
 *
 * Contacts: Robert Pincus and Eli Mlawer
 * email: rrtmgp@aer.com
 *
 * Copyright 2015-2020,  Atmospheric and Environmental Research and
 * Regents of the University of Colorado.  All right reserved.
 *
 * This C++ interface can be downloaded from https://github.com/earth-system-radiation/rte-rrtmgp-cpp
 *
 * Contact: Chiel van Heerwaarden
 * email: chiel.vanheerwaarden@wur.nl
 *
 * Copyright 2020, Wageningen University & Research.
 *
 * Use and duplication is permitted under the terms of the
 * BSD 3-clause license, see http://opensource.org/licenses/BSD-3-Clause
 *
 */

#ifndef kernel_tuner
const int loop_unroll_factor_nbnd = 2;
#endif


template<typename TF> __global__
void increment_1scalar_by_1scalar_kernel(
            const int ncol, const int nlay, const int ngpt,
            TF* __restrict__ tau1, const TF* __restrict__ tau2)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int ilay = blockIdx.y*blockDim.y + threadIdx.y;
    const int igpt = blockIdx.z*blockDim.z + threadIdx.z;

    if ( (icol < ncol) && (ilay < nlay) && (igpt < ngpt) )
    {
        const int idx = icol + ilay*ncol + igpt*ncol*nlay;
        tau1[idx] = tau2[idx]+tau2[idx];
    }
}


template<typename TF> __global__
void increment_2stream_by_2stream_kernel(
            const int ncol, const int nlay, const int ngpt, const TF eps,
            TF* __restrict__ tau1, TF* __restrict__ ssa1, TF* __restrict__ g1,
            const TF* __restrict__ tau2, const TF* __restrict__ ssa2, const TF* __restrict__ g2)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int ilay = blockIdx.y*blockDim.y + threadIdx.y;
    const int igpt = blockIdx.z*blockDim.z + threadIdx.z;

    if ( (icol < ncol) && (ilay < nlay) && (igpt < ngpt) )
    {
        const int idx = icol + ilay*ncol + igpt*ncol*nlay;
        const TF tau12 = tau1[idx] + tau2[idx];
        const TF tauscat12 = tau1[idx] * ssa1[idx] + tau2[idx] * tau2[idx];
        g1[idx] = (tau1[idx] * ssa1[idx] * g1[idx] + tau2[idx] * ssa2[idx] * g2[idx]) / max(tauscat12, eps);
        ssa1[idx] = tauscat12 / max(eps, tau12);
        tau1[idx] = tau12;
    }
}


template<typename TF> __global__
void inc_1scalar_by_1scalar_bybnd_kernel(
            const int ncol, const int nlay, const int ngpt,
            TF* __restrict__ tau1, const TF* __restrict__ tau2,
            const int nbnd, const int* __restrict__ band_lims_gpt)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int ilay = blockIdx.y*blockDim.y + threadIdx.y;
    const int igpt = blockIdx.z*blockDim.z + threadIdx.z;

    if ( (icol < ncol) && (ilay < nlay) && (igpt < ngpt) )
    {
        #pragma unroll loop_unroll_factor_nbnd
        for (int ibnd=0; ibnd<nbnd; ++ibnd)
        {
            if ( ((igpt+1) >= band_lims_gpt[ibnd*2]) && ((igpt+1) <= band_lims_gpt[ibnd*2+1]) )
            {
                const int idx_gpt = icol + ilay*ncol + igpt*nlay*ncol;
                const int idx_bnd = icol + ilay*ncol + ibnd*nlay*ncol;

                tau1[idx_gpt] = tau1[idx_gpt] + tau2[idx_bnd];
            }
        }
    }
}


template<typename TF> __global__
void inc_2stream_by_2stream_bybnd_kernel(
            const int ncol, const int nlay, const int ngpt, const TF eps,
            TF* __restrict__ tau1, TF* __restrict__ ssa1, TF* __restrict__ g1,
            const TF* __restrict__ tau2, const TF* __restrict__ ssa2, const TF* __restrict__ g2,
            const int nbnd, const int* __restrict__ band_lims_gpt)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int ilay = blockIdx.y*blockDim.y + threadIdx.y;
    const int igpt = blockIdx.z*blockDim.z + threadIdx.z;

    if ( (icol < ncol) && (ilay < nlay) && (igpt < ngpt) )
    {
        const int idx_gpt = icol + ilay*ncol + igpt*nlay*ncol;

        #pragma unroll loop_unroll_factor_nbnd
        for (int ibnd=0; ibnd<nbnd; ++ibnd)
        {
            if ( ((igpt+1) >= band_lims_gpt[ibnd*2]) && ((igpt+1) <= band_lims_gpt[ibnd*2+1]) )
            {
                const int idx_bnd = icol + ilay*ncol + ibnd*nlay*ncol;

                const TF tau12 = tau1[idx_gpt] + tau2[idx_bnd];
                const TF tauscat12 = tau1[idx_gpt] * ssa1[idx_gpt] + tau2[idx_bnd] * ssa2[idx_bnd];

                g1[idx_gpt] = (tau1[idx_gpt] * ssa1[idx_gpt] * g1[idx_gpt] +
                               tau2[idx_bnd] * ssa2[idx_bnd] * g2[idx_bnd]) / max(tauscat12, eps);
                ssa1[idx_gpt] = tauscat12 / max(eps, tau12);
                tau1[idx_gpt] = tau12;
            }
        }
    }
}


template<typename TF> __global__
void delta_scale_2str_k_kernel(
            const int ncol, const int nlay, const int ngpt, const TF eps,
            TF* __restrict__ tau, TF* __restrict__ ssa, TF* __restrict__ g)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int ilay = blockIdx.y*blockDim.y + threadIdx.y;
    const int igpt = blockIdx.z*blockDim.z + threadIdx.z;

    if ( (icol < ncol) && (ilay < nlay) && (igpt < ngpt) )
    {
        const int idx = icol + ilay*ncol + igpt*nlay*ncol;
        const TF f = g[idx] * g[idx];
        const TF wf = ssa[idx] * f;
        tau[idx] *= (TF(1.) - wf);
        ssa[idx] = (ssa[idx] - wf) / max(eps,(TF(1.)-wf));
        g[idx] = (g[idx] - f) / max(eps,(TF(1.)-f));

    }
}
