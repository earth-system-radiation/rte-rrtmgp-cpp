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

#include "Optical_props.h"
#include "Array.h"
#include "rrtmgp_kernels.h"

namespace
{
    template<typename TF>__global__
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

    template<typename TF>__global__
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

    template<typename TF>__global__
    void inc_1scalar_by_1scalar_bybnd_kernel(
                const int ncol, const int nlay, const int ngpt,
                TF* __restrict__ tau1, const TF* __restrict__ tau2,
                const int nbnd, const int* __restrict__ band_lims_gpt)
    {
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;
        const int ilay = blockIdx.y*blockDim.y + threadIdx.y;
        const int ibnd = blockIdx.z*blockDim.z + threadIdx.z;
        if ( (icol < ncol) && (ilay < nlay) && (ibnd < nbnd) )
        {
            const int gpt_start = band_lims_gpt[ibnd*2]-1;
            const int gpt_end = band_lims_gpt[ibnd*2+1];
            for (int igpt = gpt_start; igpt < gpt_end; ++igpt)
            {
                const int idx_gpt = icol + ilay*ncol + igpt*nlay*ncol;
                const int idx_bnd = icol + ilay*ncol + ibnd*nlay*ncol;
                tau1[idx_gpt] = tau1[idx_gpt] + tau2[idx_bnd];
            }
        }
    }

    template<typename TF>__global__
    void inc_2stream_by_2stream_bybnd_kernel(
                const int ncol, const int nlay, const int ngpt, const TF eps,
                TF* __restrict__ tau1, TF* __restrict__ ssa1, TF* __restrict__ g1,
                const TF* __restrict__ tau2, const TF* __restrict__ ssa2, const TF* __restrict__ g2,
                const int nbnd, const int* __restrict__ band_lims_gpt)
    {
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;
        const int ilay = blockIdx.y*blockDim.y + threadIdx.y;
        const int ibnd = blockIdx.z*blockDim.z + threadIdx.z;
        if ( (icol < ncol) && (ilay < nlay) && (ibnd < nbnd) )
        {
            const int gpt_start = band_lims_gpt[ibnd*2]-1;
            const int gpt_end = band_lims_gpt[ibnd*2+1];
            for (int igpt = gpt_start; igpt < gpt_end; ++igpt)
            {
                const int idx_gpt = icol + ilay*ncol + igpt*nlay*ncol;
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

    template<typename TF>__global__
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
}


// Optical properties per gpoint.
template<typename TF>
Optical_props_gpu<TF>::Optical_props_gpu(
        const Array<TF,2>& band_lims_wvn,
        const Array<int,2>& band_lims_gpt)
{
    Array<int,2> band_lims_gpt_lcl(band_lims_gpt);
    Array_gpu<int,2> band_lims_gpt_lcl_gpu(band_lims_gpt);

    this->band2gpt = band_lims_gpt_lcl;
    this->band2gpt_gpu = this->band2gpt;
    this->band_lims_wvn = band_lims_wvn;

    // Make a map between g-points and bands.
    this->gpt2band.set_dims({band_lims_gpt_lcl.max()});
    for (int iband=1; iband<=band_lims_gpt_lcl.dim(2); ++iband)
    {
        for (int i=band_lims_gpt_lcl({1,iband}); i<=band_lims_gpt_lcl({2,iband}); ++i)
            this->gpt2band({i}) =  iband;
    }
    this->gpt2band_gpu = this->gpt2band;
}


// Optical properties per band.
template<typename TF>
Optical_props_gpu<TF>::Optical_props_gpu(
        const Array<TF,2>& band_lims_wvn)
{
    Array<int,2> band_lims_gpt_lcl({2, band_lims_wvn.dim(2)});

    for (int iband=1; iband<=band_lims_wvn.dim(2); ++iband)
    {
        band_lims_gpt_lcl({1, iband}) = iband;
        band_lims_gpt_lcl({2, iband}) = iband;
    }

    this->band2gpt = band_lims_gpt_lcl;
    this->band2gpt_gpu = this->band2gpt;
    this->band_lims_wvn = band_lims_wvn;

    // Make a map between g-points and bands.
    this->gpt2band.set_dims({band_lims_gpt_lcl.max()});
    for (int iband=1; iband<=band_lims_gpt_lcl.dim(2); ++iband)
    {
        for (int i=band_lims_gpt_lcl({1,iband}); i<=band_lims_gpt_lcl({2,iband}); ++i)
            this->gpt2band({i}) =  iband;
    }
    this->gpt2band_gpu = this->gpt2band;
}


template<typename TF>
Optical_props_1scl_gpu<TF>::Optical_props_1scl_gpu(
        const int ncol,
        const int nlay,
        const Optical_props_gpu<TF>& optical_props_gpu) :
    Optical_props_arry_gpu<TF>(optical_props_gpu),
    tau({ncol, nlay, this->get_ngpt()})
{}

//template<typename TF>
//void Optical_props_1scl_gpu<TF>::set_subset(
//        const std::unique_ptr<Optical_props_arry_gpu<TF>>& optical_props_gpu_sub,
//        const int col_s, const int col_e)
//{
//    for (int igpt=1; igpt<=tau.dim(3); ++igpt)
//        for (int ilay=1; ilay<=tau.dim(2); ++ilay)
//            for (int icol=col_s; icol<=col_e; ++icol)
//                tau.copy({icol, ilay, igpt}, optical_props_gpu_sub->get_tau(), {icol-col_s+1, ilay, igpt});
//}
//
//template<typename TF>
//void Optical_props_1scl_gpu<TF>::get_subset(
//        const std::unique_ptr<Optical_props_arry_gpu<TF>>& optical_props_gpu_sub,
//        const int col_s, const int col_e)
//{
//    for (int igpt=1; igpt<=tau.dim(3); ++igpt)
//        for (int ilay=1; ilay<=tau.dim(2); ++ilay)
//            for (int icol=col_s; icol<=col_e; ++icol)
//                tau.copy({icol-col_s+1, ilay, igpt}, optical_props_gpu_sub->get_tau(), {icol, ilay, igpt});
//}

template<typename TF>
Optical_props_2str_gpu<TF>::Optical_props_2str_gpu(
        const int ncol,
        const int nlay,
        const Optical_props_gpu<TF>& optical_props_gpu) :
    Optical_props_arry_gpu<TF>(optical_props_gpu),
    tau({ncol, nlay, this->get_ngpt()}),
    ssa({ncol, nlay, this->get_ngpt()}),
    g  ({ncol, nlay, this->get_ngpt()})
{}

//template<typename TF>
//void Optical_props_2str_gpu<TF>::set_subset(
//        const std::unique_ptr<Optical_props_arry_gpu<TF>>& optical_props_gpu_sub,
//        const int col_s, const int col_e)
//{
//    for (int igpt=1; igpt<=tau.dim(3); ++igpt)
//        for (int ilay=1; ilay<=tau.dim(2); ++ilay)
//            for (int icol=col_s; icol<=col_e; ++icol)
//            {
//                tau.copy({icol, ilay, igpt}, optical_props_gpu_sub->get_tau(), {icol-col_s+1, ilay, igpt});
//                tau.copy({icol, ilay, igpt}, optical_props_gpu_sub->get_tau(), {icol-col_s+1, ilay, igpt});
//                ssa.copy({icol, ilay, igpt}, optical_props_gpu_sub->get_ssa(), {icol-col_s+1, ilay, igpt});
//            }
//}
//
//template<typename TF>
//void Optical_props_2str_gpu<TF>::get_subset(
//        const std::unique_ptr<Optical_props_arry_gpu<TF>>& optical_props_gpu_sub,
//        const int col_s, const int col_e)
//{
//    for (int igpt=1; igpt<=tau.dim(3); ++igpt)
//        for (int ilay=1; ilay<=tau.dim(2); ++ilay)
//            for (int icol=col_s; icol<=col_e; ++icol)
//            {
//                tau.copy({icol-col_s+1, ilay, igpt}, optical_props_gpu_sub->get_tau(), {icol, ilay, igpt});
//                ssa.copy({icol-col_s+1, ilay, igpt}, optical_props_gpu_sub->get_ssa(), {icol, ilay, igpt});
//                g  .copy({icol-col_s+1, ilay, igpt}, optical_props_gpu_sub->get_g  (), {icol, ilay, igpt});
//            }
//}

namespace rrtmgp_kernel_launcher_cuda
{
    template<typename TF> void increment_1scalar_by_1scalar(
            int ncol, int nlay, int ngpt,
            Array_gpu<TF,3>& tau_inout, const Array_gpu<TF,3>& tau_in)

    {
        const int block_gpt = 32;
        const int block_lay = 16;
        const int block_col = 1;

        const int grid_gpt  = ngpt/block_gpt + (ngpt%block_gpt > 0);
        const int grid_lay  = nlay/block_lay + (nlay%block_lay > 0);
        const int grid_col  = ncol/block_col + (ncol%block_col > 0);

        dim3 grid_gpu(grid_col, grid_lay, grid_gpt);
        dim3 block_gpu(block_col, block_lay, block_gpt);

        increment_1scalar_by_1scalar_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, ngpt,
                tau_inout.ptr(), tau_in.ptr());
    }

    template<typename TF> void increment_2stream_by_2stream(
            int ncol, int nlay, int ngpt,
            Array_gpu<TF,3>& tau_inout, Array_gpu<TF,3>& ssa_inout, Array_gpu<TF,3>& g_inout,
            const Array_gpu<TF,3>& tau_in, const Array_gpu<TF,3>& ssa_in, const Array_gpu<TF,3>& g_in)
    {
        const int block_gpt = 32;
        const int block_lay = 16;
        const int block_col = 1;

        const int grid_gpt  = ngpt/block_gpt + (ngpt%block_gpt > 0);
        const int grid_lay  = nlay/block_lay + (nlay%block_lay > 0);
        const int grid_col  = ncol/block_col + (ncol%block_col > 0);

        dim3 grid_gpu(grid_col, grid_lay, grid_gpt);
        dim3 block_gpu(block_col, block_lay, block_gpt);

        TF eps = std::numeric_limits<TF>::epsilon();
        increment_2stream_by_2stream_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, ngpt, eps,
                tau_inout.ptr(), ssa_inout.ptr(), g_inout.ptr(),
                tau_in.ptr(), ssa_in.ptr(), g_in.ptr());
    }

    template<typename TF> void inc_1scalar_by_1scalar_bybnd(
            int ncol, int nlay, int ngpt,
            Array_gpu<TF,3>& tau_inout, const Array_gpu<TF,3>& tau_in,
            int nbnd, const Array_gpu<int,2>& band_lims_gpoint)

    {
        const int block_bnd = 14;
        const int block_lay = 32;
        const int block_col = 1;

        const int grid_bnd  = nbnd/block_bnd + (nbnd%block_bnd > 0);
        const int grid_lay  = nlay/block_lay + (nlay%block_lay > 0);
        const int grid_col  = ncol/block_col + (ncol%block_col > 0);

        dim3 grid_gpu(grid_col, grid_lay, grid_bnd);
        dim3 block_gpu(block_col, block_lay, block_bnd);
        inc_1scalar_by_1scalar_bybnd_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, ngpt,
                tau_inout.ptr(), tau_in.ptr(),
                nbnd, band_lims_gpoint.ptr());
    }

    template<typename TF> void inc_2stream_by_2stream_bybnd(
            int ncol, int nlay, int ngpt,
            Array_gpu<TF,3>& tau_inout, Array_gpu<TF,3>& ssa_inout, Array_gpu<TF,3>& g_inout,
            const Array_gpu<TF,3>& tau_in, const Array_gpu<TF,3>& ssa_in, const Array_gpu<TF,3>& g_in,
            int nbnd, const Array_gpu<int,2>& band_lims_gpoint)

    {
        const int block_bnd = 14;
        const int block_lay = 32;
        const int block_col = 1;

        const int grid_bnd  = nbnd/block_bnd + (nbnd%block_bnd > 0);
        const int grid_lay  = nlay/block_lay + (nlay%block_lay > 0);
        const int grid_col  = ncol/block_col + (ncol%block_col > 0);

        dim3 grid_gpu(grid_col, grid_lay, grid_bnd);
        dim3 block_gpu(block_col, block_lay, block_bnd);
        TF eps = std::numeric_limits<TF>::epsilon();
        inc_2stream_by_2stream_bybnd_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, ngpt, eps,
                tau_inout.ptr(), ssa_inout.ptr(), g_inout.ptr(),
                tau_in.ptr(), ssa_in.ptr(), g_in.ptr(),
                nbnd, band_lims_gpoint.ptr());
    }

    template<typename TF> void delta_scale_2str_k(
            int ncol, int nlay, int ngpt,
            Array_gpu<TF,3>& tau_inout, Array_gpu<TF,3>& ssa_inout, Array_gpu<TF,3>& g_inout)
    {
        const int block_gpt = 32;
        const int block_lay = 16;
        const int block_col = 1;

        const int grid_gpt  = ngpt/block_gpt + (ngpt%block_gpt > 0);
        const int grid_lay  = nlay/block_lay + (nlay%block_lay > 0);
        const int grid_col  = ncol/block_col + (ncol%block_col > 0);

        dim3 grid_gpu(grid_col, grid_lay, grid_gpt);
        dim3 block_gpu(block_col, block_lay, block_gpt);
        TF eps = std::numeric_limits<TF>::epsilon();
        delta_scale_2str_k_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, ngpt, eps,
                tau_inout.ptr(), ssa_inout.ptr(), g_inout.ptr());
    }
}

template<typename TF>
void Optical_props_2str_gpu<TF>::delta_scale(const Array_gpu<TF,3>& forward_frac)
{
    const int ncol = this->get_ncol();
    const int nlay = this->get_nlay();
    const int ngpt = this->get_ngpt();

    rrtmgp_kernel_launcher_cuda::delta_scale_2str_k(
            ncol, nlay, ngpt,
            this->get_tau(), this->get_ssa(), this->get_g());
}

template<typename TF>
void add_to(Optical_props_1scl_gpu<TF>& op_inout, const Optical_props_1scl_gpu<TF>& op_in)
{
    const int ncol = op_inout.get_ncol();
    const int nlay = op_inout.get_nlay();
    const int ngpt = op_inout.get_ngpt();

    if (ngpt == op_in.get_ngpt())
    {
        rrtmgp_kernel_launcher_cuda::increment_1scalar_by_1scalar(
                ncol, nlay, ngpt,
                op_inout.get_tau(), op_in.get_tau());
    }
    else
    {
        if (op_in.get_ngpt() != op_inout.get_nband())
            throw std::runtime_error("Cannot add optical properties with incompatible band - gpoint combination");

        rrtmgp_kernel_launcher_cuda::inc_1scalar_by_1scalar_bybnd(
                ncol, nlay, ngpt,
                op_inout.get_tau(), op_in.get_tau(),
                op_inout.get_nband(), op_inout.get_band_lims_gpoint());
    }
}

template<typename TF>
void add_to(Optical_props_2str_gpu<TF>& op_inout, const Optical_props_2str_gpu<TF>& op_in)
{
    const int ncol = op_inout.get_ncol();
    const int nlay = op_inout.get_nlay();
    const int ngpt = op_inout.get_ngpt();

    if (ngpt == op_in.get_ngpt())
    {
        rrtmgp_kernel_launcher_cuda::increment_2stream_by_2stream(
                ncol, nlay, ngpt,
                op_inout.get_tau(), op_inout.get_ssa(), op_inout.get_g(),
                op_in   .get_tau(), op_in   .get_ssa(), op_in   .get_g());
    }
    else
    {
        if (op_in.get_ngpt() != op_inout.get_nband())
            throw std::runtime_error("Cannot add optical properties with incompatible band - gpoint combination");

        rrtmgp_kernel_launcher_cuda::inc_2stream_by_2stream_bybnd(
                ncol, nlay, ngpt,
                op_inout.get_tau(), op_inout.get_ssa(), op_inout.get_g(),
                op_in   .get_tau(), op_in   .get_ssa(), op_in   .get_g(),
                op_inout.get_nband(), op_inout.get_band_lims_gpoint());
    }
}


#ifdef RTE_RRTMGP_SINGLE_PRECISION
template class Optical_props_gpu<float>;
template class Optical_props_1scl_gpu<float>;
template class Optical_props_2str_gpu<float>;
template void add_to(Optical_props_2str_gpu<float>&, const Optical_props_2str_gpu<float>&);
template void add_to(Optical_props_1scl_gpu<float>&, const Optical_props_1scl_gpu<float>&);
#else
template class Optical_props_gpu<double>;
template class Optical_props_1scl_gpu<double>;
template class Optical_props_2str_gpu<double>;
template void add_to(Optical_props_2str_gpu<double>&, const Optical_props_2str_gpu<double>&);
template void add_to(Optical_props_1scl_gpu<double>&, const Optical_props_1scl_gpu<double>&);
#endif
