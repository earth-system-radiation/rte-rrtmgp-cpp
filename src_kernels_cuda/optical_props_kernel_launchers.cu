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

#include <limits>

#include "Array.h"
#include "Types.h"
#include "tuner.h"

namespace
{
    #include "optical_props_kernels.cu"
}

namespace optical_props_kernel_launcher_cuda
{
    template<typename TF> void increment_1scalar_by_1scalar(
            int ncol, int nlay, int ngpt,
            Array_gpu<TF,3>& tau_inout, const Array_gpu<TF,3>& tau_in)

    {
        const int block_gpt = 32;
        const int block_lay = 16;
        const int block_col = 1;

        const int grid_gpt = ngpt/block_gpt + (ngpt%block_gpt > 0);
        const int grid_lay = nlay/block_lay + (nlay%block_lay > 0);
        const int grid_col = ncol/block_col + (ncol%block_col > 0);

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

        const int grid_gpt = ngpt/block_gpt + (ngpt%block_gpt > 0);
        const int grid_lay = nlay/block_lay + (nlay%block_lay > 0);
        const int grid_col = ncol/block_col + (ncol%block_col > 0);

        dim3 grid_gpu(grid_col, grid_lay, grid_gpt);
        dim3 block_gpu(block_col, block_lay, block_gpt);

        TF eps = std::numeric_limits<TF>::min() * TF(3.);

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
        const int block_col = 16;
        const int block_lay = 4;
        const int block_gpt = 1;

        const int grid_col = ncol/block_col + (ncol%block_col > 0);
        const int grid_lay = nlay/block_lay + (nlay%block_lay > 0);
        const int grid_gpt = ngpt/block_gpt + (ngpt%block_gpt > 0);

        dim3 grid_gpu(grid_col, grid_lay, grid_gpt);
        dim3 block_gpu(block_col, block_lay, block_gpt);

        inc_1scalar_by_1scalar_bybnd_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, ngpt,
                tau_inout.ptr(), tau_in.ptr(),
                nbnd, band_lims_gpoint.ptr());
    }


    template<typename TF> void inc_2stream_by_2stream_bybnd(
            int ncol, int nlay, int ngpt,
            Array_gpu<TF,3>& tau_inout, Array_gpu<TF,3>& ssa_inout, Array_gpu<TF,3>& g_inout,
            const Array_gpu<TF,3>& tau_in, const Array_gpu<TF,3>& ssa_in, const Array_gpu<TF,3>& g_in,
            int nbnd, const Array_gpu<int,2>& band_lims_gpoint,
            Tuner_map& tunings)
    {
        dim3 grid(ncol, nlay, ngpt), block;

        TF eps = std::numeric_limits<TF>::min() * TF(3.);

        if (tunings.count("inc_2stream_by_2stream_bybnd_kernel") == 0)
        {
            std::tie(grid, block) = tune_kernel(
                    "inc_2stream_by_2stream_bybnd_kernel",
                    dim3(ncol, nlay, ngpt),
                    {1, 2, 3, 4, 8, 12, 16, 24},
                    {1, 2, 3, 4, 8, 12, 16, 24},
                    {1, 2, 3, 4, 8, 12, 16, 24},
                    inc_2stream_by_2stream_bybnd_kernel<TF>,
                    ncol, nlay, ngpt, eps,
                    tau_inout.ptr(), ssa_inout.ptr(), g_inout.ptr(),
                    tau_in.ptr(), ssa_in.ptr(), g_in.ptr(),
                    nbnd, band_lims_gpoint.ptr());

            tunings["inc_2stream_by_2stream_bybnd_kernel"].first = grid;
            tunings["inc_2stream_by_2stream_bybnd_kernel"].second = block;
        }
        else
        {
            grid = tunings["inc_2stream_by_2stream_bybnd_kernel"].first;
            block = tunings["inc_2stream_by_2stream_bybnd_kernel"].second;
        }

        inc_2stream_by_2stream_bybnd_kernel<<<grid, block>>>(
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

        TF eps = std::numeric_limits<TF>::min()*TF(3.);

        delta_scale_2str_k_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, ngpt, eps,
                tau_inout.ptr(), ssa_inout.ptr(), g_inout.ptr());
    }
}


#ifdef RTE_RRTMGP_SINGLE_PRECISION
template void optical_props_kernel_launcher_cuda::increment_1scalar_by_1scalar(
        int ncol, int nlay, int ngpt,
        Array_gpu<float,3>& tau_inout, const Array_gpu<float,3>& tau_in);

template void optical_props_kernel_launcher_cuda::increment_2stream_by_2stream(
        int ncol, int nlay, int ngpt,
        Array_gpu<float,3>& tau_inout, Array_gpu<float,3>& ssa_inout, Array_gpu<float,3>& g_inout,
        const Array_gpu<float,3>& tau_in, const Array_gpu<float,3>& ssa_in, const Array_gpu<float,3>& g_in);

template void optical_props_kernel_launcher_cuda::inc_1scalar_by_1scalar_bybnd(
        int ncol, int nlay, int ngpt,
        Array_gpu<float,3>& tau_inout, const Array_gpu<float,3>& tau_in,
        int nbnd, const Array_gpu<int,2>& band_lims_gpoint);

template void optical_props_kernel_launcher_cuda::inc_2stream_by_2stream_bybnd(
        int ncol, int nlay, int ngpt,
        Array_gpu<float,3>& tau_inout, Array_gpu<float,3>& ssa_inout, Array_gpu<float,3>& g_inout,
        const Array_gpu<float,3>& tau_in, const Array_gpu<float,3>& ssa_in, const Array_gpu<float,3>& g_in,
        int nbnd, const Array_gpu<int,2>& band_lims_gpoint,
        Tuner_map& tunings);


template void optical_props_kernel_launcher_cuda::delta_scale_2str_k(
        int ncol, int nlay, int ngpt,
        Array_gpu<float,3>& tau_inout, Array_gpu<float,3>& ssa_inout, Array_gpu<float,3>& g_inout);
#else
template void optical_props_kernel_launcher_cuda::increment_1scalar_by_1scalar(
        int ncol, int nlay, int ngpt,
        Array_gpu<double,3>& tau_inout, const Array_gpu<double,3>& tau_in);

template void optical_props_kernel_launcher_cuda::increment_2stream_by_2stream(
        int ncol, int nlay, int ngpt,
        Array_gpu<double,3>& tau_inout, Array_gpu<double,3>& ssa_inout, Array_gpu<double,3>& g_inout,
        const Array_gpu<double,3>& tau_in, const Array_gpu<double,3>& ssa_in, const Array_gpu<double,3>& g_in);

template void optical_props_kernel_launcher_cuda::inc_1scalar_by_1scalar_bybnd(
        int ncol, int nlay, int ngpt,
        Array_gpu<double,3>& tau_inout, const Array_gpu<double,3>& tau_in,
        int nbnd, const Array_gpu<int,2>& band_lims_gpoint);

template void optical_props_kernel_launcher_cuda::inc_2stream_by_2stream_bybnd(
        int ncol, int nlay, int ngpt,
        Array_gpu<double,3>& tau_inout, Array_gpu<double,3>& ssa_inout, Array_gpu<double,3>& g_inout,
        const Array_gpu<double,3>& tau_in, const Array_gpu<double,3>& ssa_in, const Array_gpu<double,3>& g_in,
        int nbnd, const Array_gpu<int,2>& band_lims_gpoint,
        Tuner_map& tunings);

template void optical_props_kernel_launcher_cuda::delta_scale_2str_k(
        int ncol, int nlay, int ngpt,
        Array_gpu<double,3>& tau_inout, Array_gpu<double,3>& ssa_inout, Array_gpu<double,3>& g_inout);
#endif
