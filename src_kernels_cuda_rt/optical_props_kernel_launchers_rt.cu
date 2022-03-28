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


#include "Array.h"
#include "Types.h"
#include "tuner.h"



namespace
{
    #include "optical_props_kernels_rt.cu"
}

namespace optical_props_kernel_launcher_cuda_rt
{
    void increment_1scalar_by_1scalar(
            int ncol, int nlay,
            Array_gpu<Float,2>& tau_inout, const Array_gpu<Float,2>& tau_in)

    {
        const int block_lay = 16;
        const int block_col = 16;

        const int grid_lay = nlay/block_lay + (nlay%block_lay > 0);
        const int grid_col = ncol/block_col + (ncol%block_col > 0);

        dim3 grid_gpu(grid_col, grid_lay, 1);
        dim3 block_gpu(block_col, block_lay, 1);

        increment_1scalar_by_1scalar_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay,
                tau_inout.ptr(), tau_in.ptr());
    }


    void increment_2stream_by_2stream(
            int ncol, int nlay,
            Array_gpu<Float,2>& tau_inout, Array_gpu<Float,2>& ssa_inout, Array_gpu<Float,2>& g_inout,
            const Array_gpu<Float,2>& tau_in, const Array_gpu<Float,2>& ssa_in, const Array_gpu<Float,2>& g_in)
    {
        const int block_lay = 16;
        const int block_col = 16;

        const int grid_lay = nlay/block_lay + (nlay%block_lay > 0);
        const int grid_col = ncol/block_col + (ncol%block_col > 0);

        dim3 grid_gpu(grid_col, grid_lay, 1);
        dim3 block_gpu(block_col, block_lay, 1);

        Float eps = std::numeric_limits<Float>::epsilon();

        increment_2stream_by_2stream_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, eps,
                tau_inout.ptr(), ssa_inout.ptr(), g_inout.ptr(),
                tau_in.ptr(), ssa_in.ptr(), g_in.ptr());
    }

    void delta_scale_2str_k(
            int ncol, int nlay, int ngpt,
            Array_gpu<Float,2>& tau_inout, Array_gpu<Float,2>& ssa_inout, Array_gpu<Float,2>& g_inout)
    {
        const int block_lay = 16;
        const int block_col = 16;

        const int grid_lay  = nlay/block_lay + (nlay%block_lay > 0);
        const int grid_col  = ncol/block_col + (ncol%block_col > 0);

        dim3 grid_gpu(grid_col, grid_lay, 1);
        dim3 block_gpu(block_col, block_lay, 1);

        Float eps = std::numeric_limits<Float>::epsilon();

        delta_scale_2str_k_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, ngpt, eps,
                tau_inout.ptr(), ssa_inout.ptr(), g_inout.ptr());
    }
}

