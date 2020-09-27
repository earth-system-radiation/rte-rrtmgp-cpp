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

#include "Gas_optics_rrtmgp.h"

template<typename TF> __global__
void compute_delta_plev(
        const int ncol, const int nlay,
        const TF* __restrict__ plev,
        TF* __restrict__ delta_plev)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int ilay = blockIdx.y*blockDim.y + threadIdx.y;

    if ( (icol < ncol) && (ilay < nlay) )
    {
        const int idx = icol + ilay*ncol;

        delta_plev[idx] = abs(plev[idx] - plev[idx + ncol]);

        // delta_plev({icol, ilay}) = std::abs(plev({icol, ilay}) - plev({icol, ilay+1}));
    }
}

template<typename TF> __global__
void compute_m_air(
        const int ncol, const int nlay,
        const TF* __restrict__ vmr_h2o,
        TF* __restrict__ m_air)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int ilay = blockIdx.y*blockDim.y + threadIdx.y;

    constexpr TF m_dry = 0.028964;
    constexpr TF m_h2o = 0.018016;

    if ( (icol < ncol) && (ilay < nlay) )
    {
        const int idx = icol + ilay*ncol;

        m_air[idx] = (m_dry + m_h2o * vmr_h2o[idx]) / (TF(1.) + vmr_h2o[idx]);

        // m_air({icol, ilay}) = (m_dry + m_h2o * vmr_h2o({icol, ilay})) / (1. + vmr_h2o({icol, ilay}));
    }
}

template<typename TF> __global__
void compute_col_dry(
        const int ncol, const int nlay,
        const TF* __restrict__ delta_plev, const TF* __restrict__ m_air, const TF* __restrict__ vmr_h2o,
        TF* __restrict__ col_dry)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int ilay = blockIdx.y*blockDim.y + threadIdx.y;

    constexpr TF g0 = 9.80665;
    constexpr TF avogad = 6.02214076e23;

    if ( (icol < ncol) && (ilay < nlay) )
    {
        const int idx = icol + ilay*ncol;

        col_dry[idx] = TF(10.) * delta_plev[idx] * avogad / (TF(1000.)*m_air[idx]*TF(100.)*g0);
        col_dry[idx] /= (TF(1.) + vmr_h2o[idx]);

        // col_dry({icol, ilay}) = TF(10.) * delta_plev({icol, ilay}) * avogad / (TF(1000.)*m_air({icol, ilay})*TF(100.)*g0);
        // col_dry({icol, ilay}) /= (TF(1.) + vmr_h2o({icol, ilay}));
    }
}

// Calculate the molecules of dry air.
template<typename TF>
void Gas_optics_rrtmgp<TF>::get_col_dry_gpu(
        Array_gpu<TF,2>& col_dry, const Array_gpu<TF,2>& vmr_h2o,
        const Array_gpu<TF,2>& plev)
{
    Array_gpu<TF,2> delta_plev({col_dry.dim(1), col_dry.dim(2)});
    Array_gpu<TF,2> m_air     ({col_dry.dim(1), col_dry.dim(2)});

    const int block_lay = 32;
    const int block_col = 32;

    const int nlay = col_dry.dim(2);
    const int ncol = col_dry.dim(1);

    const int grid_lay  = nlay/block_lay + (nlay%block_lay > 0);
    const int grid_col  = ncol/block_col + (ncol%block_col > 0);

    dim3 grid_gpu(grid_lay, grid_col);
    dim3 block_gpu(block_lay, block_col);

    compute_delta_plev<<<grid_gpu, block_gpu>>>(
            ncol, nlay,
            plev.ptr(),
            delta_plev.ptr());

    compute_m_air<<<grid_gpu, block_gpu>>>(
            ncol, nlay,
            vmr_h2o.ptr(),
            m_air.ptr());

    compute_col_dry<<<grid_gpu, block_gpu>>>(
            ncol, nlay,
            delta_plev.ptr(), m_air.ptr(), vmr_h2o.ptr(),
            col_dry.ptr());
}

#ifdef FLOAT_SINGLE_RRTMGP
template class Gas_optics_rrtmgp<float>;
#else
template class Gas_optics_rrtmgp<double>;
#endif
