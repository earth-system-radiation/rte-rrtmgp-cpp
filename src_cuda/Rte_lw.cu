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

#include "Rte_lw.h"
#include "Array.h"
#include "Optical_props.h"
#include "Source_functions.h"
#include "Fluxes.h"

#include "rrtmgp_kernels.h"
// CUDA TEST
#include "rte_kernel_launcher_cuda.h"
// END CUDA TEST

namespace
{
    template<typename TF>__global__
    void expand_and_transpose_kernel(
        const int ncol, const int nbnd, const int* __restrict__ limits,
        TF* __restrict__ arr_out, const TF* __restrict__ arr_in)
    {
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;
        const int ibnd = blockIdx.y*blockDim.y + threadIdx.y;
        if ( ( icol < ncol) && (ibnd < nbnd) )
        {
            const int gpt_start = limits[2*ibnd] - 1;
            const int gpt_end = limits[2*ibnd+1];

            for (int igpt=gpt_start; igpt<gpt_end; ++igpt)
            {
                const int idx_in  = ibnd + icol*nbnd;
                const int idx_out = icol + igpt*ncol;

                arr_out[idx_out] = arr_in[idx_in];
            }
        }
    }
}

//    template<typename TF>
//    void apply_BC(
//            int ncol, int nlay, int ngpt,
//            BOOL_TYPE top_at_1, Array<TF,3>& gpt_flux_dn)
//    {
//        rrtmgp_kernels::apply_BC_0(
//                &ncol, &nlay, &ngpt,
//                &top_at_1, gpt_flux_dn.ptr());
//    }
//
//    template<typename TF>
//    void apply_BC(
//            int ncol, int nlay, int ngpt,
//            BOOL_TYPE top_at_1, const Array<TF,2>& inc_flux,
//            Array<TF,3>& gpt_flux_dn)
//    {
//        rrtmgp_kernels::apply_BC_gpt(
//                &ncol, &nlay, &ngpt,
//                &top_at_1, const_cast<TF*>(inc_flux.ptr()), gpt_flux_dn.ptr());
//    }

template<typename TF>
void Rte_lw_gpu<TF>::rte_lw(
        const std::unique_ptr<Optical_props_arry_gpu<TF>>& optical_props,
        const BOOL_TYPE top_at_1,
        const Source_func_lw_gpu<TF>& sources,
        const Array_gpu<TF,2>& sfc_emis,
        const Array_gpu<TF,2>& inc_flux,
        Array_gpu<TF,3>& gpt_flux_up,
        Array_gpu<TF,3>& gpt_flux_dn,
        const int n_gauss_angles)
{
    const int max_gauss_pts = 4;
    const Array<TF,2> gauss_Ds(
            {      1.66,         0.,         0.,         0.,
             1.18350343, 2.81649655,         0.,         0.,
             1.09719858, 1.69338507, 4.70941630,         0.,
             1.06056257, 1.38282560, 2.40148179, 7.15513024},
            { max_gauss_pts, max_gauss_pts });

    const Array<TF,2> gauss_wts(
            {         0.5,           0.,           0.,           0.,
             0.3180413817, 0.1819586183,           0.,           0.,
             0.2009319137, 0.2292411064, 0.0698269799,           0.,
             0.1355069134, 0.2034645680, 0.1298475476, 0.0311809710},
            { max_gauss_pts, max_gauss_pts });

    const int ncol = optical_props->get_ncol();
    const int nlay = optical_props->get_nlay();
    const int ngpt = optical_props->get_ngpt();

    Array_gpu<TF,2> sfc_emis_gpt({ncol, ngpt});

    expand_and_transpose(optical_props, sfc_emis, sfc_emis_gpt);

    // Upper boundary condition.
    if (inc_flux.size() == 0)
        rte_kernel_launcher_cuda::apply_BC(ncol, nlay, ngpt, top_at_1, gpt_flux_dn);
    else
        rte_kernel_launcher_cuda::apply_BC(ncol, nlay, ngpt, top_at_1, inc_flux, gpt_flux_dn);

    // Run the radiative transfer solver.
    const int n_quad_angs = n_gauss_angles;

    Array_gpu<TF,2> gauss_Ds_subset = gauss_Ds.subset(
            {{ {1, n_quad_angs}, {n_quad_angs, n_quad_angs} }});
    Array_gpu<TF,2> gauss_wts_subset = gauss_wts.subset(
            {{ {1, n_quad_angs}, {n_quad_angs, n_quad_angs} }});

    // For now, just pass the arrays around.
    Array_gpu<TF,2> sfc_src_jac(sources.get_sfc_source().get_dims());
    Array_gpu<TF,3> gpt_flux_up_jac(gpt_flux_up.get_dims());

    rte_kernel_launcher_cuda::lw_solver_noscat_gaussquad(
            ncol, nlay, ngpt, top_at_1, n_quad_angs,
            gauss_Ds_subset, gauss_wts_subset,
            optical_props->get_tau(),
            sources.get_lay_source(),
            sources.get_lev_source_inc(), sources.get_lev_source_dec(),
            sfc_emis_gpt, sources.get_sfc_source(),
            gpt_flux_up, gpt_flux_dn,
            sfc_src_jac, gpt_flux_up_jac,
            rte_lw_map);

    // CvH: In the fortran code this call is here, I removed it for performance and flexibility.
    // fluxes->reduce(gpt_flux_up, gpt_flux_dn, optical_props, top_at_1);
}

template<typename TF>
void Rte_lw_gpu<TF>::expand_and_transpose(
        const std::unique_ptr<Optical_props_arry_gpu<TF>>& ops,
        const Array_gpu<TF,2> arr_in,
        Array_gpu<TF,2>& arr_out)
{
    const int ncol = arr_in.dim(2);
    const int nbnd = ops->get_nband();
    const int block_col = 16;
    const int block_bnd = 14;

    const int grid_col = ncol/block_col + (ncol%block_col > 0);
    const int grid_bnd = nbnd/block_bnd + (nbnd%block_bnd > 0);

    dim3 grid_gpu(grid_col, grid_bnd);
    dim3 block_gpu(block_col, block_bnd);

    Array_gpu<int,2> limits = ops->get_band_lims_gpoint_gpu();

    expand_and_transpose_kernel<<<grid_gpu, block_gpu>>>(
            ncol, nbnd, limits.ptr(), arr_out.ptr(), arr_in.ptr());
}

#ifdef RTE_RRTMGP_SINGLE_PRECISION
template class Rte_lw_gpu<float>;
#else
template class Rte_lw_gpu<double>;
#endif
