/*
 * This file is part of a C++ interface to the Radiative Transfer for Energetics (RTE)
 * and Rapid Radiative Transfer Model for GCM applications Parallel (RRTMGP).
 *
 * The original code is found at https://github.com/RobertPincus/rte-rrtmgp.
 *
 * Contacts: Robert Pincus and Eli Mlawer
 * email: rrtmgp@aer.com
 *
 * Copyright 2015-2020,  Atmospheric and Environmental Research and
 * Regents of the University of Colorado.  All right reserved.
 *
 * This C++ interface can be downloaded from https://github.com/microhh/rte-rrtmgp-cpp
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

#ifndef RTE_KERNELS_CUDA_H
#define RTE_KERNELS_CUDA_H

#include "Array.h"
#include "Types.h"


namespace rte_kernel_launcher_cuda
{
    void apply_BC(const int ncol, const int nlay, const int ngpt, const Bool top_at_1,
                  const Array_gpu<Float,2>& inc_flux_dir, const Array_gpu<Float,1>& mu0, Array_gpu<Float,3>& gpt_flux_dir);
    void apply_BC(const int ncol, const int nlay, const int ngpt, const Bool top_at_1, Array_gpu<Float,3>& gpt_flux_dn);
    void apply_BC(const int ncol, const int nlay, const int ngpt, const Bool top_at_1, const Array_gpu<Float,2>& inc_flux_dif, Array_gpu<Float,3>& gpt_flux_dn);


    void sw_solver_2stream(
            const int ncol, const int nlay, const int ngpt, const Bool top_at_1,
            const Array_gpu<Float,3>& tau, const Array_gpu<Float,3>& ssa, const Array_gpu<Float,3>& g,
            const Array_gpu<Float,1>& mu0,
            const Array_gpu<Float,2>& sfc_alb_dir, const Array_gpu<Float,2>& sfc_alb_dif,
            const Array_gpu<Float,2>& inc_flux_dir,
            Array_gpu<Float,3>& flux_up, Array_gpu<Float,3>& flux_dn, Array_gpu<Float,3>& flux_dir,
            const Bool has_dif_bc, const Array_gpu<Float,2>& inc_flux_dif,
            const Bool do_broadband, Array_gpu<Float,3>& flux_up_loc, Array_gpu<Float,3>& flux_dn_loc, Array_gpu<Float,3>& flux_dir_loc,
            void* calling_class_ptr);


    void lw_solver_noscat_gaussquad(
            const int ncol, const int nlay, const int ngpt, const Bool top_at_1, const int nmus,
            const Array_gpu<Float,3>& secants, const Array_gpu<Float,2>& weights,
            const Array_gpu<Float,3>& tau, const Array_gpu<Float,3> lay_source,
            const Array_gpu<Float,3>& lev_source_inc, const Array_gpu<Float,3>& lev_source_dec,
            const Array_gpu<Float,2>& sfc_emis, const Array_gpu<Float,2>& sfc_src,
            const Array_gpu<Float,2>& inc_flux,
            Array_gpu<Float,3>& flux_up, Array_gpu<Float,3>& flux_dn,
            const Bool do_broadband, Array_gpu<Float,3>& flux_up_loc, Array_gpu<Float,3>& flux_dn_loc,
            const Bool do_jacobians, const Array_gpu<Float,2>& sfc_src_jac, Array_gpu<Float,3>& flux_up_jac,
            void* calling_class_ptr);


    void lw_secants_array(
            const int ncol, const int ngpt, const int n_quad_angs, const int max_gauss_pts,
            const Array_gpu<Float,2>& Gauss_Ds, Array_gpu<Float,3>& secants);
}
#endif
