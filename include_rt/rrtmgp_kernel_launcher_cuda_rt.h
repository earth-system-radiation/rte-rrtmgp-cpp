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

#ifndef RRTMGP_KERNELS_CUDA_RT_H
#define RRTMGP_KERNELS_CUDA_RT_H

#include "Array.h"
#include "Types.h"
#include "Gas_concs_rt.h"

namespace rrtmgp_kernel_launcher_cuda_rt
{
    
    void fill_gases(
            const int ncol, const int nlay, const int ngas,
            Array<Float,3>& vmr_out,
            Array<Float,3>& col_gas, const Array<Float,2>& col_dry,
            const Gas_concs& gas_desc, const Array<std::string,1>& gas_names);

    
    void reorder123x321(const int ni, const int nj, const int nk,
            const Array_gpu<Float,3>& arr_in, Array_gpu<Float,3>& arr_out);

    
    void reorder12x21(const int ni, const int nj, const Array_gpu<Float,2>& arr_in, Array_gpu<Float,2>& arr_out);

    
    void zero_array(const int ni, const int nj, const int nk, const int nn, Array_gpu<Float,4>& arr);

    
    void zero_array(const int ni, const int nj, const int nk, Array_gpu<Float,3>& arr);
    
    
    void zero_array(const int ni, const int nj, Array_gpu<Float,2>& arr);

    void zero_array(const int ni, Array_gpu<int,1>& arr);
    
    void interpolation(
            const int ncol, const int nlay,
            const int ngas, const int nflav, const int neta, const int npres, const int ntemp,
            const Array_gpu<int,2>& flavor,
            const Array_gpu<Float,1>& press_ref_log,
            const Array_gpu<Float,1>& temp_ref,
            Float press_ref_log_delta,
            Float temp_ref_min,
            Float temp_ref_delta,
            Float press_ref_trop_log,
            const Array_gpu<Float,3>& vmr_ref,
            const Array_gpu<Float,2>& play,
            const Array_gpu<Float,2>& tlay,
            Array_gpu<Float,3>& col_gas,
            Array_gpu<int,2>& jtemp,
            Array_gpu<Float,6>& fmajor, Array_gpu<Float,5>& fminor,
            Array_gpu<Float,4>& col_mix,
            Array_gpu<Bool,2>& tropo,
            Array_gpu<int,4>& jeta,
            Array_gpu<int,2>& jpress);

    
    void minor_scalings(
            const int ncol, const int nlay, const int nflav, const int ngpt, 
            const int nminorlower, const int nminorupper,
            const int idx_h2o,  
            const Array_gpu<int,2>& gpoint_flavor,
            const Array_gpu<int,2>& minor_limits_gpt_lower,
            const Array_gpu<int,2>& minor_limits_gpt_upper,
            const Array_gpu<Bool,1>& minor_scales_with_density_lower,
            const Array_gpu<Bool,1>& minor_scales_with_density_upper,
            const Array_gpu<Bool,1>& scale_by_complement_lower,
            const Array_gpu<Bool,1>& scale_by_complement_upper,
            const Array_gpu<int,1>& idx_minor_lower,
            const Array_gpu<int,1>& idx_minor_upper,
            const Array_gpu<int,1>& idx_minor_scaling_lower,
            const Array_gpu<int,1>& idx_minor_scaling_upper,
            const Array_gpu<Float,2>& play,
            const Array_gpu<Float,2>& tlay,
            const Array_gpu<Float,3>& col_gas,
            const Array_gpu<Bool,2>& tropo,
            Array_gpu<Float,3>& scalings_lower,
            Array_gpu<Float,3>& scalings_upper);
    
    
    void combine_abs_and_rayleigh(
            const int ncol, const int nlay,
            const Array_gpu<Float,2>& tau_local, const Array_gpu<Float,2>& tau_rayleigh,
            Array_gpu<Float,2>& tau, Array_gpu<Float,2>& ssa, Array_gpu<Float,2>& g);

    
    void compute_tau_rayleigh(
            const int ncol, const int nlay, const int nband, const int ngpt, const int igpt,
            const int ngas, const int nflav, const int neta, const int npres, const int ntemp,
            const Array_gpu<int, 2>& gpoint_flavor,
            const Array_gpu<int, 1>& gpoint_bands,
            const Array_gpu<int, 2>& band_lims_gpt,
            const Array_gpu<Float,4>& krayl,
            int idx_h2o, const Array_gpu<Float,2>& col_dry, const Array_gpu<Float,3>& col_gas,
            const Array_gpu<Float,5>& fminor, const Array_gpu<int,4>& jeta,
            const Array_gpu<Bool,2>& tropo, const Array_gpu<int,2>& jtemp,
            Array_gpu<Float,2>& tau_rayleigh);

    
    void compute_tau_absorption(
            const int ncol, const int nlay, const int nband, const int ngpt, const int igpt,
            const int ngas, const int nflav, const int neta, const int npres, const int ntemp,
            const int nminorlower, const int nminorklower,
            const int nminorupper, const int nminorkupper,
            const int idx_h2o,
            const Array_gpu<int,2>& gpoint_flavor,
            const Array_gpu<int,2>& band_lims_gpt,
            const Array_gpu<Float,4>& kmajor,
            const Array_gpu<Float,3>& kminor_lower,
            const Array_gpu<Float,3>& kminor_upper,
            const Array_gpu<int,2>& minor_limits_gpt_lower,
            const Array_gpu<int,2>& minor_limits_gpt_upper,
            const Array_gpu<int,2>& first_last_minor_lower,
            const Array_gpu<int,2>& first_last_minor_upper,
            const Array_gpu<Bool,1>& minor_scales_with_density_lower,
            const Array_gpu<Bool,1>& minor_scales_with_density_upper,
            const Array_gpu<Bool,1>& scale_by_complement_lower,
            const Array_gpu<Bool,1>& scale_by_complement_upper,
            const Array_gpu<int,1>& idx_minor_lower,
            const Array_gpu<int,1>& idx_minor_upper,
            const Array_gpu<int,1>& idx_minor_scaling_lower,
            const Array_gpu<int,1>& idx_minor_scaling_upper,
            const Array_gpu<int,1>& kminor_start_lower,
            const Array_gpu<int,1>& kminor_start_upper,
            const Array_gpu<Bool,2>& tropo,
            const Array_gpu<Float,4>& col_mix, const Array_gpu<Float,6>& fmajor,
            const Array_gpu<Float,5>& fminor, const Array_gpu<Float,2>& play,
            const Array_gpu<Float,2>& tlay, const Array_gpu<Float,3>& col_gas,
            const Array_gpu<int,4>& jeta, const Array_gpu<int,2>& jtemp,
            const Array_gpu<int,2>& jpress, 
            const Array_gpu<Float,3>& scalings_lower,
            const Array_gpu<Float,3>& scalings_upper,
            Array_gpu<Float,2>& tau);

    
    void Planck_source(
            const int ncol, const int nlay, const int nbnd, const int ngpt, const int igpt,
            const int nflav, const int neta, const int npres, const int ntemp,
            const int nPlanckTemp,
            const Array_gpu<Float,2>& tlay,
            const Array_gpu<Float,2>& tlev,
            const Array_gpu<Float,1>& tsfc,
            const int sfc_lay,
            const Array_gpu<Float,6>& fmajor,
            const Array_gpu<int,4>& jeta,
            const Array_gpu<Bool,2>& tropo,
            const Array_gpu<int,2>& jtemp,
            const Array_gpu<int,2>& jpress,
            const Array_gpu<int,1>& gpoint_bands,
            const Array_gpu<int,2>& band_lims_gpt,
            const Array_gpu<Float,4>& pfracin,
            const Float temp_ref_min, const Float totplnk_delta,
            const Array_gpu<Float,2>& totplnk,
            const Array_gpu<int,2>& gpoint_flavor,
            Array_gpu<Float,1>& sfc_src,
            Array_gpu<Float,2>& lay_src,
            Array_gpu<Float,2>& lev_src_inc,
            Array_gpu<Float,2>& lev_src_dec,
            Array_gpu<Float,1>& sfc_src_jac);
}
#endif
