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

#ifndef RRTMGP_KERNELS_CUDA_H
#define RRTMGP_KERNELS_CUDA_H

#include "Array.h"
#include "define_bool.h"
#include "Gas_concs.h"

namespace rrtmgp_kernel_launcher_cuda
{
    template<typename TF>
    void fill_gases(
            const int ncol, const int nlay, const int ngas, 
            Array<TF,3>& vmr_out,
            Array<TF,3>& col_gas, const Array<TF,2>& col_dry,
            const Gas_concs<TF>& gas_desc, const Array<std::string,1>& gas_names);

    template<typename TF>
    void reorder123x321(const int ni, const int nj, const int nk, Array<TF,3>& arr_in, Array<TF,3>& arr_out);

    template<typename TF>
    void reorder12x21(const int ni, const int nj, Array<TF,3>& arr_in, Array<TF,3>& arr_out);

    template<typename TF>
    void zero_array(const int ni, const int nj, const int nk, Array<TF,3>& arr);

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
            Array<int,2>& jpress);

    template<typename TF>
    void combine_and_reorder_2str(
            const int ncol, const int nlay, const int ngpt,
            const Array<TF,3>& tau_local, const Array<TF,3>& tau_rayleigh,
            Array<TF,3>& tau, Array<TF,3>& ssa, Array<TF,3>& g);
    
    template<typename TF>
    void compute_tau_rayleigh(
            const int ncol, const int nlay, const int nband, const int ngpt,
            const int ngas, const int nflav, const int neta, const int npres, const int ntemp,
            const Array<int, 2>& gpoint_flavor,
            const Array<int, 2>& band_lims_gpt,
            const Array<TF,4>& krayl,
            int idx_h2o, const Array<TF,2>& col_dry, const Array<TF,3>& col_gas,
            const Array<TF,5>& fminor, const Array<int,4>& jeta,
            const Array<BOOL_TYPE,2>& tropo, const Array<int,2>& jtemp,
            Array<TF,3>& tau_rayleigh);

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
            const Array<int,2>& jpress, Array<TF,3>& tau);
}
#endif
