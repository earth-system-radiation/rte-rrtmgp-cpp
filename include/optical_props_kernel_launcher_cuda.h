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

#ifndef OPTICAL_PROPS_KERNEL_LAUNCHER_CUDA_H
#define OPTICAL_PROPS_KERNEL_LAUNCHER_CUDA_H

#include "Array.h"
#include "Types.h"

namespace optical_props_kernel_launcher_cuda
{
    template<typename TF> void increment_1scalar_by_1scalar(
            int ncol, int nlay, int ngpt,
            Array_gpu<TF,3>& tau_inout, const Array_gpu<TF,3>& tau_in);

    template<typename TF> void increment_2stream_by_2stream(
            int ncol, int nlay, int ngpt,
            Array_gpu<TF,3>& tau_inout, Array_gpu<TF,3>& ssa_inout, Array_gpu<TF,3>& g_inout,
            const Array_gpu<TF,3>& tau_in, const Array_gpu<TF,3>& ssa_in, const Array_gpu<TF,3>& g_in);

    template<typename TF> void inc_1scalar_by_1scalar_bybnd(
            int ncol, int nlay, int ngpt,
            Array_gpu<TF,3>& tau_inout, const Array_gpu<TF,3>& tau_in,
            int nbnd, const Array_gpu<int,2>& band_lims_gpoint);

    template<typename TF> void inc_2stream_by_2stream_bybnd(
            int ncol, int nlay, int ngpt,
            Array_gpu<TF,3>& tau_inout, Array_gpu<TF,3>& ssa_inout, Array_gpu<TF,3>& g_inout,
            const Array_gpu<TF,3>& tau_in, const Array_gpu<TF,3>& ssa_in, const Array_gpu<TF,3>& g_in,
            int nbnd, const Array_gpu<int,2>& band_lims_gpoint);

    template<typename TF> void delta_scale_2str_k(
            int ncol, int nlay, int ngpt,
            Array_gpu<TF,3>& tau_inout, Array_gpu<TF,3>& ssa_inout, Array_gpu<TF,3>& g_inout);
}
#endif
