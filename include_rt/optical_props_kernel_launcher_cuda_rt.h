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

#ifndef OPTICAL_PROPS_KERNEL_LAUNCHER_CUDA_RT_H
#define OPTICAL_PROPS_KERNEL_LAUNCHER_CUDA_RT_H

#include "Array.h"
#include "Types.h"

namespace optical_props_kernel_launcher_cuda_rt
{
    void increment_1scalar_by_1scalar(
            int ncol, int nlay,
            Array_gpu<Float,2>& tau_inout, const Array_gpu<Float,2>& tau_in);

    void increment_2stream_by_2stream(
            int ncol, int nlay,
            Array_gpu<Float,2>& tau_inout, Array_gpu<Float,2>& ssa_inout, Array_gpu<Float,2>& g_inout,
            const Array_gpu<Float,2>& tau_in, const Array_gpu<Float,2>& ssa_in, const Array_gpu<Float,2>& g_in);

    void delta_scale_2str_k(
            int ncol, int nlay, int ngpt,
            Array_gpu<Float,2>& tau_inout, Array_gpu<Float,2>& ssa_inout, Array_gpu<Float,2>& g_inout);
}
#endif
