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
#include "define_bool.h"
#include "Gas_concs.h"

namespace rte_kernel_launcher_cuda
{
    template<typename TF>
    void sw_solver_2stream(const int ncol, const int nlay, const int ngpt, const BOOL_TYPE top_at_1,
                           const Array<TF,3>& tau, const Array<TF,3>& ssa, const Array<TF,3>& g,
                           const Array<TF,1>& mu0, const Array<TF,2>& sfc_alb_dir, const Array<TF,2>& sfc_alb_dif,
                           Array<TF,3>& flux_up, Array<TF,3>& flux_dn, Array<TF,3>& flux_dir);
}
#endif
