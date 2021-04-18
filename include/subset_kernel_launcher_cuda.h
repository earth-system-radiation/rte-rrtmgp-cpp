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
#include "Gas_concs.h"

namespace subset_kernel_launcher_cuda
{
    template<typename TF>
    void get_from_subset(const int ncol, const int nbnd, const int ncol_in, const int col_s_in,
                  Array_gpu<TF,2>& var_full, const Array_gpu<TF,2>& var_sub);

    template<typename TF>
    void get_from_subset(const int ncol, const int nlay, const int ncol_in, const int col_s_in,
                  Array_gpu<TF,2>& var1_full, Array_gpu<TF,2>& var2_full, Array_gpu<TF,2>& var3_full,  Array_gpu<TF,2>& var4_full,
                  const Array_gpu<TF,2>& var1_sub, const Array_gpu<TF,2>& var2_sub, const Array_gpu<TF,2>& var3_sub, const Array_gpu<TF,2>& var4_sub);

    template<typename TF>
    void get_from_subset(const int ncol, const int nlay, const int ncol_in, const int col_s_in,
                  Array_gpu<TF,2>& var1_full, Array_gpu<TF,2>& var2_full, Array_gpu<TF,2>& var3_full,
                  const Array_gpu<TF,2>& var1_sub, const Array_gpu<TF,2>& var2_sub, const Array_gpu<TF,2>& var3_sub);

    template<typename TF>
    void get_from_subset(const int ncol, const int nlay, const int nbnd, const int ncol_in, const int col_s_in,
                  Array_gpu<TF,3>& var1_full, Array_gpu<TF,3>& var2_full, Array_gpu<TF,3>& var3_full,  Array_gpu<TF,3>& var4_full,
                  const Array_gpu<TF,3>& var1_sub, const Array_gpu<TF,3>& var2_sub, const Array_gpu<TF,3>& var3_sub, const Array_gpu<TF,3>& var4_sub);

    template<typename TF>
    void get_from_subset(const int ncol, const int nlay, const int nbnd, const int ncol_in, const int col_s_in,
                  Array_gpu<TF,3>& var1_full, Array_gpu<TF,3>& var2_full, Array_gpu<TF,3>& var3_full,
                  const Array_gpu<TF,3>& var1_sub, const Array_gpu<TF,3>& var2_sub, const Array_gpu<TF,3>& var3_sub);
}
#endif
