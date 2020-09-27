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

// Calculate the molecules of dry air.
template<typename TF>
void Gas_optics_rrtmgp<TF>::get_col_dry_gpu(
        Array_gpu<TF,2>& col_dry, const Array_gpu<TF,2>& vmr_h2o,
        const Array_gpu<TF,2>& plev)
{
    // CvH: RRTMGP uses more accurate method based on latitude.
    constexpr TF g0 = 9.80665;

    constexpr TF avogad = 6.02214076e23;
    constexpr TF m_dry = 0.028964;
    constexpr TF m_h2o = 0.018016;

    Array_gpu<TF,2> delta_plev({col_dry.dim(1), col_dry.dim(2)});
    Array_gpu<TF,2> m_air     ({col_dry.dim(1), col_dry.dim(2)});

    // CvH: the code below should be replaced by a kernel
    /*
    for (int ilay=1; ilay<=col_dry.dim(2); ++ilay)
        for (int icol=1; icol<=col_dry.dim(1); ++icol)
            delta_plev({icol, ilay}) = std::abs(plev({icol, ilay}) - plev({icol, ilay+1}));

    for (int ilay=1; ilay<=col_dry.dim(2); ++ilay)
        for (int icol=1; icol<=col_dry.dim(1); ++icol)
            m_air({icol, ilay}) = (m_dry + m_h2o * vmr_h2o({icol, ilay})) / (1. + vmr_h2o({icol, ilay}));

    for (int ilay=1; ilay<=col_dry.dim(2); ++ilay)
        for (int icol=1; icol<=col_dry.dim(1); ++icol)
        {
            col_dry({icol, ilay}) = TF(10.) * delta_plev({icol, ilay}) * avogad / (TF(1000.)*m_air({icol, ilay})*TF(100.)*g0);
            col_dry({icol, ilay}) /= (TF(1.) + vmr_h2o({icol, ilay}));
        }
        */
}

#ifdef FLOAT_SINGLE_RRTMGP
template class Gas_optics_rrtmgp<float>;
#else
template class Gas_optics_rrtmgp<double>;
#endif
