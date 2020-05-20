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

#ifndef RADIATION_SOLVER_H
#define RADIATION_SOLVER_H

#include "Array.h"
#include "Gas_concs.h"
#include "Gas_optics_rrtmgp.h"

template<typename TF>
class Radiation_solver
{
    public:
        Radiation_solver(const Gas_concs<TF>& gas_concs);
        void solve(
                const Gas_concs<TF>& gas_concs,
                const Array<TF,2>& p_lay, const Array<TF,2>& p_lev,
                const Array<TF,2>& t_lay, const Array<TF,2>& t_lev,
                const Array<TF,2>& col_dry,
                const Array<TF,1>& t_sfc, const Array<TF,2>& emis_sfc,
                Array<TF,2>& lw_flux_up, Array<TF,2>& lw_flux_dn, Array<TF,2>& lw_flux_net,
                Array<TF,3>& lw_bnd_flux_up, Array<TF,3>& lw_bnd_flux_dn, Array<TF,3>& lw_bnd_flux_net);

        int get_n_gpt() const { return this->kdist_lw->get_ngpt(); };

    private:
        std::unique_ptr<Gas_optics_rrtmgp<TF>> kdist_lw;
};
#endif
