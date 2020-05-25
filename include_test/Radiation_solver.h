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
class Radiation_solver_longwave
{
    public:
        Radiation_solver_longwave(const Gas_concs<TF>& gas_concs, const std::string& file_name);

        void solve(
                const bool sw_output_optical,
                const bool sw_output_bnd_fluxes,
                const Gas_concs<TF>& gas_concs,
                const Array<TF,2>& p_lay, const Array<TF,2>& p_lev,
                const Array<TF,2>& t_lay, const Array<TF,2>& t_lev,
                const Array<TF,2>& col_dry,
                const Array<TF,1>& t_sfc, const Array<TF,2>& emis_sfc,
                Array<TF,3>& tau, Array<TF,3>& lay_source,
                Array<TF,3>& lev_source_inc, Array<TF,3>& lev_source_dec, Array<TF,2>& sfc_source,
                Array<TF,2>& lw_flux_up, Array<TF,2>& lw_flux_dn, Array<TF,2>& lw_flux_net,
                Array<TF,3>& lw_bnd_flux_up, Array<TF,3>& lw_bnd_flux_dn, Array<TF,3>& lw_bnd_flux_net) const;

        int get_n_gpt() const { return this->kdist->get_ngpt(); };
        int get_n_bnd() const { return this->kdist->get_nband(); };

        Array<int,2> get_band_lims_gpoint() const
        { return this->kdist->get_band_lims_gpoint(); }

        Array<TF,2> get_band_lims_wavenumber() const
        { return this->kdist->get_band_lims_wavenumber(); }

    private:
        std::unique_ptr<Gas_optics_rrtmgp<TF>> kdist;
};

template<typename TF>
class Radiation_solver_shortwave
{
    public:
        Radiation_solver_shortwave(const Gas_concs<TF>& gas_concs, const std::string& file_name);

        void solve(
                const bool sw_output_optical,
                const bool sw_output_bnd_fluxes,
                const Gas_concs<TF>& gas_concs,
                const Array<TF,2>& p_lay, const Array<TF,2>& p_lev,
                const Array<TF,2>& t_lay, const Array<TF,2>& t_lev,
                const Array<TF,2>& col_dry,
                const Array<TF,2>& sfc_alb_dir, const Array<TF,2>& sfc_alb_dif,
                const Array<TF,1>& mu0, const TF tsi_scaling,
                Array<TF,3>& tau, Array<TF,3>& ssa, Array<TF,3>& g,
                Array<TF,2>& sw_flux_up, Array<TF,2>& sw_flux_dn, Array<TF,2>& sw_flux_dn_dir, Array<TF,2>& sw_flux_net,
                Array<TF,3>& sw_bnd_flux_up, Array<TF,3>& sw_bnd_flux_dn, Array<TF,3>& sw_bnd_flux_dn_dir, Array<TF,3>& sw_bnd_flux_net) const;

        int get_n_gpt() const { return this->kdist->get_ngpt(); };
        int get_n_bnd() const { return this->kdist->get_nband(); };

        Array<int,2> get_band_lims_gpoint() const
        { return this->kdist->get_band_lims_gpoint(); }

        Array<TF,2> get_band_lims_wavenumber() const
        { return this->kdist->get_band_lims_wavenumber(); }

    private:
        std::unique_ptr<Gas_optics_rrtmgp<TF>> kdist;
};
#endif
