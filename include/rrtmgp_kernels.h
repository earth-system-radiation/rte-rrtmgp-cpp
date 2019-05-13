#ifndef RRTMGP_KERNELS_H
#define RRTMGP_KERNELS_H

// Kernels of fluxes.
namespace rrtmgp_kernels
{
    extern "C" void sum_broadband(
            int* ncol, int* nlev, int* ngpt,
            double* spectral_flux, double* broadband_flux);

    extern "C" void net_broadband_precalc(
            int* ncol, int* nlev,
            double* broadband_flux_dn, double* broadband_flux_up,
            double* broadband_flux_net);

    extern "C" void sum_byband(
            int* ncol, int* nlev, int* ngpt, int* nbnd,
            int* band_lims,
            double* spectral_flux,
            double* byband_flux);

    extern "C" void net_byband_precalc(
            int* ncol, int* nlev, int* nbnd,
            double* byband_flux_dn, double* byband_flux_up,
            double* byband_flux_net);
}

// Kernels of gas optics.
namespace rrtmgp_kernels
{
    extern "C" void zero_array_3D(
            int* ni, int* nj, int* nk, double* array);

    extern "C" void zero_array_4D(
             int* ni, int* nj, int* nk, int* nl, double* array);

    extern "C" void interpolation(
                int* ncol, int* nlay,
                int* ngas, int* nflav, int* neta, int* npres, int* ntemp,
                int* flavor,
                double* press_ref_log,
                double* temp_ref,
                double* press_ref_log_delta,
                double* temp_ref_min,
                double* temp_ref_delta,
                double* press_ref_trop_log,
                double* vmr_ref,
                double* play,
                double* tlay,
                double* col_gas,
                int* jtemp,
                double* fmajor, double* fminor,
                double* col_mix,
                int* tropo,
                int* jeta,
                int* jpress);

    extern "C" void compute_tau_absorption(
            int* ncol, int* nlay, int* nband, int* ngpt,
            int* ngas, int* nflav, int* neta, int* npres, int* ntemp,
            int* nminorlower, int* nminorklower,
            int* nminorupper, int* nminorkupper,
            int* idx_h2o,
            int* gpoint_flavor,
            int* band_lims_gpt,
            double* kmajor,
            double* kminor_lower,
            double* kminor_upper,
            int* minor_limits_gpt_lower,
            int* minor_limits_gpt_upper,
            int* minor_scales_with_density_lower,
            int* minor_scales_with_density_upper,
            int* scale_by_complement_lower,
            int* scale_by_complement_upper,
            int* idx_minor_lower,
            int* idx_minor_upper,
            int* idx_minor_scaling_lower,
            int* idx_minor_scaling_upper,
            int* kminor_start_lower,
            int* kminor_start_upper,
            int* tropo,
            double* col_mix, double* fmajor, double* fminor,
            double* play, double* tlay, double* col_gas,
            int* jeta, int* jtemp, int* jpress,
            double* tau);

    extern "C" void reorder_123x321_kernel(
            int* dim1, int* dim2, int* dim3,
            double* array, double* array_out);

    extern "C" void combine_and_reorder_2str(
            int* ncol, int* nlay, int* ngpt,
            double* tau_local, double* tau_rayleigh,
            double* tau, double* ssa, double* g);

    extern "C" void compute_Planck_source(
            int* ncol, int* nlay, int* nbnd, int* ngpt,
            int* nflav, int* neta, int* npres, int* ntemp, int* nPlanckTemp,
            double* tlay, double* tlev, double* tsfc, int* sfc_lay,
            double* fmajor, int* jeta, int* tropo, int* jtemp, int* jpress,
            int* gpoint_bands, int* band_lims_gpt, double* pfracin, double* temp_ref_min,
            double* totplnk_delta, double* totplnk, int* gpoint_flavor,
            double* sfc_src, double* lay_src, double* lev_src, double* lev_source_dec);

    extern "C" void compute_tau_rayleigh(
            int* ncol, int* nlay, int* nband, int* ngpt,
            int* ngas, int* nflav, int* neta, int* npres, int* ntemp,
            int* gpoint_flavor,
            int* band_lims_gpt,
            double* krayl,
            int* idx_h2o, double* col_dry, double* col_gas,
            double* fminor, int* eta,
            int* tropo, int* jtemp,
            double* tau_rayleigh);
}

// Kernels of longwave solver.
namespace rrtmgp_kernels
{
    extern "C" void apply_BC_0(
            int* ncol, int* nlay, int* ngpt,
            int* top_at_1, double* gpt_flux_dn);

    extern "C" void lw_solver_noscat_GaussQuad(
            int* ncol, int* nlay, int* ngpt, int* top_at_1, int* n_quad_angs,
            double* gauss_Ds_subset, double* gauss_wts_subset,
            double* tau,
            double* lay_source, double* lev_source_inc, double* lev_source_dec,
            double* sfc_emis_gpt, double* sfc_source,
            double* gpt_flux_up, double* gpt_flux_dn);
}

// Kernels of shortwave solver.
namespace rrtmgp_kernels
{
    extern "C" void apply_BC_0(
            int* ncol, int* nlay, int* ngpt,
            int* top_at_1, double* gpt_flux_dn);

    extern "C" void apply_BC_factor(
            int* ncol, int* nlay, int* ngpt,
            int* top_at_1, double* inc_flux,
            double* factor, double* flux_dn);

    extern "C" void sw_solver_2stream(
            int* ncol, int* nlay, int* ngpt, int* top_at_1,
            double* tau,
            double* ssa,
            double* g,
            double* mu0,
            double* sfc_alb_dir_gpt, double* sfc_alb_dif_gpt,
            double* gpt_flux_up, double* gpt_flux_dn, double* gpt_flux_dir);
}

#endif
