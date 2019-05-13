#include <boost/algorithm/string.hpp>
#include <cmath>

#include "Netcdf_interface.h"
#include "Array.h"
#include "Gas_concs.h"
#include "Gas_optics.h"
#include "Source_functions.h"
#include "Fluxes.h"
#include "Rte_lw.h"
#include "Rte_sw.h"

namespace
{
    std::vector<std::string> get_variable_string(
            const std::string& var_name,
            std::vector<int> i_count,
            Netcdf_handle& input_nc,
            const int string_len,
            bool trim=true)
    {
        // Multiply all elements in i_count.
        int total_count = std::accumulate(i_count.begin(), i_count.end(), 1, std::multiplies<>());

        // Add the string length as the rightmost dimension.
        i_count.push_back(string_len);

        // Multiply all elements in i_count.
        // int total_count_char = std::accumulate(i_count.begin(), i_count.end(), 1, std::multiplies<>());

        // Read the entire char array;
        std::vector<char> var_char;
        var_char = input_nc.get_variable<char>(var_name, i_count);

        std::vector<std::string> var;

        for (int n=0; n<total_count; ++n)
        {
            std::string s(var_char.begin()+n*string_len, var_char.begin()+(n+1)*string_len);
            if (trim)
                boost::trim(s);
            var.push_back(s);
        }

        return var;
    }

    Gas_optics<double> load_and_init_gas_optics(
            Master& master,
            const Gas_concs<double>& gas_concs,
            const std::string& coef_file)
    {
        // READ THE COEFFICIENTS FOR THE OPTICAL SOLVER.
        Netcdf_file coef_nc(master, coef_file, Netcdf_mode::Read);

        // Read k-distribution information.
        int n_temps = coef_nc.get_dimension_size("temperature");
        int n_press = coef_nc.get_dimension_size("pressure");
        int n_absorbers = coef_nc.get_dimension_size("absorber");
        int n_char = coef_nc.get_dimension_size("string_len");
        int n_minorabsorbers = coef_nc.get_dimension_size("minor_absorber");
        int n_extabsorbers = coef_nc.get_dimension_size("absorber_ext");
        int n_mixingfracs = coef_nc.get_dimension_size("mixing_fraction");
        int n_layers = coef_nc.get_dimension_size("atmos_layer");
        int n_bnds = coef_nc.get_dimension_size("bnd");
        int n_gpts = coef_nc.get_dimension_size("gpt");
        int n_pairs = coef_nc.get_dimension_size("pair");
        int n_minor_absorber_intervals_lower = coef_nc.get_dimension_size("minor_absorber_intervals_lower");
        int n_minor_absorber_intervals_upper = coef_nc.get_dimension_size("minor_absorber_intervals_upper");
        int n_contributors_lower = coef_nc.get_dimension_size("contributors_lower");
        int n_contributors_upper = coef_nc.get_dimension_size("contributors_upper");

        // Read gas names.
        Array<std::string,1> gas_names(
                get_variable_string("gas_names", {n_absorbers}, coef_nc, n_char, true), {n_absorbers});

        Array<int,3> key_species(
                coef_nc.get_variable<int>("key_species", {n_bnds, n_layers, 2}),
                {2, n_layers, n_bnds});
        Array<double,2> band_lims(coef_nc.get_variable<double>("bnd_limits_wavenumber", {n_bnds, 2}), {2, n_bnds});
        Array<int,2> band2gpt(coef_nc.get_variable<int>("bnd_limits_gpt", {n_bnds, 2}), {2, n_bnds});
        Array<double,1> press_ref(coef_nc.get_variable<double>("press_ref", {n_press}), {n_press});
        Array<double,1> temp_ref(coef_nc.get_variable<double>("temp_ref", {n_temps}), {n_temps});

        double temp_ref_p = coef_nc.get_variable<double>("absorption_coefficient_ref_P");
        double temp_ref_t = coef_nc.get_variable<double>("absorption_coefficient_ref_T");
        double press_ref_trop = coef_nc.get_variable<double>("press_ref_trop");

        Array<double,3> kminor_lower(
                coef_nc.get_variable<double>("kminor_lower", {n_temps, n_mixingfracs, n_contributors_lower}),
                {n_contributors_lower, n_mixingfracs, n_temps});
        Array<double,3> kminor_upper(
                coef_nc.get_variable<double>("kminor_upper", {n_temps, n_mixingfracs, n_contributors_upper}),
                {n_contributors_upper, n_mixingfracs, n_temps});

        Array<std::string,1> gas_minor(get_variable_string("gas_minor", {n_minorabsorbers}, coef_nc, n_char),
                {n_minorabsorbers});

        Array<std::string,1> identifier_minor(
                get_variable_string("identifier_minor", {n_minorabsorbers}, coef_nc, n_char), {n_minorabsorbers});

        Array<std::string,1> minor_gases_lower(
                get_variable_string("minor_gases_lower", {n_minor_absorber_intervals_lower}, coef_nc, n_char),
                {n_minor_absorber_intervals_lower});
        Array<std::string,1> minor_gases_upper(
                get_variable_string("minor_gases_upper", {n_minor_absorber_intervals_upper}, coef_nc, n_char),
                {n_minor_absorber_intervals_upper});

        Array<int,2> minor_limits_gpt_lower(
                coef_nc.get_variable<int>("minor_limits_gpt_lower", {n_minor_absorber_intervals_lower, n_pairs}),
                {n_pairs, n_minor_absorber_intervals_lower});
        Array<int,2> minor_limits_gpt_upper(
                coef_nc.get_variable<int>("minor_limits_gpt_upper", {n_minor_absorber_intervals_upper, n_pairs}),
                {n_pairs, n_minor_absorber_intervals_upper});

        Array<int,1> minor_scales_with_density_lower(
                coef_nc.get_variable<int>("minor_scales_with_density_lower", {n_minor_absorber_intervals_lower}),
                {n_minor_absorber_intervals_lower});
        Array<int,1> minor_scales_with_density_upper(
                coef_nc.get_variable<int>("minor_scales_with_density_upper", {n_minor_absorber_intervals_upper}),
                {n_minor_absorber_intervals_upper});

        Array<int,1> scale_by_complement_lower(
                coef_nc.get_variable<int>("scale_by_complement_lower", {n_minor_absorber_intervals_lower}),
                {n_minor_absorber_intervals_lower});
        Array<int,1> scale_by_complement_upper(
                coef_nc.get_variable<int>("scale_by_complement_upper", {n_minor_absorber_intervals_upper}),
                {n_minor_absorber_intervals_upper});

        Array<std::string,1> scaling_gas_lower(
                get_variable_string("scaling_gas_lower", {n_minor_absorber_intervals_lower}, coef_nc, n_char),
                {n_minor_absorber_intervals_lower});
        Array<std::string,1> scaling_gas_upper(
                get_variable_string("scaling_gas_upper", {n_minor_absorber_intervals_upper}, coef_nc, n_char),
                {n_minor_absorber_intervals_upper});

        Array<int,1> kminor_start_lower(
                coef_nc.get_variable<int>("kminor_start_lower", {n_minor_absorber_intervals_lower}),
                {n_minor_absorber_intervals_lower});
        Array<int,1> kminor_start_upper(
                coef_nc.get_variable<int>("kminor_start_upper", {n_minor_absorber_intervals_upper}),
                {n_minor_absorber_intervals_upper});

        Array<double,3> vmr_ref(
                coef_nc.get_variable<double>("vmr_ref", {n_temps, n_extabsorbers, n_layers}),
                {n_layers, n_extabsorbers, n_temps});

        Array<double,4> kmajor(
                coef_nc.get_variable<double>("kmajor", {n_temps, n_press+1, n_mixingfracs, n_gpts}),
                {n_gpts, n_mixingfracs, n_press+1, n_temps});

        // Keep the size at zero, if it does not exist.
        Array<double,3> rayl_lower;
        Array<double,3> rayl_upper;

        if (coef_nc.variable_exists("rayl_lower"))
        {
            rayl_lower.set_dims({n_gpts, n_mixingfracs, n_temps});
            rayl_upper.set_dims({n_gpts, n_mixingfracs, n_temps});
            rayl_lower = coef_nc.get_variable<double>("rayl_lower", {n_temps, n_mixingfracs, n_gpts});
            rayl_upper = coef_nc.get_variable<double>("rayl_upper", {n_temps, n_mixingfracs, n_gpts});
        }

        // Is it really LW if so read these variables as well.
        if (coef_nc.variable_exists("totplnk"))
        {
            int n_internal_sourcetemps = coef_nc.get_dimension_size("temperature_Planck");

            Array<double,2> totplnk(
                    coef_nc.get_variable<double>( "totplnk", {n_bnds, n_internal_sourcetemps}),
                    {n_internal_sourcetemps, n_bnds});
            Array<double,4> planck_frac(
                    coef_nc.get_variable<double>("plank_fraction", {n_temps, n_press+1, n_mixingfracs, n_gpts}),
                    {n_gpts, n_mixingfracs, n_press+1, n_temps});

            // Construct the k-distribution.
            return Gas_optics<double>(
                    gas_concs,
                    gas_names,
                    key_species,
                    band2gpt,
                    band_lims,
                    press_ref,
                    press_ref_trop,
                    temp_ref,
                    temp_ref_p,
                    temp_ref_t,
                    vmr_ref,
                    kmajor,
                    kminor_lower,
                    kminor_upper,
                    gas_minor,
                    identifier_minor,
                    minor_gases_lower,
                    minor_gases_upper,
                    minor_limits_gpt_lower,
                    minor_limits_gpt_upper,
                    minor_scales_with_density_lower,
                    minor_scales_with_density_upper,
                    scaling_gas_lower,
                    scaling_gas_upper,
                    scale_by_complement_lower,
                    scale_by_complement_upper,
                    kminor_start_lower,
                    kminor_start_upper,
                    totplnk,
                    planck_frac,
                    rayl_lower,
                    rayl_upper);
        }
        else
        {
            Array<double,1> solar_src(
                    coef_nc.get_variable<double>("solar_source", {n_gpts}), {n_gpts});

            return Gas_optics<double>(
                    gas_concs,
                    gas_names,
                    key_species,
                    band2gpt,
                    band_lims,
                    press_ref,
                    press_ref_trop,
                    temp_ref,
                    temp_ref_p,
                    temp_ref_t,
                    vmr_ref,
                    kmajor,
                    kminor_lower,
                    kminor_upper,
                    gas_minor,
                    identifier_minor,
                    minor_gases_lower,
                    minor_gases_upper,
                    minor_limits_gpt_lower,
                    minor_limits_gpt_upper,
                    minor_scales_with_density_lower,
                    minor_scales_with_density_upper,
                    scaling_gas_lower,
                    scaling_gas_upper,
                    scale_by_complement_lower,
                    scale_by_complement_upper,
                    kminor_start_lower,
                    kminor_start_upper,
                    solar_src,
                    rayl_lower,
                    rayl_upper);
        }
        // End reading of k-distribution.
    }
}

int main()
{
    Master master;
    try
    {
        master.start();
        master.init();

        Netcdf_file input_nc(master, "rrtmgp-inputs-outputs.nc", Netcdf_mode::Read);

        // READ THE ATMOSPHERIC DATA.
        int n_lay = input_nc.get_dimension_size("lay");
        int n_lev = input_nc.get_dimension_size("lev");
        int n_col = input_nc.get_dimension_size("col");

        Array<double,2> p_lay(input_nc.get_variable<double>("p_lay", {n_lay, n_col}), {n_col, n_lay});
        Array<double,2> t_lay(input_nc.get_variable<double>("t_lay", {n_lay, n_col}), {n_col, n_lay});
        Array<double,2> p_lev(input_nc.get_variable<double>("p_lev", {n_lev, n_col}), {n_col, n_lev});
        Array<double,2> t_lev(input_nc.get_variable<double>("t_lev", {n_lev, n_col}), {n_col, n_lev});

        const int top_at_1 = p_lay({1, 1}) < p_lay({1, n_lay});

        Gas_concs<double> gas_concs;
        Gas_concs<double> gas_concs_subset;

        gas_concs.set_vmr("h2o",
                Array<double,2>(input_nc.get_variable<double>("vmr_h2o", {n_lay, n_col}), {n_col, n_lay}));
        gas_concs.set_vmr("co2",
                Array<double,2>(input_nc.get_variable<double>("vmr_co2", {n_lay, n_col}), {n_col, n_lay}));
        gas_concs.set_vmr("o3",
                Array<double,2>(input_nc.get_variable<double>("vmr_o3", {n_lay, n_col}), {n_col, n_lay}));
        gas_concs.set_vmr("n2o",
                Array<double,2>(input_nc.get_variable<double>("vmr_n2o", {n_lay, n_col}), {n_col, n_lay}));
        gas_concs.set_vmr("co",
                Array<double,2>(input_nc.get_variable<double>("vmr_co", {n_lay, n_col}), {n_col, n_lay}));
        gas_concs.set_vmr("ch4",
                Array<double,2>(input_nc.get_variable<double>("vmr_ch4", {n_lay, n_col}), {n_col, n_lay}));
        gas_concs.set_vmr("o2",
                Array<double,2>(input_nc.get_variable<double>("vmr_o2", {n_lay, n_col}), {n_col, n_lay}));
        gas_concs.set_vmr("n2",
                Array<double,2>(input_nc.get_variable<double>("vmr_n2", {n_lay, n_col}), {n_col, n_lay}));

        // CvH: does this one need to be present?
        Array<double,2> col_dry(input_nc.get_variable<double>("col_dry", {n_lay, n_col}), {n_col, n_lay});

        // Construct the gas optics class.
        Gas_optics<double> kdist = load_and_init_gas_optics(master, gas_concs, "coefficients.nc");

        const int n_gpt = kdist.get_ngpt();
        const int n_bnd = kdist.get_nband();

        // Boundary conditions for longwave.
        Array<double,2> emis_sfc;
        Array<double,1> t_sfc;

        // Boundary conditions for shortwave.
        Array<double,1> sza;
        Array<double,1> tsi;
        Array<double,2> sfc_alb_dir;
        Array<double,2> sfc_alb_dif;

        const int n_col_block = 4;

        std::unique_ptr<Optical_props_arry<double>> optical_props;
        std::unique_ptr<Optical_props_arry<double>> optical_props_subset;
        std::unique_ptr<Optical_props_arry<double>> optical_props_left;

        if (kdist.source_is_internal())
        {
            ////// SOLVING THE OPTICAL PROPERTIES FOR LONGWAVE RADIATION //////
            master.print_message("STEP 1: Computing optical depths for longwave radiation.\n");

            // Download surface boundary conditions for long wave.
            Array<double,2> emis_sfc_tmp(
                    input_nc.get_variable<double>(
                            "emis_sfc", {n_col, n_bnd}), {n_bnd, n_col});
            Array<double,1> t_sfc_tmp(
                    input_nc.get_variable<double>(
                            "t_sfc", {n_col}), {n_col});

            emis_sfc = emis_sfc_tmp;
            t_sfc = t_sfc_tmp;

            // Read the sources and create containers for the substeps.
            int n_blocks = n_col / n_col_block;
            int n_col_block_left = n_col % n_col_block;

            optical_props        = std::make_unique<Optical_props_1scl<double>>(n_col      , n_lay, kdist);
            optical_props_subset = std::make_unique<Optical_props_1scl<double>>(n_col_block, n_lay, kdist);

            Source_func_lw<double> sources       (n_col      , n_lay, kdist);
            Source_func_lw<double> sources_subset(n_col_block, n_lay, kdist);

            auto calc_optical_props_subset = [&](
                    const int col_s_in, const int col_e_in,
                    std::unique_ptr<Optical_props_arry<double>>& optical_props_subset_in,
                    Source_func_lw<double>& sources_subset_in)
            {
                const int n_col_in = col_e_in - col_s_in + 1;
                Gas_concs<double> gas_concs_subset(gas_concs, col_s_in, n_col_in);

                kdist.gas_optics(
                        p_lay.subset({{ {col_s_in, col_e_in}, {1, n_lay} }}),
                        p_lev.subset({{ {col_s_in, col_e_in}, {1, n_lev} }}),
                        t_lay.subset({{ {col_s_in, col_e_in}, {1, n_lay} }}),
                        t_sfc.subset({{ {col_s_in, col_e_in} }}),
                        gas_concs_subset,
                        optical_props_subset_in,
                        sources_subset_in,
                        col_dry.subset({{ {col_s_in, col_e_in}, {1, n_lay} }}),
                        t_lev.subset  ({{ {col_s_in, col_e_in}, {1, n_lev} }}) );

                optical_props->set_subset(optical_props_subset_in, col_s_in, col_e_in);
                sources.set_subset(sources_subset_in, col_s_in, col_e_in);
            };

            for (int b=1; b<=n_blocks; ++b)
            {
                const int col_s = (b-1) * n_col_block + 1;
                const int col_e =  b    * n_col_block;

                calc_optical_props_subset(
                        col_s, col_e,
                        optical_props_subset,
                        sources_subset);
            }

            if (n_col_block_left > 0)
            {
                optical_props_left = std::make_unique<Optical_props_1scl<double>>(n_col_block_left, n_lay, kdist);
                Source_func_lw<double> sources_left(n_col_block_left, n_lay, kdist);

                const int col_s = n_col - n_col_block_left + 1;
                const int col_e = n_col;

                calc_optical_props_subset(
                        col_s, col_e,
                        optical_props_left,
                        sources_left);
            }


            ////// SOLVING THE FLUXES FOR LONGWAVE RADIATION //////
            master.print_message("STEP 2: Computing the longwave radiation fluxes.\n");

            const int n_ang = input_nc.get_variable<double>("angle");

            Array<double,2> flux_up ({n_col, n_lev});
            Array<double,2> flux_dn ({n_col, n_lev});
            Array<double,2> flux_net({n_col, n_lev});

            Array<double,3> bnd_flux_up ({n_col, n_lev, n_bnd});
            Array<double,3> bnd_flux_dn ({n_col, n_lev, n_bnd});
            Array<double,3> bnd_flux_net({n_col, n_lev, n_bnd});

            auto calc_fluxes_subset = [&](
                    const int col_s_in, const int col_e_in,
                    const std::unique_ptr<Optical_props_arry<double>>& optical_props_subset_in,
                    const Source_func_lw<double>& sources_subset_in,
                    const Array<double,2> emis_sfc_subset_in,
                    std::unique_ptr<Fluxes_broadband<double>>& fluxes)
            {
                const int n_col_block_subset = col_e_in - col_s_in + 1;

                // CvH: I removed the pointer assignments of the fluxes, as this is unportable Fortran code.
                Rte_lw<double>::rte_lw(
                        optical_props_subset_in,
                        top_at_1,
                        sources_subset_in,
                        emis_sfc_subset_in,
                        fluxes,
                        n_ang);

                // Copy the data to the output.
                for (int ilev=1; ilev<=n_lev; ++ilev)
                    for (int icol=1; icol<=n_col_block_subset; ++icol)
                    {
                        flux_up ({icol+col_s_in-1, ilev}) = fluxes->get_flux_up ()({icol, ilev});
                        flux_dn ({icol+col_s_in-1, ilev}) = fluxes->get_flux_dn ()({icol, ilev});
                        flux_net({icol+col_s_in-1, ilev}) = fluxes->get_flux_net()({icol, ilev});
                    }

                // Copy the data to the output.
                for (int ibnd=1; ibnd<=n_bnd; ++ibnd)
                    for (int ilev=1; ilev<=n_lev; ++ilev)
                        for (int icol=1; icol<=n_col_block_subset; ++icol)
                        {
                            bnd_flux_up ({icol+col_s_in-1, ilev, ibnd}) = fluxes->get_bnd_flux_up ()({icol, ilev, ibnd});
                            bnd_flux_dn ({icol+col_s_in-1, ilev, ibnd}) = fluxes->get_bnd_flux_dn ()({icol, ilev, ibnd});
                            bnd_flux_net({icol+col_s_in-1, ilev, ibnd}) = fluxes->get_bnd_flux_net()({icol, ilev, ibnd});
                        }
            };

            for (int b=1; b<=n_blocks; ++b)
            {
                const int col_s = (b-1) * n_col_block + 1;
                const int col_e =  b    * n_col_block;

                optical_props_subset->get_subset(optical_props, col_s, col_e);
                sources_subset.get_subset(sources, col_s, col_e);

                Array<double,2> emis_sfc_subset = emis_sfc.subset({{ {1, n_bnd}, {col_s, col_e} }});

                std::unique_ptr<Fluxes_broadband<double>> fluxes_subset =
                        std::make_unique<Fluxes_byband<double>>(n_col_block, n_lev, n_bnd);

                calc_fluxes_subset(
                        col_s, col_e,
                        optical_props_subset,
                        sources_subset,
                        emis_sfc_subset,
                        fluxes_subset);
            }

            if (n_col_block_left > 0)
            {
                const int col_s = n_col - n_col_block_left + 1;
                const int col_e = n_col;

                // CvH, check for reuse of this field.
                Source_func_lw<double> sources_left(n_col_block_left, n_lay, kdist);

                optical_props_left->get_subset(optical_props, col_s, col_e);
                sources_left.get_subset(sources, col_s, col_e);

                Array<double,2> emis_sfc_left = emis_sfc.subset({{ {1, n_bnd}, {col_s, col_e} }});

                std::unique_ptr<Fluxes_broadband<double>> fluxes_left =
                        std::make_unique<Fluxes_byband<double>>(n_col_block_left, n_lev, n_bnd);

                calc_fluxes_subset(
                        col_s, col_e,
                        optical_props_left,
                        sources_left,
                        emis_sfc_left,
                        fluxes_left);
            }


            ////// SAVING THE MODEL OUTPUT //////
            master.print_message("STEP 3: Saving the output to NetCDF.\n");

            // Save the output of the optical solver to disk.
            Netcdf_file output_nc(master, "test_rrtmgp_out.nc", Netcdf_mode::Create);
            output_nc.add_dimension("col", n_col);
            output_nc.add_dimension("lay", n_lay);
            output_nc.add_dimension("lev", n_lev);
            output_nc.add_dimension("gpt", n_gpt);
            output_nc.add_dimension("band", n_bnd);
            output_nc.add_dimension("pair", 2);

            // WARNING: The storage in the NetCDF interface uses C-ordering and indexing.
            // First, store the optical properties.
            auto nc_band_lims_wvn = output_nc.add_variable<double>("band_lims_wvn", {"band", "pair"});
            auto nc_band_lims_gpt = output_nc.add_variable<int>   ("band_lims_gpt", {"band", "pair"});
            auto nc_tau = output_nc.add_variable<double>("tau", {"gpt", "lay", "col"});

            nc_band_lims_wvn.insert(optical_props->get_band_lims_wavenumber().v(), {0, 0});
            nc_band_lims_gpt.insert(optical_props->get_band_lims_gpoint().v()    , {0, 0});
            nc_tau.insert(optical_props->get_tau().v(), {0, 0, 0});

            // Second, store the sources.
            auto nc_lay_src     = output_nc.add_variable<double>("lay_src"    , {"gpt", "lay", "col"});
            auto nc_lev_src_inc = output_nc.add_variable<double>("lev_src_inc", {"gpt", "lay", "col"});
            auto nc_lev_src_dec = output_nc.add_variable<double>("lev_src_dec", {"gpt", "lay", "col"});

            auto nc_sfc_src = output_nc.add_variable<double>("sfc_src", {"gpt", "col"});

            nc_lay_src.insert    (sources.get_lay_source().v()    , {0, 0, 0});
            nc_lev_src_inc.insert(sources.get_lev_source_inc().v(), {0, 0, 0});
            nc_lev_src_dec.insert(sources.get_lev_source_dec().v(), {0, 0, 0});

            nc_sfc_src.insert(sources.get_sfc_source().v(), {0, 0});

            // Save the output of the flux calculation to disk.
            auto nc_flux_up  = output_nc.add_variable<double>("flux_up" , {"lev", "col"});
            auto nc_flux_dn  = output_nc.add_variable<double>("flux_dn" , {"lev", "col"});
            auto nc_flux_net = output_nc.add_variable<double>("flux_net", {"lev", "col"});

            auto nc_bnd_flux_up  = output_nc.add_variable<double>("bnd_flux_up" , {"band", "lev", "col"});
            auto nc_bnd_flux_dn  = output_nc.add_variable<double>("bnd_flux_dn" , {"band", "lev", "col"});
            auto nc_bnd_flux_net = output_nc.add_variable<double>("bnd_flux_net", {"band", "lev", "col"});

            nc_flux_up .insert(flux_up .v(), {0, 0});
            nc_flux_dn .insert(flux_dn .v(), {0, 0});
            nc_flux_net.insert(flux_net.v(), {0, 0});

            nc_bnd_flux_up .insert(bnd_flux_up .v(), {0, 0, 0});
            nc_bnd_flux_dn .insert(bnd_flux_dn .v(), {0, 0, 0});
            nc_bnd_flux_net.insert(bnd_flux_net.v(), {0, 0, 0});
        }
        else
        {
            ////// SOLVING THE OPTICAL PROPERTIES FOR SHORTWAVE RADIATION //////
            master.print_message("STEP 1: Computing optical depths for shortwave radiation.\n");

            // Read the boundary conditions from disk.
            sza.set_dims({n_col});
            tsi.set_dims({n_col});
            sfc_alb_dir.set_dims({n_bnd, n_col});
            sfc_alb_dif.set_dims({n_bnd, n_col});

            sza = input_nc.get_variable<double>("solar_zenith_angle", {n_col});
            tsi = input_nc.get_variable<double>("total_solar_irradiance", {n_col});
            sfc_alb_dir = input_nc.get_variable<double>("sfc_alb_direct" , {n_col, n_bnd});
            sfc_alb_dif = input_nc.get_variable<double>("sfc_alb_diffuse", {n_col, n_bnd});

            double tsi_scaling = -999.;
            if (input_nc.variable_exists("tsi_scaling"))
                tsi_scaling = input_nc.get_variable<double>("t_sfc");

            Array<double,1> mu0({n_col});
            for (int icol=1; icol<=n_col; ++icol)
            {
                const double pi = std::atan(1.)*4.;
                mu0({icol}) = std::cos(sza({icol}) * pi/180.);
            }

            Array<double,2> toa_src({n_col, n_gpt});

            // Read the sources and create containers for the substeps.
            int n_blocks = n_col / n_col_block;
            int n_col_block_left = n_col % n_col_block;

            optical_props        = std::make_unique<Optical_props_2str<double>>(n_col      , n_lay, kdist);
            optical_props_subset = std::make_unique<Optical_props_2str<double>>(n_col_block, n_lay, kdist);

            auto calc_optical_props_subset = [&](
                    const int col_s_in, const int col_e_in,
                    std::unique_ptr<Optical_props_arry<double>>& optical_props_subset_in)
            {
                const int n_col_in = col_e_in - col_s_in + 1;
                Gas_concs<double> gas_concs_subset(gas_concs, col_s_in, n_col_in);
                Array<double,2> toa_src_subset({n_col_in, n_gpt});

                kdist.gas_optics(
                        p_lay.subset({{ {col_s_in, col_e_in}, {1, n_lay} }}),
                        p_lev.subset({{ {col_s_in, col_e_in}, {1, n_lev} }}),
                        t_lay.subset({{ {col_s_in, col_e_in}, {1, n_lay} }}),
                        gas_concs_subset,
                        optical_props_subset_in,
                        toa_src_subset,
                        col_dry.subset({{ {col_s_in, col_e_in}, {1, n_lay} }}) );

                optical_props->set_subset(optical_props_subset_in, col_s_in, col_e_in);

                // Copy the data to the output.
                for (int igpt=1; igpt<=n_gpt; ++igpt)
                    for (int icol=1; icol<=n_col_in; ++icol)
                        toa_src({icol+col_s_in-1, igpt}) = toa_src_subset({icol, igpt});
            };

            for (int b=1; b<=n_blocks; ++b)
            {
                const int col_s = (b-1) * n_col_block + 1;
                const int col_e =  b    * n_col_block;

                calc_optical_props_subset(
                        col_s, col_e,
                        optical_props_subset);
            }

            if (n_col_block_left > 0)
            {
                optical_props_left = std::make_unique<Optical_props_2str<double>>(n_col_block_left, n_lay, kdist);

                const int col_s = n_col - n_col_block_left + 1;
                const int col_e = n_col;

                calc_optical_props_subset(
                        col_s, col_e,
                        optical_props_left);
            }


            ////// SOLVING THE FLUXES FOR SHORTWAVE RADIATION //////
            master.print_message("STEP 2: Computing the shortwave radiation fluxes.\n");

            Array<double,2> flux_up    ({n_col, n_lev});
            Array<double,2> flux_dn    ({n_col, n_lev});
            Array<double,2> flux_dn_dir({n_col, n_lev});
            Array<double,2> flux_net   ({n_col, n_lev});

            Array<double,3> bnd_flux_up    ({n_col, n_lev, n_bnd});
            Array<double,3> bnd_flux_dn    ({n_col, n_lev, n_bnd});
            Array<double,3> bnd_flux_dn_dir({n_col, n_lev, n_bnd});
            Array<double,3> bnd_flux_net   ({n_col, n_lev, n_bnd});

            auto calc_fluxes_subset = [&](
                    const int col_s_in, const int col_e_in,
                    const std::unique_ptr<Optical_props_arry<double>>& optical_props_subset_in,
                    const Array<double,1>& mu0_subset_in,
                    const Array<double,2>& toa_src_subset_in,
                    const Array<double,2>& sfc_alb_dir_subset_in,
                    const Array<double,2>& sfc_alb_dif_subset_in,
                    std::unique_ptr<Fluxes_broadband<double>>& fluxes)
            {
                const int n_col_block_subset = col_e_in - col_s_in + 1;

                Rte_sw<double>::rte_sw(
                        optical_props_subset_in,
                        top_at_1,
                        mu0_subset_in,
                        toa_src_subset_in,
                        sfc_alb_dir_subset_in,
                        sfc_alb_dif_subset_in,
                        fluxes);

                // Copy the data to the output.
                for (int ilev=1; ilev<=n_lev; ++ilev)
                    for (int icol=1; icol<=n_col_block_subset; ++icol)
                    {
                        flux_up    ({icol+col_s_in-1, ilev}) = fluxes->get_flux_up    ()({icol, ilev});
                        flux_dn    ({icol+col_s_in-1, ilev}) = fluxes->get_flux_dn    ()({icol, ilev});
                        flux_dn_dir({icol+col_s_in-1, ilev}) = fluxes->get_flux_dn_dir()({icol, ilev});
                        flux_net   ({icol+col_s_in-1, ilev}) = fluxes->get_flux_net   ()({icol, ilev});
                    }

                // Copy the data to the output.
                for (int ibnd=1; ibnd<=n_bnd; ++ibnd)
                    for (int ilev=1; ilev<=n_lev; ++ilev)
                        for (int icol=1; icol<=n_col_block_subset; ++icol)
                        {
                            bnd_flux_up    ({icol+col_s_in-1, ilev, ibnd}) = fluxes->get_bnd_flux_up    ()({icol, ilev, ibnd});
                            bnd_flux_dn    ({icol+col_s_in-1, ilev, ibnd}) = fluxes->get_bnd_flux_dn    ()({icol, ilev, ibnd});
                            bnd_flux_dn_dir({icol+col_s_in-1, ilev, ibnd}) = fluxes->get_bnd_flux_dn_dir()({icol, ilev, ibnd});
                            bnd_flux_net   ({icol+col_s_in-1, ilev, ibnd}) = fluxes->get_bnd_flux_net   ()({icol, ilev, ibnd});
                        }
            };

            for (int b=1; b<=n_blocks; ++b)
            {
                const int col_s = (b-1) * n_col_block + 1;
                const int col_e =  b    * n_col_block;

                optical_props_subset->get_subset(optical_props, col_s, col_e);

                Array<double,1> mu0_subset = mu0.subset({{ {col_s, col_e} }});
                Array<double,2> toa_src_subset = toa_src.subset({{ {col_s, col_e}, {1, n_gpt} }});
                Array<double,2> sfc_alb_dir_subset = sfc_alb_dir.subset({{ {1, n_bnd}, {col_s, col_e} }});
                Array<double,2> sfc_alb_dif_subset = sfc_alb_dif.subset({{ {1, n_bnd}, {col_s, col_e} }});

                std::unique_ptr<Fluxes_broadband<double>> fluxes_subset =
                        std::make_unique<Fluxes_byband<double>>(n_col_block, n_lev, n_bnd);

                calc_fluxes_subset(
                        col_s, col_e,
                        optical_props_subset,
                        mu0_subset,
                        toa_src_subset,
                        sfc_alb_dir_subset,
                        sfc_alb_dif_subset,
                        fluxes_subset);
            }

            if (n_col_block_left > 0)
            {
                const int col_s = n_col - n_col_block_left + 1;
                const int col_e = n_col;

                optical_props_left->get_subset(optical_props, col_s, col_e);

                Array<double,1> mu0_left = mu0.subset({{ {col_s, col_e} }});
                Array<double,2> toa_src_left = toa_src.subset({{ {col_s, col_e}, {1, n_gpt} }});
                Array<double,2> sfc_alb_dir_left = sfc_alb_dir.subset({{ {1, n_bnd}, {col_s, col_e} }});
                Array<double,2> sfc_alb_dif_left = sfc_alb_dif.subset({{ {1, n_bnd}, {col_s, col_e} }});

                std::unique_ptr<Fluxes_broadband<double>> fluxes_left =
                        std::make_unique<Fluxes_byband<double>>(n_col_block_left, n_lev, n_bnd);

                calc_fluxes_subset(
                        col_s, col_e,
                        optical_props_left,
                        mu0_left,
                        toa_src_left,
                        sfc_alb_dir_left,
                        sfc_alb_dif_left,
                        fluxes_left);
            }


            ////// SAVING THE MODEL OUTPUT //////
            master.print_message("STEP 3: Saving the output to NetCDF.\n");

            // Save the output of the optical solver to disk.
            Netcdf_file output_nc(master, "test_rrtmgp_out.nc", Netcdf_mode::Create);
            output_nc.add_dimension("col", n_col);
            output_nc.add_dimension("lay", n_lay);
            output_nc.add_dimension("lev", n_lev);
            output_nc.add_dimension("gpt", n_gpt);
            output_nc.add_dimension("band", n_bnd);
            output_nc.add_dimension("pair", 2);

            // WARNING: The storage in the NetCDF interface uses C-ordering and indexing.
            // First, store the optical properties.
            auto nc_band_lims_wvn = output_nc.add_variable<double>("band_lims_wvn", {"band", "pair"});
            auto nc_band_lims_gpt = output_nc.add_variable<int>   ("band_lims_gpt", {"band", "pair"});
            auto nc_tau = output_nc.add_variable<double>("tau", {"gpt", "lay", "col"});
            auto nc_ssa = output_nc.add_variable<double>("ssa", {"gpt", "lay", "col"});
            auto nc_g   = output_nc.add_variable<double>("g"  , {"gpt", "lay", "col"});

            nc_band_lims_wvn.insert(optical_props->get_band_lims_wavenumber().v(), {0, 0});
            nc_band_lims_gpt.insert(optical_props->get_band_lims_gpoint().v()    , {0, 0});

            nc_tau.insert(optical_props->get_tau().v(), {0, 0, 0});
            nc_ssa.insert(optical_props->get_ssa().v(), {0, 0, 0});
            nc_g  .insert(optical_props->get_g  ().v(), {0, 0, 0});

            auto nc_toa_src = output_nc.add_variable<double>("toa_src", {"gpt", "col"});
            nc_toa_src.insert(toa_src.v(), {0, 0});

            // Save the output of the flux calculation to disk.
            auto nc_flux_up     = output_nc.add_variable<double>("flux_up"    , {"lev", "col"});
            auto nc_flux_dn     = output_nc.add_variable<double>("flux_dn"    , {"lev", "col"});
            auto nc_flux_dn_dir = output_nc.add_variable<double>("flux_dn_dir", {"lev", "col"});
            auto nc_flux_net    = output_nc.add_variable<double>("flux_net"   , {"lev", "col"});

            auto nc_bnd_flux_up     = output_nc.add_variable<double>("bnd_flux_up"    , {"band", "lev", "col"});
            auto nc_bnd_flux_dn     = output_nc.add_variable<double>("bnd_flux_dn"    , {"band", "lev", "col"});
            auto nc_bnd_flux_dn_dir = output_nc.add_variable<double>("bnd_flux_dn_dir", {"band", "lev", "col"});
            auto nc_bnd_flux_net    = output_nc.add_variable<double>("bnd_flux_net"   , {"band", "lev", "col"});

            nc_flux_up    .insert(flux_up    .v(), {0, 0});
            nc_flux_dn    .insert(flux_dn    .v(), {0, 0});
            nc_flux_dn_dir.insert(flux_dn_dir.v(), {0, 0});
            nc_flux_net   .insert(flux_net   .v(), {0, 0});

            nc_bnd_flux_up    .insert(bnd_flux_up    .v(), {0, 0, 0});
            nc_bnd_flux_dn    .insert(bnd_flux_dn    .v(), {0, 0, 0});
            nc_bnd_flux_dn_dir.insert(bnd_flux_dn_dir.v(), {0, 0, 0});
            nc_bnd_flux_net   .insert(bnd_flux_net   .v(), {0, 0, 0});
        }
    }

    // Catch any exceptions and return 1.
    catch (const std::exception& e)
    {
        master.print_message("EXCEPTION: %s\n", e.what());
        return 1;
    }
    catch (...)
    {
        master.print_message("UNHANDLED EXCEPTION!\n");
        return 1;
    }

    // Return 0 in case of normal exit.
    return 0;
}
