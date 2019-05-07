#include <boost/algorithm/string.hpp>
#include "Netcdf_interface.h"
#include "Array.h"
#include "Gas_concs.h"
#include "Gas_optics.h"
#include "Source_functions.h"
#include "Fluxes.h"
#include "Rte_lw.h"

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

        Array<double, 2> p_lay(input_nc.get_variable<double>("p_lay", {n_lay, n_col}), {n_col, n_lay});
        Array<double, 2> t_lay(input_nc.get_variable<double>("t_lay", {n_lay, n_col}), {n_col, n_lay});
        Array<double, 2> p_lev(input_nc.get_variable<double>("p_lev", {n_lev, n_col}), {n_col, n_lev});
        Array<double, 2> t_lev(input_nc.get_variable<double>("t_lev", {n_lev, n_col}), {n_col, n_lev});

        const int top_at_1 = p_lay({1, 1}) < p_lay({1, n_lay});

        Gas_concs<double> gas_concs;
        Gas_concs<double> gas_concs_subset;

        gas_concs.set_vmr("h2o",
                          Array<double, 2>(input_nc.get_variable<double>("vmr_h2o", {n_lay, n_col}), {n_col, n_lay}));
        gas_concs.set_vmr("co2",
                          Array<double, 2>(input_nc.get_variable<double>("vmr_co2", {n_lay, n_col}), {n_col, n_lay}));
        gas_concs.set_vmr("o3",
                          Array<double, 2>(input_nc.get_variable<double>("vmr_o3", {n_lay, n_col}), {n_col, n_lay}));
        gas_concs.set_vmr("n2o",
                          Array<double, 2>(input_nc.get_variable<double>("vmr_n2o", {n_lay, n_col}), {n_col, n_lay}));
        gas_concs.set_vmr("co",
                          Array<double, 2>(input_nc.get_variable<double>("vmr_co", {n_lay, n_col}), {n_col, n_lay}));
        gas_concs.set_vmr("ch4",
                          Array<double, 2>(input_nc.get_variable<double>("vmr_ch4", {n_lay, n_col}), {n_col, n_lay}));
        gas_concs.set_vmr("o2",
                          Array<double, 2>(input_nc.get_variable<double>("vmr_o2", {n_lay, n_col}), {n_col, n_lay}));
        gas_concs.set_vmr("n2",
                          Array<double, 2>(input_nc.get_variable<double>("vmr_n2", {n_lay, n_col}), {n_col, n_lay}));

        // CvH: does this one need to be present?
        Array<double, 2> col_dry(input_nc.get_variable<double>("col_dry", {n_lay, n_col}), {n_col, n_lay});

        // READ THE COEFFICIENTS FOR THE OPTICAL SOLVER.
        Netcdf_file coef_lw_nc(master, "coefficients_lw.nc", Netcdf_mode::Read);

        // Read k-distribution information.
        int n_temps = coef_lw_nc.get_dimension_size("temperature");
        int n_press = coef_lw_nc.get_dimension_size("pressure");
        int n_absorbers = coef_lw_nc.get_dimension_size("absorber");
        int n_char = coef_lw_nc.get_dimension_size("string_len");
        int n_minorabsorbers = coef_lw_nc.get_dimension_size("minor_absorber");
        int n_extabsorbers = coef_lw_nc.get_dimension_size("absorber_ext");
        int n_mixingfracs = coef_lw_nc.get_dimension_size("mixing_fraction");
        int n_layers = coef_lw_nc.get_dimension_size("atmos_layer");
        int n_bnds = coef_lw_nc.get_dimension_size("bnd");
        int n_gpts = coef_lw_nc.get_dimension_size("gpt");
        int n_pairs = coef_lw_nc.get_dimension_size("pair");
        int n_minor_absorber_intervals_lower = coef_lw_nc.get_dimension_size("minor_absorber_intervals_lower");
        int n_minor_absorber_intervals_upper = coef_lw_nc.get_dimension_size("minor_absorber_intervals_upper");
        int n_internal_sourcetemps = coef_lw_nc.get_dimension_size("temperature_Planck");
        int n_contributors_lower = coef_lw_nc.get_dimension_size("contributors_lower");
        int n_contributors_upper = coef_lw_nc.get_dimension_size("contributors_upper");

        // Read gas names.
        Array<std::string, 1> gas_names(get_variable_string("gas_names", {n_absorbers}, coef_lw_nc, n_char, true),
                                        {n_absorbers});

        Array<int, 3> key_species(coef_lw_nc.get_variable<int>("key_species", {n_bnds, n_layers, 2}),
                                  {2, n_layers, n_bnds});
        Array<double, 2> band_lims(coef_lw_nc.get_variable<double>("bnd_limits_wavenumber", {n_bnds, 2}), {2, n_bnds});
        Array<int, 2> band2gpt(coef_lw_nc.get_variable<int>("bnd_limits_gpt", {n_bnds, 2}), {2, n_bnds});
        Array<double, 1> press_ref(coef_lw_nc.get_variable<double>("press_ref", {n_press}), {n_press});
        Array<double, 1> temp_ref(coef_lw_nc.get_variable<double>("temp_ref", {n_temps}), {n_temps});

        double temp_ref_p = coef_lw_nc.get_variable<double>("absorption_coefficient_ref_P");
        double temp_ref_t = coef_lw_nc.get_variable<double>("absorption_coefficient_ref_T");
        double press_ref_trop = coef_lw_nc.get_variable<double>("press_ref_trop");

        Array<double, 3> kminor_lower(
                coef_lw_nc.get_variable<double>("kminor_lower", {n_temps, n_mixingfracs, n_contributors_lower}),
                {n_contributors_lower, n_mixingfracs, n_temps});
        Array<double, 3> kminor_upper(
                coef_lw_nc.get_variable<double>("kminor_upper", {n_temps, n_mixingfracs, n_contributors_upper}),
                {n_contributors_upper, n_mixingfracs, n_temps});

        Array<std::string, 1> gas_minor(get_variable_string("gas_minor", {n_minorabsorbers}, coef_lw_nc, n_char),
                                        {n_minorabsorbers});
        Array<std::string, 1> identifier_minor(
                get_variable_string("identifier_minor", {n_minorabsorbers}, coef_lw_nc, n_char), {n_minorabsorbers});

        Array<std::string, 1> minor_gases_lower(
                get_variable_string("minor_gases_lower", {n_minor_absorber_intervals_lower}, coef_lw_nc, n_char),
                {n_minor_absorber_intervals_lower});
        Array<std::string, 1> minor_gases_upper(
                get_variable_string("minor_gases_upper", {n_minor_absorber_intervals_upper}, coef_lw_nc, n_char),
                {n_minor_absorber_intervals_upper});

        Array<int, 2> minor_limits_gpt_lower(
                coef_lw_nc.get_variable<int>("minor_limits_gpt_lower", {n_minor_absorber_intervals_lower, n_pairs}),
                {n_pairs, n_minor_absorber_intervals_lower});
        Array<int, 2> minor_limits_gpt_upper(
                coef_lw_nc.get_variable<int>("minor_limits_gpt_upper", {n_minor_absorber_intervals_upper, n_pairs}),
                {n_pairs, n_minor_absorber_intervals_upper});

        Array<int, 1> minor_scales_with_density_lower(
                coef_lw_nc.get_variable<int>("minor_scales_with_density_lower", {n_minor_absorber_intervals_lower}),
                {n_minor_absorber_intervals_lower});
        Array<int, 1> minor_scales_with_density_upper(
                coef_lw_nc.get_variable<int>("minor_scales_with_density_upper", {n_minor_absorber_intervals_upper}),
                {n_minor_absorber_intervals_upper});

        Array<int, 1> scale_by_complement_lower(
                coef_lw_nc.get_variable<int>("scale_by_complement_lower", {n_minor_absorber_intervals_lower}),
                {n_minor_absorber_intervals_lower});
        Array<int, 1> scale_by_complement_upper(
                coef_lw_nc.get_variable<int>("scale_by_complement_upper", {n_minor_absorber_intervals_upper}),
                {n_minor_absorber_intervals_upper});

        Array<std::string, 1> scaling_gas_lower(
                get_variable_string("scaling_gas_lower", {n_minor_absorber_intervals_lower}, coef_lw_nc, n_char),
                {n_minor_absorber_intervals_lower});
        Array<std::string, 1> scaling_gas_upper(
                get_variable_string("scaling_gas_upper", {n_minor_absorber_intervals_upper}, coef_lw_nc, n_char),
                {n_minor_absorber_intervals_upper});

        Array<int, 1> kminor_start_lower(
                coef_lw_nc.get_variable<int>("kminor_start_lower", {n_minor_absorber_intervals_lower}),
                {n_minor_absorber_intervals_lower});
        Array<int, 1> kminor_start_upper(
                coef_lw_nc.get_variable<int>("kminor_start_upper", {n_minor_absorber_intervals_upper}),
                {n_minor_absorber_intervals_upper});

        Array<double, 3> vmr_ref(coef_lw_nc.get_variable<double>("vmr_ref", {n_temps, n_extabsorbers, n_layers}),
                                 {n_layers, n_extabsorbers, n_temps});

        Array<double, 4> kmajor(
                coef_lw_nc.get_variable<double>("kmajor", {n_temps, n_press + 1, n_mixingfracs, n_gpts}),
                {n_gpts, n_mixingfracs, n_press + 1, n_temps});

        Array<double, 3> rayl_lower({n_gpts, n_mixingfracs, n_temps});
        Array<double, 3> rayl_upper({n_gpts, n_mixingfracs, n_temps});

        if (coef_lw_nc.variable_exists("rayl_lower"))
        {
            throw std::runtime_error("rayl reading not implemented!");
            // rayl_lower = read_field(ncid, 'rayl_lower', ngpts, nmixingfracs, ntemps)
            // rayl_upper = read_field(ncid, 'rayl_upper', ngpts, nmixingfracs, ntemps)
        }

        // Is it really LW if so read these variables as well.
        Array<double, 2> totplnk({n_internal_sourcetemps, n_bnds});
        Array<double, 4> planck_frac({n_gpts, n_mixingfracs, n_press + 1, n_temps});

        if (coef_lw_nc.variable_exists("totplnk"))
        {
            totplnk = coef_lw_nc.get_variable<double>("totplnk", {n_bnds, n_internal_sourcetemps});
            planck_frac = coef_lw_nc.get_variable<double>("plank_fraction",
                                                          {n_temps, n_press + 1, n_mixingfracs, n_gpts});
        }
        else
        {
            throw std::runtime_error("short wave not implemented!");
        }
        // End reading of k-distribution.

        // Construct the k-distribution.
        Gas_optics<double> kdist(
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

        const int n_gpt = kdist.get_ngpt();
        const int n_bnd = kdist.get_nband();

        Array<double, 2> emis_sfc;
        Array<double, 1> t_sfc;

        const int n_col_block = 4;

        std::unique_ptr<Optical_props_arry<double>> optical_props;
        std::unique_ptr<Optical_props_arry<double>> optical_props_subset;
        std::unique_ptr<Optical_props_arry<double>> optical_props_left;

        if (kdist.source_is_internal())
        {
            /// SOLVING THE OPTICAL PROPERTIES FOR LONGWAVE RADIATION.
            master.print_message("STEP 1: Computing optical depths for longwave radiation\n");

            // Download surface boundary conditions for long wave.
            Array<double, 2> emis_sfc_tmp(input_nc.get_variable<double>("emis_sfc", {n_col, n_bnd}), {n_bnd, n_col});
            Array<double, 1> t_sfc_tmp(input_nc.get_variable<double>("t_sfc", {n_col}), {n_col});

            emis_sfc = emis_sfc_tmp;
            t_sfc = t_sfc_tmp;

            // Read the sources and create containers for the substeps.
            int n_blocks = n_col / n_col_block;
            int n_col_block_left = n_col % n_col_block;

            optical_props = std::make_unique<Optical_props_1scl<double>>(n_col, n_lay, kdist);
            optical_props_subset = std::make_unique<Optical_props_1scl<double>>(n_col_block, n_lay, kdist);
            Source_func_lw<double> sources(n_col, n_lay, kdist);
            Source_func_lw<double> sources_subset(n_col_block, n_lay, kdist);

            auto calc_optical_props_subset = [&](
                    const int col_s_in, const int col_e_in,
                    std::unique_ptr<Optical_props_arry<double>>& optical_props_subset_in,
                    Source_func_lw<double>& sources_subset_in)
            {
                const int n_col_in = col_e_in - col_s_in + 1;
                Gas_concs<double> gas_concs_subset(gas_concs, col_s_in, n_col_in);

                kdist.gas_optics(
                        p_lay.subset({{{col_s_in, col_e_in}, {1, n_lay}}}),
                        p_lev.subset({{{col_s_in, col_e_in}, {1, n_lev}}}),
                        t_lay.subset({{{col_s_in, col_e_in}, {1, n_lay}}}),
                        t_sfc.subset({{{col_s_in, col_e_in}}}),
                        gas_concs_subset,
                        optical_props_subset_in,
                        sources_subset_in,
                        col_dry.subset({{{col_s_in, col_e_in}, {1, n_lay}}}),
                        t_lev.subset({{{col_s_in, col_e_in}, {1, n_lev}}})
                );

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

            // Save the output to disk.
            /*
            Netcdf_file output_nc(master, "test_rrtmgp_out.nc", Netcdf_mode::Create);
            output_nc.add_dimension("col", n_col);
            output_nc.add_dimension("lay", n_lay);
            output_nc.add_dimension("gpt", n_gpt);
            output_nc.add_dimension("band", n_bnd);
            output_nc.add_dimension("pair", 2);

            // WARNING: The storage in the NetCDF interface uses C-ordering and indexing.
            // First, store the optical properties.
            auto nc_band_lims_wvn = output_nc.add_variable<double>("band_lims_wvn", {"band", "pair"});
            auto nc_band_lims_gpt = output_nc.add_variable<int>("band_lims_gpt", {"band", "pair"});
            auto nc_tau = output_nc.add_variable<double>("tau", {"gpt", "lay", "col"});

            nc_band_lims_wvn.insert(optical_props->get_band_lims_wavenumber().v(), {0, 0});
            nc_band_lims_gpt.insert(optical_props->get_band_lims_gpoint().v(), {0, 0});
            nc_tau.insert(optical_props->get_tau().v(), {0, 0, 0});

            // Second, store the sources.
            auto nc_lay_src = output_nc.add_variable<double>("lay_src", {"gpt", "lay", "col"});
            auto nc_lev_src_inc = output_nc.add_variable<double>("lev_src_inc", {"gpt", "lay", "col"});
            auto nc_lev_src_dec = output_nc.add_variable<double>("lev_src_dec", {"gpt", "lay", "col"});
            auto nc_sfc_src = output_nc.add_variable<double>("sfc_src", {"gpt", "col"});

            nc_lay_src.insert(sources.get_lay_source().v(), {0, 0, 0});
            nc_lev_src_inc.insert(sources.get_lev_source_inc().v(), {0, 0, 0});
            nc_lev_src_dec.insert(sources.get_lev_source_dec().v(), {0, 0, 0});
            nc_sfc_src.insert(sources.get_sfc_source().v(), {0, 0});
            */

            /// SOLVING THE FLUXES FOR LONGWAVE RADIATION.
            master.print_message("STEP 2: Computing optical depths for longwave radiation\n");

            const int n_ang = input_nc.get_variable<double>("angle");

            Array<double,2> flux_up({n_col, n_lay+1});
            Array<double,2> flux_dn({n_col, n_lay+1});
            Array<double,3> bnd_flux_up({n_col, n_lay+1, n_bnd});
            Array<double,3> bnd_flux_dn({n_col, n_lay+1, n_bnd});

            Array<double,2> heating_rate({n_col, n_lay});
            Array<double,3> bnd_heating_rate({n_col, n_lay, n_bnd});

            auto calc_fluxes_subset = [&](
                    const int col_s_in, const int col_e_in,
                    const std::unique_ptr<Optical_props_arry<double>>& optical_props_subset_in,
                    const Source_func_lw<double>& sources_subset_in,
                    const Array<double,2> emis_sfc_subset_in)
            {
                const int n_col_in = col_e_in - col_s_in + 1;

                std::unique_ptr<Fluxes<double>> fluxes = std::make_unique<Fluxes_byband<double>>();

                Rte_lw<double>::rte_lw(
                        optical_props_subset_in,
                        top_at_1,
                        sources_subset_in,
                        emis_sfc_subset_in,
                        fluxes,
                        n_ang);
            };

            for (int b=1; b<=n_blocks; ++b)
            {
                const int col_s = (b-1) * n_col_block + 1;
                const int col_e =  b    * n_col_block;

                optical_props_subset->get_subset(optical_props, col_s, col_e);
                sources_subset.get_subset(sources, col_s, col_e);

                Array<double,2> emis_sfc_subset = emis_sfc.subset({{ {1, n_bnd}, {col_s, col_e} }});

                calc_fluxes_subset(
                        col_s, col_e,
                        optical_props_subset,
                        sources_subset,
                        emis_sfc_subset);
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

                calc_fluxes_subset(
                        col_s, col_e,
                        optical_props_left,
                        sources_left,
                        emis_sfc_left);
            }
        }
        else
        {
            master.print_message("Computing optical depths for shortwave radiation\n");
            throw std::runtime_error("Shortwave radiation not implemented");
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
