#include <boost/algorithm/string.hpp>
#include "Netcdf_interface.h"
#include "Array.h"
#include "Gas_concs.h"
#include "Gas_optics.h"
#include "Source_functions.h"

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

    Gas_concs<double> gas_concs;
    Gas_concs<double> gas_concs_subset;

    gas_concs.set_vmr("h2o", Array<double,2>(input_nc.get_variable<double>("vmr_h2o", {n_lay, n_col}), {n_col, n_lay}));
    gas_concs.set_vmr("co2", Array<double,2>(input_nc.get_variable<double>("vmr_co2", {n_lay, n_col}), {n_col, n_lay}));
    gas_concs.set_vmr("o3" , Array<double,2>(input_nc.get_variable<double>("vmr_o3" , {n_lay, n_col}), {n_col, n_lay}));
    gas_concs.set_vmr("n2o", Array<double,2>(input_nc.get_variable<double>("vmr_n2o", {n_lay, n_col}), {n_col, n_lay}));
    gas_concs.set_vmr("co" , Array<double,2>(input_nc.get_variable<double>("vmr_co" , {n_lay, n_col}), {n_col, n_lay}));
    gas_concs.set_vmr("ch4", Array<double,2>(input_nc.get_variable<double>("vmr_ch4", {n_lay, n_col}), {n_col, n_lay}));
    gas_concs.set_vmr("o2" , Array<double,2>(input_nc.get_variable<double>("vmr_o2" , {n_lay, n_col}), {n_col, n_lay}));
    gas_concs.set_vmr("n2" , Array<double,2>(input_nc.get_variable<double>("vmr_n2" , {n_lay, n_col}), {n_col, n_lay}));

    // CvH: does this one need to be present?
    Array<double,2> col_dry(input_nc.get_variable<double>("col_dry", {n_lay, n_col}), {n_col, n_lay});

    // READ THE COEFFICIENTS FOR THE OPTICAL SOLVER.
    Netcdf_file coef_lw_nc(master, "coefficients_lw.nc", Netcdf_mode::Read);

    // const int top_at_1 = pres_layer({1, 1}) < pres_layer({1, n_lay});

    // Download surface boundary conditions for long wave.
    // Array<double,1> surface_emissivity (group_nc.get_variable<double>("surface_emissivity" , {n_col}), {n_col});
    // Array<double,1> surface_temperature(group_nc.get_variable<double>("surface_temperature", {n_col}), {n_col});

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
    Array<std::string,1> gas_names(get_variable_string("gas_names", {n_absorbers}, coef_lw_nc, n_char, true), {n_absorbers});

    Array<int,3> key_species(coef_lw_nc.get_variable<int>("key_species", {n_bnds, n_layers, 2}), {2, n_layers, n_bnds});
    Array<double,2> band_lims(coef_lw_nc.get_variable<double>("bnd_limits_wavenumber", {n_bnds, 2}), {2, n_bnds});
    Array<int,2> band2gpt(coef_lw_nc.get_variable<int>("bnd_limits_gpt", {n_bnds, 2}), {2, n_bnds});
    Array<double,1> press_ref(coef_lw_nc.get_variable<double>("press_ref", {n_press}), {n_press});
    Array<double,1> temp_ref(coef_lw_nc.get_variable<double>("temp_ref", {n_temps}), {n_temps});

    double temp_ref_p = coef_lw_nc.get_variable<double>("absorption_coefficient_ref_P");
    double temp_ref_t = coef_lw_nc.get_variable<double>("absorption_coefficient_ref_T");
    double press_ref_trop = coef_lw_nc.get_variable<double>("press_ref_trop");

    Array<double,3> kminor_lower(coef_lw_nc.get_variable<double>("kminor_lower", {n_temps, n_mixingfracs, n_contributors_lower}), {n_contributors_lower, n_mixingfracs, n_temps});
    Array<double,3> kminor_upper(coef_lw_nc.get_variable<double>("kminor_upper", {n_temps, n_mixingfracs, n_contributors_upper}), {n_contributors_upper, n_mixingfracs, n_temps});

    Array<std::string,1> gas_minor(get_variable_string("gas_minor", {n_minorabsorbers}, coef_lw_nc, n_char), {n_minorabsorbers});
    Array<std::string,1> identifier_minor(get_variable_string("identifier_minor", {n_minorabsorbers}, coef_lw_nc, n_char), {n_minorabsorbers});

    Array<std::string,1> minor_gases_lower(get_variable_string("minor_gases_lower", {n_minor_absorber_intervals_lower}, coef_lw_nc, n_char), {n_minor_absorber_intervals_lower});
    Array<std::string,1> minor_gases_upper(get_variable_string("minor_gases_upper", {n_minor_absorber_intervals_upper}, coef_lw_nc, n_char), {n_minor_absorber_intervals_upper});

    Array<int,2> minor_limits_gpt_lower(coef_lw_nc.get_variable<int>("minor_limits_gpt_lower", {n_minor_absorber_intervals_lower, n_pairs}), {n_pairs, n_minor_absorber_intervals_lower});
    Array<int,2> minor_limits_gpt_upper(coef_lw_nc.get_variable<int>("minor_limits_gpt_upper", {n_minor_absorber_intervals_upper, n_pairs}), {n_pairs, n_minor_absorber_intervals_upper});

    Array<int,1> minor_scales_with_density_lower(coef_lw_nc.get_variable<int>("minor_scales_with_density_lower", {n_minor_absorber_intervals_lower}), {n_minor_absorber_intervals_lower});
    Array<int,1> minor_scales_with_density_upper(coef_lw_nc.get_variable<int>("minor_scales_with_density_upper", {n_minor_absorber_intervals_upper}), {n_minor_absorber_intervals_upper});

    Array<int,1> scale_by_complement_lower(coef_lw_nc.get_variable<int>("scale_by_complement_lower", {n_minor_absorber_intervals_lower}), {n_minor_absorber_intervals_lower});
    Array<int,1> scale_by_complement_upper(coef_lw_nc.get_variable<int>("scale_by_complement_upper", {n_minor_absorber_intervals_upper}), {n_minor_absorber_intervals_upper});

    Array<std::string,1> scaling_gas_lower(get_variable_string("scaling_gas_lower", {n_minor_absorber_intervals_lower}, coef_lw_nc, n_char), {n_minor_absorber_intervals_lower});
    Array<std::string,1> scaling_gas_upper(get_variable_string("scaling_gas_upper", {n_minor_absorber_intervals_upper}, coef_lw_nc, n_char), {n_minor_absorber_intervals_upper});

    Array<int,1> kminor_start_lower(coef_lw_nc.get_variable<int>("kminor_start_lower", {n_minor_absorber_intervals_lower}), {n_minor_absorber_intervals_lower});
    Array<int,1> kminor_start_upper(coef_lw_nc.get_variable<int>("kminor_start_upper", {n_minor_absorber_intervals_upper}), {n_minor_absorber_intervals_upper});

    Array<double,3> vmr_ref(coef_lw_nc.get_variable<double>("vmr_ref", {n_temps, n_extabsorbers, n_layers}), {n_layers, n_extabsorbers, n_temps});

    Array<double,4> kmajor(coef_lw_nc.get_variable<double>("kmajor", {n_temps, n_press+1, n_mixingfracs, n_gpts}), {n_gpts, n_mixingfracs, n_press+1, n_temps});

    Array<double,3> rayl_lower({n_gpts, n_mixingfracs, n_temps});
    Array<double,3> rayl_upper({n_gpts, n_mixingfracs, n_temps});

    if (coef_lw_nc.variable_exists("rayl_lower"))
    {
        throw std::runtime_error("rayl reading not implemented!");
        // rayl_lower = read_field(ncid, 'rayl_lower', ngpts, nmixingfracs, ntemps)
        // rayl_upper = read_field(ncid, 'rayl_upper', ngpts, nmixingfracs, ntemps)
    }

    // Is it really LW if so read these variables as well.
    Array<double,2> totplnk({n_internal_sourcetemps, n_bnds});
    Array<double,4> planck_frac({n_gpts, n_mixingfracs, n_press+1, n_temps});

    if (coef_lw_nc.variable_exists("totplnk"))
    {
        totplnk = coef_lw_nc.get_variable<double>("totplnk", {n_bnds, n_internal_sourcetemps});
        planck_frac = coef_lw_nc.get_variable<double>("plank_fraction", {n_temps, n_press+1, n_mixingfracs, n_gpts});
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

    int n_gpt = kdist.get_ngpt();
    int n_bnd = kdist.get_nband();

    Array<double,2> emis_sfc;
    Array<double,1> t_sfc;

    const int n_col_block = 4;

    std::unique_ptr<Optical_props_arry<double>> optical_props;
    std::unique_ptr<Optical_props_arry<double>> optical_props_subset;

    if (kdist.source_is_internal())
    {
        master.print_message("Computing optical depths for longwave radiation\n");

        Source_func_lw<double> sources       (n_col      , n_lay, kdist);
        Source_func_lw<double> sources_subset(n_col_block, n_lay, kdist);
        optical_props = std::make_unique<Optical_props_1scl<double>>(n_col, n_lay, kdist);
        optical_props_subset = std::make_unique<Optical_props_1scl<double>>(n_col_block, n_lay, kdist);

        // Download surface boundary conditions for long wave.
        Array<double,2> emis_sfc_tmp(input_nc.get_variable<double>("emis_sfc", {n_col, n_bnd}), {n_bnd, n_col});
        Array<double,1> t_sfc_tmp(input_nc.get_variable<double>("t_sfc", {n_col}), {n_col});

        emis_sfc = emis_sfc_tmp;
        t_sfc = t_sfc_tmp;

        int n_blocks = n_col / n_col_block;
        for (int b=1; b<=n_col_block; ++b)
        {
            const int col_s = (b-1) * n_col_block + 1;
            const int col_e = b     * n_col_block;

            Gas_concs<double> gas_concs_subset(gas_concs, col_s, n_col_block);

            kdist.gas_optics(
                    p_lay.subset({{ {col_s, col_e}, {1, n_lay} }}),
                    p_lev.subset({{ {col_s, col_e}, {1, n_lev} }}),
                    t_lay.subset({{ {col_s, col_e}, {1, n_lay} }}),
                    t_sfc.subset({{ {col_s, col_e} }}),
                    gas_concs_subset,
                    optical_props_subset,
                    sources_subset,
                    col_dry.subset({{ {col_s, col_e}, {1, n_lay} }}),
                    t_lev.subset  ({{ {col_s, col_e}, {1, n_lev} }})
                    );

            optical_props->set_subset(optical_props_subset, col_s, col_e);
        }
    }
    else
    {
        master.print_message("Computing optical depths for shortwave radiation\n");
        throw std::runtime_error("Shortwave radiation not implemented");
    }

    return 0;
}
