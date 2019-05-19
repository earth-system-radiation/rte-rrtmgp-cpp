#include <boost/algorithm/string.hpp>
#include <cmath>

#include "Netcdf_interface.h"
#include "Array.h"
#include "Gas_concs.h"
#include "Gas_optics.h"
#include "Optical_props.h"
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

    template<typename TF>
    Gas_optics<TF> load_and_init_gas_optics(
            Master& master,
            const Gas_concs<TF>& gas_concs,
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
        Array<TF,2> band_lims(coef_nc.get_variable<TF>("bnd_limits_wavenumber", {n_bnds, 2}), {2, n_bnds});
        Array<int,2> band2gpt(coef_nc.get_variable<int>("bnd_limits_gpt", {n_bnds, 2}), {2, n_bnds});
        Array<TF,1> press_ref(coef_nc.get_variable<TF>("press_ref", {n_press}), {n_press});
        Array<TF,1> temp_ref(coef_nc.get_variable<TF>("temp_ref", {n_temps}), {n_temps});

        TF temp_ref_p = coef_nc.get_variable<TF>("absorption_coefficient_ref_P");
        TF temp_ref_t = coef_nc.get_variable<TF>("absorption_coefficient_ref_T");
        TF press_ref_trop = coef_nc.get_variable<TF>("press_ref_trop");

        Array<TF,3> kminor_lower(
                coef_nc.get_variable<TF>("kminor_lower", {n_temps, n_mixingfracs, n_contributors_lower}),
                {n_contributors_lower, n_mixingfracs, n_temps});
        Array<TF,3> kminor_upper(
                coef_nc.get_variable<TF>("kminor_upper", {n_temps, n_mixingfracs, n_contributors_upper}),
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

        Array<TF,3> vmr_ref(
                coef_nc.get_variable<TF>("vmr_ref", {n_temps, n_extabsorbers, n_layers}),
                {n_layers, n_extabsorbers, n_temps});

        Array<TF,4> kmajor(
                coef_nc.get_variable<TF>("kmajor", {n_temps, n_press+1, n_mixingfracs, n_gpts}),
                {n_gpts, n_mixingfracs, n_press+1, n_temps});

        // Keep the size at zero, if it does not exist.
        Array<TF,3> rayl_lower;
        Array<TF,3> rayl_upper;

        if (coef_nc.variable_exists("rayl_lower"))
        {
            rayl_lower.set_dims({n_gpts, n_mixingfracs, n_temps});
            rayl_upper.set_dims({n_gpts, n_mixingfracs, n_temps});
            rayl_lower = coef_nc.get_variable<TF>("rayl_lower", {n_temps, n_mixingfracs, n_gpts});
            rayl_upper = coef_nc.get_variable<TF>("rayl_upper", {n_temps, n_mixingfracs, n_gpts});
        }

        // Is it really LW if so read these variables as well.
        if (coef_nc.variable_exists("totplnk"))
        {
            int n_internal_sourcetemps = coef_nc.get_dimension_size("temperature_Planck");

            Array<TF,2> totplnk(
                    coef_nc.get_variable<TF>( "totplnk", {n_bnds, n_internal_sourcetemps}),
                    {n_internal_sourcetemps, n_bnds});
            Array<TF,4> planck_frac(
                    coef_nc.get_variable<TF>("plank_fraction", {n_temps, n_press+1, n_mixingfracs, n_gpts}),
                    {n_gpts, n_mixingfracs, n_press+1, n_temps});

            // Construct the k-distribution.
            return Gas_optics<TF>(
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
            Array<TF,1> solar_src(
                    coef_nc.get_variable<TF>("solar_source", {n_gpts}), {n_gpts});

            return Gas_optics<TF>(
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

template<typename TF>
void load_gas_concs(Gas_concs<TF>& gas_concs, Netcdf_file& input_nc)
{
    // This part is contained in the create
    Netcdf_group rad_nc = input_nc.get_group("radiation");

    const int n_lay = rad_nc.get_dimension_size("lay");

    gas_concs.set_vmr("h2o",
            Array<TF,1>(rad_nc.get_variable<TF>("h2o", {n_lay}), {n_lay}));
    gas_concs.set_vmr("co2",
            rad_nc.get_variable<TF>("co2"));
    gas_concs.set_vmr("o3",
            Array<TF,1>(rad_nc.get_variable<TF>("o3", {n_lay}), {n_lay}));
    gas_concs.set_vmr("n2o",
            rad_nc.get_variable<TF>("n2o"));
    gas_concs.set_vmr("ch4",
            rad_nc.get_variable<TF>("ch4"));
    gas_concs.set_vmr("o2",
            rad_nc.get_variable<TF>("o2"));
    gas_concs.set_vmr("n2",
            rad_nc.get_variable<TF>("n2"));
}

int main()
{
    Master master;
    try
    {
        master.start();
        master.init();

        // We are doing a single column run.
        const int n_col = 1;

        // These are the global variables that need to be contained in a class.
        Gas_concs<double> gas_concs;

        std::unique_ptr<Gas_optics<double>> kdist_lw;
        std::unique_ptr<Gas_optics<double>> kdist_sw;

        // This is the part that is done in the initialization.
        Netcdf_file file_nc(master, "test_rcemip_input.nc", Netcdf_mode::Read);

        load_gas_concs<double>(gas_concs, file_nc);
        kdist_lw = std::make_unique<Gas_optics<double>>(
                load_and_init_gas_optics(master, gas_concs, "coefficients_lw.nc"));
        kdist_sw = std::make_unique<Gas_optics<double>>(
                load_and_init_gas_optics(master, gas_concs, "coefficients_sw.nc"));

        // LOAD THE LONGWAVE SPECIFIC BOUNDARY CONDITIONS.
        // Set the surface temperature and emissivity.
        Array<double,1> t_sfc({1});
        t_sfc({1}) = 300.;

        const int n_bnd = kdist_lw->get_nband();
        Array<double,2> emis_sfc({n_bnd, 1});
        for (int ibnd=1; ibnd<=n_bnd; ++ibnd)
            emis_sfc({ibnd, 1}) = 1.;

        const int n_ang = 1;

        // LOAD THE SHORTWAVE SPECIFIC BOUNDARY CONDITIONS.
        Array<double,1> sza({n_col});
        Array<double,2> sfc_alb_dir({n_bnd, n_col});
        Array<double,2> sfc_alb_dif({n_bnd, n_col});

        sza({1}) = 0.7339109504636155;

        for (int ibnd=1; ibnd<=n_bnd; ++ibnd)
        {
            sfc_alb_dir({ibnd, 1}) = 0.07;
            sfc_alb_dif({ibnd, 1}) = 0.07;
        }

        Array<double,1> mu0({n_col});
        mu0({1}) = std::cos(sza({1}));

        // Solve the full column once.
        Netcdf_group input_nc = file_nc.get_group("radiation");

        const int n_lay = input_nc.get_dimension_size("lay");
        const int n_lev = input_nc.get_dimension_size("lev");

        Array<double,2> p_lay(input_nc.get_variable<double>("p_lay", {n_lay, n_col}), {n_col, n_lay});
        Array<double,2> t_lay(input_nc.get_variable<double>("t_lay", {n_lay, n_col}), {n_col, n_lay});
        Array<double,2> p_lev(input_nc.get_variable<double>("p_lev", {n_lev, n_col}), {n_col, n_lev});
        Array<double,2> t_lev(input_nc.get_variable<double>("t_lev", {n_lev, n_col}), {n_col, n_lev});

        Array<double,2> col_dry({n_col, n_lay});
        if (input_nc.variable_exists("col_dry"))
            col_dry = input_nc.get_variable<double>("col_dry", {n_lay, n_col});
        else
        {
            kdist_lw->get_col_dry(col_dry, gas_concs.get_vmr("h2o"), p_lev);
            kdist_sw->get_col_dry(col_dry, gas_concs.get_vmr("h2o"), p_lev);
        }

        // Solve the longwave first.
        std::unique_ptr<Optical_props_arry<double>> optical_props_lw =
                std::make_unique<Optical_props_1scl<double>>(n_col, n_lay, *kdist_lw);

        Source_func_lw<double> sources(n_col, n_lay, *kdist_lw);

        kdist_lw->gas_optics(
                p_lay,
                p_lev,
                t_lay,
                t_sfc,
                gas_concs,
                optical_props_lw,
                sources,
                col_dry,
                t_lev);

        std::unique_ptr<Fluxes_broadband<double>> fluxes =
                std::make_unique<Fluxes_broadband<double>>(n_col, n_lev);

        const int top_at_1 = p_lay({1, 1}) < p_lay({1, n_lay});

        const int n_gpt_lw = optical_props_lw->get_ngpt();
        Array<double,3> lw_gpt_flux_up({n_col, n_lev, n_gpt_lw});
        Array<double,3> lw_gpt_flux_dn({n_col, n_lev, n_gpt_lw});

        Rte_lw<double>::rte_lw(
                optical_props_lw,
                top_at_1,
                sources,
                emis_sfc,
                lw_gpt_flux_up,
                lw_gpt_flux_dn,
                n_ang);

        fluxes->reduce(
                lw_gpt_flux_up, lw_gpt_flux_dn,
                optical_props_lw, top_at_1);

        Array<double,2> lw_flux_up ({n_col, n_lev});
        Array<double,2> lw_flux_dn ({n_col, n_lev});
        Array<double,2> lw_flux_net({n_col, n_lev});
        Array<double,2> lw_heating ({n_col, n_lay});

        // Copy the data to the output.
        for (int ilev=1; ilev<=n_lev; ++ilev)
        {
            lw_flux_up ({1, ilev}) = fluxes->get_flux_up ()({1, ilev});
            lw_flux_dn ({1, ilev}) = fluxes->get_flux_dn ()({1, ilev});
            lw_flux_net({1, ilev}) = fluxes->get_flux_net()({1, ilev});
        }

        const int n_gpt_sw = kdist_sw->get_ngpt();
        Array<double,2> toa_src({n_col, n_gpt_sw});

        std::unique_ptr<Optical_props_arry<double>> optical_props_sw =
                std::make_unique<Optical_props_2str<double>>(n_col, n_lay, *kdist_sw);

        kdist_sw->gas_optics(
                p_lay,
                p_lev,
                t_lay,
                gas_concs,
                optical_props_sw,
                toa_src,
                col_dry);

        const double tsi_scaling = 0.4053176301654965;
        for (int igpt=1; igpt<=n_gpt_sw; ++igpt)
            toa_src({1, igpt}) *= tsi_scaling;

        Array<double,3> sw_gpt_flux_up    ({n_col, n_lev, n_gpt_sw});
        Array<double,3> sw_gpt_flux_dn    ({n_col, n_lev, n_gpt_sw});
        Array<double,3> sw_gpt_flux_dn_dir({n_col, n_lev, n_gpt_sw});

        Rte_sw<double>::rte_sw(
                optical_props_sw,
                top_at_1,
                mu0,
                toa_src,
                sfc_alb_dir,
                sfc_alb_dif,
                sw_gpt_flux_up,
                sw_gpt_flux_dn,
                sw_gpt_flux_dn_dir);

        fluxes->reduce(
                sw_gpt_flux_up, sw_gpt_flux_dn, sw_gpt_flux_dn_dir,
                optical_props_sw, top_at_1);

        Array<double,2> sw_flux_up ({n_col, n_lev});
        Array<double,2> sw_flux_dn ({n_col, n_lev});
        Array<double,2> sw_flux_net({n_col, n_lev});
        Array<double,2> sw_heating ({n_col, n_lay});

        // Copy the data to the output.
        for (int ilev=1; ilev<=n_lev; ++ilev)
        {
            sw_flux_up ({1, ilev}) = fluxes->get_flux_up ()({1, ilev});
            sw_flux_dn ({1, ilev}) = fluxes->get_flux_dn ()({1, ilev});
            sw_flux_net({1, ilev}) = fluxes->get_flux_net()({1, ilev});
        }

        // Compute the heating rates.
        constexpr double g = 9.80655;
        constexpr double cp = 1005.;

        Array<double,2> heating ({n_col, n_lay});

        for (int ilay=1; ilay<=n_lay; ++ilay)
        {
            lw_heating({1, ilay}) =
                    ( lw_flux_up({1, ilay+1}) - lw_flux_up({1, ilay})
                    - lw_flux_dn({1, ilay+1}) + lw_flux_dn({1, ilay}) )
                    * g / ( cp * (p_lev({1, ilay+1}) - p_lev({1, ilay})) ) * 86400.;

            sw_heating({1, ilay}) =
                    ( sw_flux_up({1, ilay+1}) - sw_flux_up({1, ilay})
                    - sw_flux_dn({1, ilay+1}) + sw_flux_dn({1, ilay}) )
                    * g / ( cp * (p_lev({1, ilay+1}) - p_lev({1, ilay})) ) * 86400.;

            heating({1, ilay}) = lw_heating({1, ilay}) + sw_heating({1, ilay});
        }

        // Store the radiation fluxes to a file
        Netcdf_file output_nc(master, "test_rcemip_output.nc", Netcdf_mode::Create);
        output_nc.add_dimension("col", n_col);
        output_nc.add_dimension("lev", n_lev);
        output_nc.add_dimension("lay", n_lay);

        auto nc_p_lev = output_nc.add_variable<double>("lev", {"lev"});
        auto nc_p_lay = output_nc.add_variable<double>("lay", {"lay"});
        nc_p_lev.insert(p_lev.v(), {0});
        nc_p_lay.insert(p_lay.v(), {0});

        auto nc_lw_flux_up  = output_nc.add_variable<double>("lw_flux_up" , {"lev", "col"});
        auto nc_lw_flux_dn  = output_nc.add_variable<double>("lw_flux_dn" , {"lev", "col"});
        auto nc_lw_flux_net = output_nc.add_variable<double>("lw_flux_net", {"lev", "col"});
        auto nc_lw_heating  = output_nc.add_variable<double>("lw_heating" , {"lay", "col"});

        nc_lw_flux_up .insert(lw_flux_up .v(), {0, 0});
        nc_lw_flux_dn .insert(lw_flux_dn .v(), {0, 0});
        nc_lw_flux_net.insert(lw_flux_net.v(), {0, 0});
        nc_lw_heating .insert(lw_heating .v(), {0, 0});

        auto nc_sw_flux_up  = output_nc.add_variable<double>("sw_flux_up" , {"lev", "col"});
        auto nc_sw_flux_dn  = output_nc.add_variable<double>("sw_flux_dn" , {"lev", "col"});
        auto nc_sw_flux_net = output_nc.add_variable<double>("sw_flux_net", {"lev", "col"});
        auto nc_sw_heating  = output_nc.add_variable<double>("sw_heating" , {"lay", "col"});

        nc_sw_flux_up .insert(sw_flux_up .v(), {0, 0});
        nc_sw_flux_dn .insert(sw_flux_dn .v(), {0, 0});
        nc_sw_flux_net.insert(sw_flux_net.v(), {0, 0});
        nc_sw_heating .insert(sw_heating .v(), {0, 0});

        auto nc_heating = output_nc.add_variable<double>("heating", {"lay", "col"});
        nc_heating.insert(heating.v(), {0, 0});
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
