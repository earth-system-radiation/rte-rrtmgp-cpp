#ifndef GAS_OPTICS_H
#define GAS_OPTICS_H

#include <cmath>
#include "Optical_props.h"
#include "Array.h"
#include "Source_functions.h"

template<typename TF> class Gas_concs;

template<typename TF>
class Gas_optics : public Optical_props<TF>
{
    public:
        Gas_optics(
                Gas_concs<TF>& available_gases,
                Array<std::string,1>& gas_names,
                Array<int,3>& key_species,
                Array<int,2>& band2gpt,
                Array<TF,2>& band_lims_wavenum,
                Array<TF,1>& press_ref,
                TF press_ref_trop,
                Array<TF,1>& temp_ref,
                TF temp_ref_p,
                TF temp_ref_t,
                Array<TF,3>& vmr_ref,
                Array<TF,4>& kmajor,
                Array<TF,3>& kminor_lower,
                Array<TF,3>& kminor_upper,
                Array<std::string,1>& gas_minor,
                Array<std::string,1>& identifier_minor,
                Array<std::string,1>& minor_gases_lower,
                Array<std::string,1>& minor_gases_upper,
                Array<int,2>& minor_limits_gpt_lower,
                Array<int,2>& minor_limits_gpt_upper,
                Array<int,1>& minor_scales_with_density_lower,
                Array<int,1>& minor_scales_with_density_upper,
                Array<std::string,1>& scaling_gas_lower,
                Array<std::string,1>& scaling_gas_upper,
                Array<int,1>& scale_by_complement_lower,
                Array<int,1>& scale_by_complement_upper,
                Array<int,1>& kminor_start_lower,
                Array<int,1>& kminor_start_upper,
                Array<TF,2>& totplnk,
                Array<TF,4>& planck_frac,
                Array<TF,3>& rayl_lower,
                Array<TF,3>& rayl_upper);

        bool source_is_internal() const { return (totplnk.size() > 0) && (planck_frac.size() > 0); }
        TF get_press_ref_min() const { return press_ref_min; }

        int get_nflav() const { return flavor.dim(2); }
        int get_neta() const { return kmajor.dim(2); }
        int get_npres() const { return kmajor.dim(3)-1; }
        int get_ntemp() const { return kmajor.dim(4); }

        void gas_optics(
                const Array<TF,2>& play,
                const Array<TF,2>& plev,
                const Array<TF,2>& tlay,
                const Array<TF,1>& tsfc,
                const Gas_concs<TF>& gas_desc,
                std::unique_ptr<Optical_props_arry<TF>>& optical_props,
                Source_func_lw<TF>& sources,
                const Array<TF,2>& col_dry,
                const Array<TF,2>& tlev);

    void combine_and_reorder(
            const Array<TF,3>& tau,
            const Array<TF,3>& tau_rayleigh,
            const bool has_rayleigh,
            std::unique_ptr<Optical_props_arry<TF>>& optical_props);

    void source(
            const int ncol, const int nlay, const int nband, const int ngpt,
            const Array<TF,2>& play, const Array<TF,2>& plev,
            const Array<TF,2>& tlay, const Array<TF,1>& tsfc,
            const Array<int,2>& jtemp, const Array<int,2>& jpress,
            const Array<int,4>& jeta, const Array<int,2>& tropo,
            const Array<TF,6>& fmajor,
            Source_func_lw<TF>& sources,
            const Array<TF,2>& tlev);

    private:
        Array<TF,2> totplnk;
        Array<TF,4> planck_frac;
        TF totplnk_delta;
        TF temp_ref_min, temp_ref_max;
        TF press_ref_min, press_ref_max;
        TF press_ref_trop_log;

        TF press_ref_log_delta;
        TF temp_ref_delta;

        Array<TF,1> press_ref, press_ref_log, temp_ref;

        Array<std::string,1> gas_names;

        Array<TF,3> vmr_ref;

        Array<int,2> flavor;
        Array<int,2> gpoint_flavor;

        Array<TF,4> kmajor;

        Array<TF,3> kminor_lower;
        Array<TF,3> kminor_upper;

        Array<int,2> minor_limits_gpt_lower;
        Array<int,2> minor_limits_gpt_upper;

        Array<int,1> minor_scales_with_density_lower;
        Array<int,1> minor_scales_with_density_upper;

        Array<int,1> scale_by_complement_lower;
        Array<int,1> scale_by_complement_upper;

        Array<int,1> kminor_start_lower;
        Array<int,1> kminor_start_upper;

        Array<int,1> idx_minor_lower;
        Array<int,1> idx_minor_upper;

        Array<int,1> idx_minor_scaling_lower;
        Array<int,1> idx_minor_scaling_upper;

        Array<int,1> is_key;

        int get_ngas() const { return this->gas_names.dim(1); }

        void init_abs_coeffs(
                const Gas_concs<TF>& available_gases,
                const Array<std::string,1>& gas_names,
                const Array<int,3>& key_species,
                const Array<int,2>& band2gpt,
                const Array<TF,2>& band_lims_wavenum,
                const Array<TF,1>& press_ref,
                const Array<TF,1>& temp_ref,
                const TF press_ref_trop,
                const TF temp_ref_p,
                const TF temp_ref_t,
                const Array<TF,3>& vmr_ref,
                const Array<TF,4>& kmajor,
                const Array<TF,3>& kminor_lower,
                const Array<TF,3>& kminor_upper,
                const Array<std::string,1>& gas_minor,
                const Array<std::string,1>& identifier_minor,
                const Array<std::string,1>& minor_gases_lower,
                const Array<std::string,1>& minor_gases_upper,
                const Array<int,2>& minor_limits_gpt_lower,
                const Array<int,2>& minor_limits_gpt_upper,
                const Array<int,1>& minor_scales_with_density_lower,
                const Array<int,1>& minor_scales_with_density_upper,
                const Array<std::string,1>& scaling_gas_lower,
                const Array<std::string,1>& scaling_gas_upper,
                const Array<int,1>& scale_by_complement_lower,
                const Array<int,1>& scale_by_complement_upper,
                const Array<int,1>& kminor_start_lower,
                const Array<int,1>& kminor_start_upper,
                const Array<TF,3>& rayl_lower,
                const Array<TF,3>& rayl_upper);

        void compute_gas_taus(
                const int ncol, const int nlay, const int ngpt, const int nband,
                const Array<TF,2>& play,
                const Array<TF,2>& plev,
                const Array<TF,2>& tlay,
                const Gas_concs<TF>& gas_desc,
                std::unique_ptr<Optical_props_arry<TF>>& optical_props,
                Array<int,2>& jtemp, Array<int,2>& jpress,
                Array<int,4>& jeta,
                Array<int,2>& tropo,
                Array<TF,6>& fmajor,
                const Array<TF,2>& col_dry);
};

namespace
{
    int find_index(
            const Array<std::string,1>& data, const std::string& value)
    {
        auto it = std::find(data.v().begin(), data.v().end(), value);
        if (it == data.v().end())
            return -1;
        else
            return it - data.v().begin() + 1;
    }

    template<typename TF>
    void reduce_minor_arrays(
                const Gas_concs<TF>& available_gases,
                const Array<std::string,1>& gas_names,
                const Array<std::string,1>& gas_minor,
                const Array<std::string,1>& identifier_minor,
                const Array<TF,3>& kminor_atm,
                const Array<std::string,1>& minor_gases_atm,
                const Array<int,2>& minor_limits_gpt_atm,
                const Array<int,1>& minor_scales_with_density_atm, // CvH: logical bool or int?
                const Array<std::string,1>& scaling_gas_atm,
                const Array<int,1>& scale_by_complement_atm, // CvH: bool or int
                const Array<int,1>& kminor_start_atm,

                Array<TF,3>& kminor_atm_red,
                Array<std::string,1>& minor_gases_atm_red,
                Array<int,2>& minor_limits_gpt_atm_red,
                Array<int,1>& minor_scales_with_density_atm_red, // CvH bool or int
                Array<std::string,1>& scaling_gas_atm_red,
                Array<int,1>& scale_by_complement_atm_red,
                Array<int,1>& kminor_start_atm_red)
    {
        int nm = minor_gases_atm.dim(1);
        int tot_g = 0;

        Array<int,1> gas_is_present({nm});

        for (int i=1; i<=nm; ++i)
        {
            const int idx_mnr = find_index(identifier_minor, minor_gases_atm({i}));

            // Search for
            std::string gas_minor_trimmed = gas_minor({idx_mnr});
            boost::trim(gas_minor_trimmed);

            gas_is_present({i}) = available_gases.exists(gas_minor_trimmed);
            if (gas_is_present({i}))
                tot_g += minor_limits_gpt_atm({2,i}) - minor_limits_gpt_atm({1,i}) + 1;
        }

        const int red_nm = std::accumulate(gas_is_present.v().begin(), gas_is_present.v().end(), 0);

        if (red_nm == nm)
        {
            kminor_atm_red = kminor_atm;
            minor_gases_atm_red = minor_gases_atm;
            minor_limits_gpt_atm_red = minor_limits_gpt_atm;
            minor_scales_with_density_atm_red = minor_scales_with_density_atm;
            scaling_gas_atm_red = scaling_gas_atm;
            scale_by_complement_atm_red = scale_by_complement_atm;
            kminor_start_atm_red = kminor_start_atm;
        }
        else
        {
            // Use a lambda function as the operation has to be repeated many times.
            auto resize_and_set = [=](auto& a_red, const auto& a)
            {
                a_red.set_dims({red_nm});
                int counter = 1;
                for (int i=1; i<=gas_is_present.dim(1); ++i)
                {
                    if (gas_is_present({i}))
                    {
                       a_red({counter}) = a({i});
                       ++counter;
                    }
                }
            };

            resize_and_set(minor_gases_atm_red, minor_gases_atm);
            resize_and_set(minor_scales_with_density_atm_red, minor_scales_with_density_atm);
            resize_and_set(scaling_gas_atm_red, scaling_gas_atm);
            resize_and_set(scale_by_complement_atm_red, scale_by_complement_atm);
            resize_and_set(kminor_start_atm_red, kminor_start_atm);

            minor_limits_gpt_atm_red.set_dims({2, red_nm});
            kminor_atm_red.set_dims({tot_g, kminor_atm.dim(2), kminor_atm.dim(3)});

            int icnt = 0;
            int n_elim = 0;
            for (int i=1; i<=nm; ++i)
            {
                int ng = minor_limits_gpt_atm({2,i}) - minor_limits_gpt_atm({1,i}) + 1;
                if (gas_is_present({i}))
                {
                    ++icnt;
                    minor_limits_gpt_atm_red({1,icnt}) = minor_limits_gpt_atm({1,i});
                    minor_limits_gpt_atm_red({2,icnt}) = minor_limits_gpt_atm({2,i});
                    kminor_start_atm_red({icnt}) = kminor_start_atm({i}) - n_elim;

                    for (int j=1; j<=ng; ++j)
                        for (int i2=1; i2<=kminor_atm.dim(2); ++i2)
                            for (int i3=1; i3<=kminor_atm.dim(3); ++i3)
                                kminor_atm_red({kminor_start_atm_red({icnt})+j-1,i2,i3}) =
                                        kminor_atm({kminor_start_atm({i})+j-1,i2,i3});
                }
                else
                    n_elim += ng;
            }
        }
    }

    void create_idx_minor(
            const Array<std::string,1>& gas_names,
            const Array<std::string,1>& gas_minor,
            const Array<std::string,1>& identifier_minor,
            const Array<std::string,1>& minor_gases_atm,
            Array<int,1>& idx_minor_atm)
    {
        Array<int,1> idx_minor_atm_out({minor_gases_atm.dim(1)});

        for (int imnr=1; imnr<=minor_gases_atm.dim(1); ++imnr)
        {
            // Find identifying string for minor species in list of possible identifiers (e.g. h2o_slf)
            const int idx_mnr = find_index(identifier_minor, minor_gases_atm({imnr}));

            // Find name of gas associated with minor species identifier (e.g. h2o)
            idx_minor_atm_out({imnr}) = find_index(gas_names, gas_minor({idx_mnr}));
        }

        idx_minor_atm = idx_minor_atm_out;
    }

    void create_idx_minor_scaling(
            const Array<std::string,1>& gas_names,
            const Array<std::string,1>& scaling_gas_atm,
            Array<int,1>& idx_minor_scaling_atm)
    {
        Array<int,1> idx_minor_scaling_atm_out({scaling_gas_atm.dim(1)});

        for (int imnr=1; imnr<=scaling_gas_atm.dim(1); ++imnr)
            idx_minor_scaling_atm_out({imnr}) = find_index(gas_names, scaling_gas_atm({imnr}));

        idx_minor_scaling_atm = idx_minor_scaling_atm_out;
    }

    void create_key_species_reduce(
            const Array<std::string,1>& gas_names,
            const Array<std::string,1>& gas_names_red,
            const Array<int,3>& key_species,
            Array<int,3>& key_species_red,
            Array<int,1>& key_species_present_init)
    {
        const int np = key_species.dim(1);
        const int na = key_species.dim(2);
        const int nt = key_species.dim(3);

        key_species_red.set_dims({key_species.dim(1), key_species.dim(2), key_species.dim(3)});
        key_species_present_init.set_dims({gas_names.dim(1)});

        for (int i=1; i<=key_species_present_init.dim(1); ++i)
            key_species_present_init({i}) = 1;

        for (int ip=1; ip<=np; ++ip)
            for (int ia=1; ia<=na; ++ia)
                for (int it=1; it<=nt; ++it)
                {
                    const int ks = key_species({ip,ia,it});
                    if (ks != 0)
                    {
                        key_species_red({ip,ia,it}) = find_index(gas_names_red, gas_names({ks}));
                        if (ks == -1)
                            key_species_present_init({ks}) = 0;
                    }
                    else
                        key_species_red({ip,ia,it}) = ks;
                }
    }

    void check_key_species_present_init(
            const Array<std::string,1>& gas_names,
            const Array<int,1>& key_species_present_init
            )
    {
        for (int i=1; i<=key_species_present_init.dim(1); ++i)
        {
            if (key_species_present_init({i}) == 0)
            {
                std::string error_message = "Gas optics: required gas " + gas_names({i}) + " is missing";
                throw std::runtime_error(error_message);
            }
        }
    }

    void create_flavor(
            const Array<int,3>& key_species,
            Array<int,2>& flavor)
    {
        Array<int,2> key_species_list({2, key_species.dim(3)*2});

        // Prepare list of key species.
        int i = 1;
        for (int ibnd=1; ibnd<=key_species.dim(3); ++ibnd)
            for (int iatm=1; iatm<=key_species.dim(1); ++iatm)
            {
                key_species_list({1,i}) = key_species({1,iatm,ibnd});
                key_species_list({2,i}) = key_species({2,iatm,ibnd});
                ++i;
            }

        // Rewrite single key_species pairs.
        for (int i=1; i<=key_species_list.dim(2); ++i)
        {
            if ( key_species_list({1,i}) == 0 && key_species_list({2,i}) == 0 )
            {
                key_species_list({1,i}) = 2;
                key_species_list({2,i}) = 2;
            }
        }

        // Count unique key species pairs.
        int iflavor = 0;
        for (int i=1; i<=key_species_list.dim(2); ++i)
        {
            bool pair_exists = false;
            for (int ii=1; ii<=i-1; ++ii)
            {
                if ( (key_species_list({1,i}) == key_species_list({1,ii})) &&
                     (key_species_list({2,i}) == key_species_list({2,ii})) )
                {
                    pair_exists = true;
                    break;
                }
            }
            if (!pair_exists)
                ++iflavor;
        }

        // Fill flavors.
        flavor.set_dims({2,iflavor});
        iflavor = 0;
        for (int i=1; i<=key_species_list.dim(2); ++i)
        {
            bool pair_exists = false;
            for (int ii=1; ii<=i-1; ++ii)
            {
                if ( (key_species_list({1,i}) == key_species_list({1,ii})) &&
                     (key_species_list({2,i}) == key_species_list({2,ii})) )
                {
                    pair_exists = true;
                    break;
                }
            }
            if (!pair_exists)
            {
                ++iflavor;
                flavor({1,iflavor}) = key_species_list({1,i});
                flavor({2,iflavor}) = key_species_list({2,i});
            }
        }
    }

    int key_species_pair2flavor(
            const Array<int,2>& flavor,
            const Array<int,1>& key_species_pair)
    {
        // Search for match.
        for (int iflav=1; iflav<=flavor.dim(2); ++iflav)
        {
            if ( key_species_pair({1}) == flavor({1, iflav}) &&
                 key_species_pair({2}) == flavor({2, iflav}) )
                return iflav;
        }

        // No match found.
        return -1;
    }

    void create_gpoint_flavor(
            const Array<int,3>& key_species,
            const Array<int,1>& gpt2band,
            const Array<int,2>& flavor,
            Array<int,2>& gpoint_flavor)
    {
        const int ngpt = gpt2band.dim(1);
        gpoint_flavor.set_dims({2,ngpt});

        for (int igpt=1; igpt<=ngpt; ++igpt)
            for (int iatm=1; iatm<=2; ++iatm)
            {
                int pair_1 = key_species( {1, iatm, gpt2band({igpt})} );
                int pair_2 = key_species( {2, iatm, gpt2band({igpt})} );

                // Rewrite species pair.
                Array<int,1> rewritten_pair({2});
                if ( pair_1 == 0 && pair_2 == 0)
                {
                    rewritten_pair({1}) = 2;
                    rewritten_pair({2}) = 2;
                }
                else
                {
                    rewritten_pair({1}) = pair_1;
                    rewritten_pair({2}) = pair_2;
                }

                // Write the output.
                gpoint_flavor({iatm,igpt}) = key_species_pair2flavor(
                        flavor, rewritten_pair);
            }
    }
}

template<typename TF>
Gas_optics<TF>::Gas_optics(
        Gas_concs<TF>& available_gases,
        Array<std::string,1>& gas_names,
        Array<int,3>& key_species,
        Array<int,2>& band2gpt,
        Array<TF,2>& band_lims_wavenum,
        Array<TF,1>& press_ref,
        TF press_ref_trop,
        Array<TF,1>& temp_ref,
        TF temp_ref_p,
        TF temp_ref_t,
        Array<TF,3>& vmr_ref,
        Array<TF,4>& kmajor,
        Array<TF,3>& kminor_lower,
        Array<TF,3>& kminor_upper,
        Array<std::string,1>& gas_minor,
        Array<std::string,1>& identifier_minor,
        Array<std::string,1>& minor_gases_lower,
        Array<std::string,1>& minor_gases_upper,
        Array<int,2>& minor_limits_gpt_lower,
        Array<int,2>& minor_limits_gpt_upper,
        Array<int,1>& minor_scales_with_density_lower,
        Array<int,1>& minor_scales_with_density_upper,
        Array<std::string,1>& scaling_gas_lower,
        Array<std::string,1>& scaling_gas_upper,
        Array<int,1>& scale_by_complement_lower,
        Array<int,1>& scale_by_complement_upper,
        Array<int,1>& kminor_start_lower,
        Array<int,1>& kminor_start_upper,
        Array<TF,2>& totplnk,
        Array<TF,4>& planck_frac,
        Array<TF,3>& rayl_lower,
        Array<TF,3>& rayl_upper) :
            Optical_props<TF>(band_lims_wavenum, band2gpt),
            totplnk(totplnk),
            planck_frac(planck_frac)
{
    // Temperature steps for Planck function interpolation.
    // Assumes that temperature minimum and max are the same for the absorption coefficient grid and the
    // Planck grid and the Planck grid is equally spaced.
    totplnk_delta = (temp_ref_max - temp_ref_min) / (totplnk.dim(1)-1);

    // Initialize the absorption coefficient array, including Rayleigh scattering
    // tables if provided.
    init_abs_coeffs(
            available_gases,
            gas_names, key_species,
            band2gpt, band_lims_wavenum,
            press_ref, temp_ref,
            press_ref_trop, temp_ref_p, temp_ref_t,
            vmr_ref,
            kmajor, kminor_lower, kminor_upper,
            gas_minor,identifier_minor,
            minor_gases_lower, minor_gases_upper,
            minor_limits_gpt_lower,
            minor_limits_gpt_upper,
            minor_scales_with_density_lower,
            minor_scales_with_density_upper,
            scaling_gas_lower, scaling_gas_upper,
            scale_by_complement_lower,
            scale_by_complement_upper,
            kminor_start_lower,
            kminor_start_upper,
            rayl_lower, rayl_upper);
}

template<typename TF>
void Gas_optics<TF>::init_abs_coeffs(
        const Gas_concs<TF>& available_gases,
        const Array<std::string,1>& gas_names,
        const Array<int,3>& key_species,
        const Array<int,2>& band2gpt,
        const Array<TF,2>& band_lims_wavenum,
        const Array<TF,1>& press_ref,
        const Array<TF,1>& temp_ref,
        const TF press_ref_trop,
        const TF temp_ref_p,
        const TF temp_ref_t,
        const Array<TF,3>& vmr_ref,
        const Array<TF,4>& kmajor,
        const Array<TF,3>& kminor_lower,
        const Array<TF,3>& kminor_upper,
        const Array<std::string,1>& gas_minor,
        const Array<std::string,1>& identifier_minor,
        const Array<std::string,1>& minor_gases_lower,
        const Array<std::string,1>& minor_gases_upper,
        const Array<int,2>& minor_limits_gpt_lower,
        const Array<int,2>& minor_limits_gpt_upper,
        const Array<int,1>& minor_scales_with_density_lower,
        const Array<int,1>& minor_scales_with_density_upper,
        const Array<std::string,1>& scaling_gas_lower,
        const Array<std::string,1>& scaling_gas_upper,
        const Array<int,1>& scale_by_complement_lower,
        const Array<int,1>& scale_by_complement_upper,
        const Array<int,1>& kminor_start_lower,
        const Array<int,1>& kminor_start_upper,
        const Array<TF,3>& rayl_lower,
        const Array<TF,3>& rayl_upper)
{
    // Which gases known to the gas optics are present in the host model (available_gases)?
    std::vector<std::string> gas_names_to_use;

    for (const std::string &s : gas_names.v())
    {
        if (available_gases.exists(s))
            gas_names_to_use.push_back(s);
    }

    // Now the number of gases is the union of those known to the k-distribution and provided
    // by the host model.
    const int n_gas = gas_names_to_use.size();
    Array<std::string, 1> gas_names_this(std::move(gas_names_to_use), {n_gas});
    this->gas_names = gas_names_this;

    // Initialize the gas optics object, keeping only those gases known to the
    // gas optics and also present in the host model.
    // Add an offset to the indexing to interface the negative ranging of fortran.
    Array<TF, 3> vmr_ref_red({vmr_ref.dim(1), n_gas + 1, vmr_ref.dim(3)});
    vmr_ref_red.set_offsets({0, -1, 0});

    // Gas 0 is used in single-key species method, set to 1.0 (col_dry)
    for (int i1 = 1; i1 <= vmr_ref_red.dim(1); ++i1)
        for (int i3 = 1; i3 <= vmr_ref_red.dim(3); ++i3)
            vmr_ref_red({i1, 0, i3}) = vmr_ref({i1, 1, i3});

    for (int i = 1; i <= n_gas; ++i)
    {
        int idx = find_index(gas_names, this->gas_names({i}));
        for (int i1 = 1; i1 <= vmr_ref_red.dim(1); ++i1)
            for (int i3 = 1; i3 <= vmr_ref_red.dim(3); ++i3)
                vmr_ref_red({i1, i, i3}) = vmr_ref({i1, idx + 1, i3}); // CvH: why +1?
    }

    this->vmr_ref = std::move(vmr_ref_red);

    // Reduce minor arrays so variables only contain minor gases that are available.
    // Reduce size of minor Arrays.
    Array<std::string, 1> minor_gases_lower_red;
    Array<std::string, 1> scaling_gas_lower_red;
    Array<std::string, 1> minor_gases_upper_red;
    Array<std::string, 1> scaling_gas_upper_red;

    reduce_minor_arrays(
            available_gases,
            gas_names,
            gas_minor, identifier_minor,
            kminor_lower,
            minor_gases_lower,
            minor_limits_gpt_lower,
            minor_scales_with_density_lower,
            scaling_gas_lower,
            scale_by_complement_lower,
            kminor_start_lower,
            this->kminor_lower,
            minor_gases_lower_red,
            this->minor_limits_gpt_lower,
            this->minor_scales_with_density_lower,
            scaling_gas_lower_red,
            this->scale_by_complement_lower,
            this->kminor_start_lower);

    reduce_minor_arrays(
            available_gases,
            gas_names,
            gas_minor,
            identifier_minor,
            kminor_upper,
            minor_gases_upper,
            minor_limits_gpt_upper,
            minor_scales_with_density_upper,
            scaling_gas_upper,
            scale_by_complement_upper,
            kminor_start_upper,
            this->kminor_upper,
            minor_gases_upper_red,
            this->minor_limits_gpt_upper,
            this->minor_scales_with_density_upper,
            scaling_gas_upper_red,
            this->scale_by_complement_upper,
            this->kminor_start_upper);

    // Arrays not reduced by the presence, or lack thereof, of a gas
    this->press_ref = press_ref;
    this->temp_ref = temp_ref;
    this->kmajor = kmajor;

    /*
    if(allocated(rayl_lower) .neqv. allocated(rayl_upper)) then
      err_message = "rayl_lower and rayl_upper must have the same allocation status"
      return
    end if
    if (allocated(rayl_lower)) then
      allocate(this%krayl(size(rayl_lower,dim=1),size(rayl_lower,dim=2),size(rayl_lower,dim=3),2))
      this%krayl(:,:,:,1) = rayl_lower
      this%krayl(:,:,:,2) = rayl_upper
    end if
    */

    // ---- post processing ----
    //  creates log reference pressure
    this->press_ref_log = this->press_ref;
    for (int i1 = 1; i1 <= this->press_ref_log.dim(1); ++i1)
        this->press_ref_log({i1}) = std::log(this->press_ref_log({i1}));

    // log scale of reference pressure
    this->press_ref_trop_log = std::log(press_ref_trop);

    // Get index of gas (if present) for determining col_gas
    create_idx_minor(
            this->gas_names, gas_minor, identifier_minor, minor_gases_lower_red, this->idx_minor_lower);
    create_idx_minor(
            this->gas_names, gas_minor, identifier_minor, minor_gases_upper_red, this->idx_minor_upper);

    // Get index of gas (if present) that has special treatment in density scaling
    create_idx_minor_scaling(
            this->gas_names, scaling_gas_lower_red, this->idx_minor_scaling_lower);
    create_idx_minor_scaling(
            this->gas_names, scaling_gas_upper_red, this->idx_minor_scaling_upper);

    // Create flavor list.
    // Reduce (remap) key_species list; checks that all key gases are present in incoming
    Array<int, 3> key_species_red;
    Array<int, 1> key_species_present_init; // CvH bool or int?

    create_key_species_reduce(
            gas_names, this->gas_names, key_species, key_species_red, key_species_present_init);

    check_key_species_present_init(gas_names, key_species_present_init);

    // create flavor list
    create_flavor(key_species_red, this->flavor);

    // create gpoint flavor list
    create_gpoint_flavor(
            key_species_red, this->get_gpoint_bands(), this->flavor, this->gpoint_flavor);

    // minimum, maximum reference temperature, pressure -- assumes low-to-high ordering
    // for T, high-to-low ordering for p
    this->temp_ref_min = this->temp_ref({1});
    this->temp_ref_max = this->temp_ref({temp_ref.dim(1)});
    this->press_ref_min = this->press_ref({press_ref.dim(1)});
    this->press_ref_max = this->press_ref({1});

    // creates press_ref_log, temp_ref_delta
    this->press_ref_log_delta =
            (std::log(this->press_ref_min) - std::log(this->press_ref_max)) / (this->press_ref.dim(1) - 1);
    this->temp_ref_delta = (this->temp_ref_max - this->temp_ref_min) / (this->temp_ref.dim(1) - 1);

    // Which species are key in one or more bands?
    // this->flavor is an index into this->gas_names
    // if (allocated(this%is_key)) deallocate(this%is_key) ! Shouldn't ever happen...
    Array<int, 1> is_key({get_ngas()}); // CvH bool, defaults to 0.?

    for (int j = 1; j <= this->flavor.dim(2); ++j)
        for (int i = 1; i <= this->flavor.dim(1); ++i)
        {
            if (this->flavor({i, j}) != 0)
                is_key({this->flavor({i, j})}) = true;
        }

    this->is_key = is_key;
}

template<typename TF>
void Gas_optics<TF>::gas_optics(
        const Array<TF,2>& play,
        const Array<TF,2>& plev,
        const Array<TF,2>& tlay,
        const Array<TF,1>& tsfc,
        const Gas_concs<TF>& gas_desc,
        std::unique_ptr<Optical_props_arry<TF>>& optical_props,
        Source_func_lw<TF>& sources,
        const Array<TF,2>& col_dry,
        const Array<TF,2>& tlev)
{
    const int ncol = play.dim(1);
    const int nlay = play.dim(2);
    const int ngpt = this->get_ngpt();
    const int nband = this->get_nband();

    Array<int,2> jtemp({play.dim(1), play.dim(2)});
    Array<int,2> jpress({play.dim(1), play.dim(2)});
    Array<int,2> tropo({play.dim(1), play.dim(2)});
    Array<TF,6> fmajor({2, 2, 2, this->get_nflav(), play.dim(1), play.dim(2)});
    Array<int,4> jeta({2, this->get_nflav(), play.dim(1), play.dim(2)});

    // Gas optics.
    compute_gas_taus(
            ncol, nlay, ngpt, nband,
            play, plev, tlay, gas_desc,
            optical_props,
            jtemp, jpress, jeta, tropo, fmajor,
            col_dry);

    // External sources.
    source(
            ncol, nlay, nband, ngpt,
            play, plev, tlay, tsfc,
            jtemp, jpress, jeta, tropo, fmajor,
            sources, tlev);
}

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

    template<typename TF> void zero_array(
            int ni, int nj, int nk, Array<TF,3>& array)
    {
        zero_array_3D(&ni, &nj, &nk, array.v().data());
    }

    template<typename TF> void zero_array(
            int ni, int nj, int nk, int nl, Array<TF,4>& array)
    {
        zero_array_4D(&ni, &nj, &nk, &nl, array.v().data());
    }

    template<typename TF>
    void interpolation(
            int ncol, int nlay,
            int ngas, int nflav, int neta, int npres, int ntemp,
            Array<int,2>& flavor,
            Array<TF,1>& press_ref_log,
            Array<TF,1>& temp_ref,
            TF press_ref_log_delta,
            TF temp_ref_min,
            TF temp_ref_delta,
            TF press_ref_trop_log,
            Array<TF,3>& vmr_ref,
            const Array<TF,2>& play,
            const Array<TF,2>& tlay,
            Array<TF,3>& col_gas,
            Array<int,2>& jtemp,
            Array<TF,6>& fmajor, Array<TF,5>& fminor,
            Array<TF,4>& col_mix,
            Array<int,2>& tropo,
            Array<int,4>& jeta,
            Array<int,2>& jpress)
    {
        interpolation(
                &ncol, &nlay,
                &ngas, &nflav, &neta, &npres, &ntemp,
                flavor.v().data(),
                press_ref_log.v().data(),
                temp_ref.v().data(),
                &press_ref_log_delta,
                &temp_ref_min,
                &temp_ref_delta,
                &press_ref_trop_log,
                vmr_ref.v().data(),
                const_cast<TF*>(play.v().data()),
                const_cast<TF*>(tlay.v().data()),
                col_gas.v().data(),
                jtemp.v().data(),
                fmajor.v().data(), fminor.v().data(),
                col_mix.v().data(),
                tropo.v().data(),
                jeta.v().data(),
                jpress.v().data());
    }

    template<typename TF>
    void compute_tau_absorption(
            int ncol, int nlay, int nband, int ngpt,
            int ngas, int nflav, int neta, int npres, int ntemp,
            int nminorlower, int nminorklower,
            int nminorupper, int nminorkupper,
            int idx_h2o,
            Array<int,2>& gpoint_flavor,
            Array<int,2>& band_lims_gpt,
            Array<TF,4>& kmajor,
            Array<TF,3>& kminor_lower,
            Array<TF,3>& kminor_upper,
            Array<int,2>& minor_limits_gpt_lower,
            Array<int,2>& minor_limits_gpt_upper,
            Array<int,1>& minor_scales_with_density_lower,
            Array<int,1>& minor_scales_with_density_upper,
            Array<int,1>& scale_by_complement_lower,
            Array<int,1>& scale_by_complement_upper,
            Array<int,1>& idx_minor_lower,
            Array<int,1>& idx_minor_upper,
            Array<int,1>& idx_minor_scaling_lower,
            Array<int,1>& idx_minor_scaling_upper,
            Array<int,1>& kminor_start_lower,
            Array<int,1>& kminor_start_upper,
            Array<int,2>& tropo,
            Array<TF,4>& col_mix, Array<TF,6>& fmajor, Array<TF,5>& fminor,
            const Array<TF,2>& play, const Array<TF,2>& tlay, Array<TF,3>& col_gas,
            Array<int,4>& jeta, Array<int,2>& jtemp, Array<int,2>& jpress,
            Array<TF,3>& tau)
    {
        compute_tau_absorption(
            &ncol, &nlay, &nband, &ngpt,
            &ngas, &nflav, &neta, &npres, &ntemp,
            &nminorlower, &nminorklower,
            &nminorupper, &nminorkupper,
            &idx_h2o,
            gpoint_flavor.v().data(),
            band_lims_gpt.v().data(),
            kmajor.v().data(),
            kminor_lower.v().data(),
            kminor_upper.v().data(),
            minor_limits_gpt_lower.v().data(),
            minor_limits_gpt_upper.v().data(),
            minor_scales_with_density_lower.v().data(),
            minor_scales_with_density_upper.v().data(),
            scale_by_complement_lower.v().data(),
            scale_by_complement_upper.v().data(),
            idx_minor_lower.v().data(),
            idx_minor_upper.v().data(),
            idx_minor_scaling_lower.v().data(),
            idx_minor_scaling_upper.v().data(),
            kminor_start_lower.v().data(),
            kminor_start_upper.v().data(),
            tropo.v().data(),
            col_mix.v().data(), fmajor.v().data(), fminor.v().data(),
            const_cast<TF*>(play.v().data()), const_cast<TF*>(tlay.v().data()), col_gas.v().data(),
            jeta.v().data(), jtemp.v().data(), jpress.v().data(),
            tau.v().data());
    }

    template<typename TF>
    void reorder123x321(
            const Array<TF,3>& data,
            Array<TF,3>& data_out)
    {
        int dim1 = data.dim(1);
        int dim2 = data.dim(2);
        int dim3 = data.dim(3);
        reorder_123x321_kernel(
                &dim1, &dim2, &dim3,
                const_cast<TF*>(data.v().data()),
                data_out.v().data());
    }
}

template<typename TF>
void Gas_optics<TF>::compute_gas_taus(
        const int ncol, const int nlay, const int ngpt, const int nband,
        const Array<TF,2>& play,
        const Array<TF,2>& plev,
        const Array<TF,2>& tlay,
        const Gas_concs<TF>& gas_desc,
        std::unique_ptr<Optical_props_arry<TF>>& optical_props,
        Array<int,2>& jtemp, Array<int,2>& jpress,
        Array<int,4>& jeta,
        Array<int,2>& tropo,
        Array<TF,6>& fmajor,
        const Array<TF,2>& col_dry)
{
    Array<TF,3> tau({ngpt, nlay, ncol});
    Array<TF,3> tau_rayleigh({ngpt, nlay, ncol});
    Array<TF,3> vmr({ncol, nlay, this->get_ngas()});
    Array<TF,3> col_gas({ncol, nlay, this->get_ngas()+1});
    col_gas.set_offsets({0, 0, -1});
    Array<TF,4> col_mix({2, this->get_nflav(), ncol, nlay});
    Array<TF,5> fminor({2, 2, this->get_nflav(), ncol, nlay});

    // CvH add all the checking...
    const int ngas = this->get_ngas();
    const int nflav = this->get_nflav();
    const int neta = this->get_neta();
    const int npres = this->get_npres();
    const int ntemp = this->get_ntemp();

    const int nminorlower = this->minor_scales_with_density_lower.dim(1);
    const int nminorklower = this->kminor_lower.dim(1);
    const int nminorupper = this->minor_scales_with_density_upper.dim(1);
    const int nminorkupper = this->kminor_upper.dim(1);

    for (int igas=1; igas<=ngas; ++igas)
    {
        Array<TF,2> vmr_2d({ncol, nlay});
        gas_desc.get_vmr(this->gas_names({igas}), vmr_2d);

        for (int icol=1; icol<=ncol; ++icol)
            for (int ilay=1; ilay<=nlay; ++ilay)
                vmr({icol, ilay, igas}) = vmr_2d({icol, ilay});
    }

    // CvH: Assume that col_dry is provided.

    for (int ilay=1; ilay<=nlay; ++ilay)
        for (int icol=1; icol<=ncol; ++icol)
            col_gas({icol, ilay, 0}) = col_dry({icol, ilay});

    for (int igas=1; igas<=ngas; ++igas)
        for (int ilay=1; ilay<=nlay; ++ilay)
            for (int icol=1; icol<=ncol; ++icol)
                col_gas({icol, ilay, igas}) = vmr({icol, ilay, igas}) * col_dry({icol, ilay});

    // Call the fortran kernels
    rrtmgp_kernels::zero_array(ngpt, nlay, ncol, tau);

    rrtmgp_kernels::interpolation(
            ncol, nlay,
            ngas, nflav, neta, npres, ntemp,
            this->flavor,
            this->press_ref_log,
            this->temp_ref,
            this->press_ref_log_delta,
            this->temp_ref_min,
            this->temp_ref_delta,
            this->press_ref_trop_log,
            this->vmr_ref,
            play,
            tlay,
            col_gas,
            jtemp,
            fmajor, fminor,
            col_mix,
            tropo,
            jeta, jpress);

    int idx_h2o = -1;
    for (int i=1; i<=this->gas_names.dim(1); ++i)
        if (gas_names({i}) == "h2o")
        {
            idx_h2o = i;
            break;
        }

    if (idx_h2o == -1)
        throw std::runtime_error("idx_h2o cannot be found");

    auto band_lims_gpoint = this->get_band_lims_gpoint();
    rrtmgp_kernels::compute_tau_absorption(
            ncol, nlay, nband, ngpt,
            ngas, nflav, neta, npres, ntemp,
            nminorlower, nminorklower,
            nminorupper, nminorkupper,
            idx_h2o,
            this->gpoint_flavor,
            band_lims_gpoint,
            this->kmajor,
            this->kminor_lower,
            this->kminor_upper,
            this->minor_limits_gpt_lower,
            this->minor_limits_gpt_upper,
            this->minor_scales_with_density_lower,
            this->minor_scales_with_density_upper,
            this->scale_by_complement_lower,
            this->scale_by_complement_upper,
            this->idx_minor_lower,
            this->idx_minor_upper,
            this->idx_minor_scaling_lower,
            this->idx_minor_scaling_upper,
            this->kminor_start_lower,
            this->kminor_start_upper,
            tropo,
            col_mix, fmajor, fminor,
            play, tlay, col_gas,
            jeta, jtemp, jpress,
            tau);

    bool has_rayleigh = false;
    combine_and_reorder(tau, tau_rayleigh, has_rayleigh, optical_props);
}

template<typename TF>
void Gas_optics<TF>::combine_and_reorder(
        const Array<TF,3>& tau,
        const Array<TF,3>& tau_rayleigh,
        const bool has_rayleigh,
        std::unique_ptr<Optical_props_arry<TF>>& optical_props)
{
    int ncol = tau.dim(3);
    int nlay = tau.dim(2);
    int ngpt = tau.dim(1);

    if (!has_rayleigh)
        rrtmgp_kernels::reorder123x321(tau, optical_props->get_tau());
    else
        throw std::runtime_error("Rayleigh scattering not implemented yet");

    // CvH for 2 stream and n-stream zero the g and ssa
}

template<typename TF>
void Gas_optics<TF>::source(
        const int ncol, const int nlay, const int nband, const int ngpt,
        const Array<TF,2>& play, const Array<TF,2>& plev,
        const Array<TF,2>& tlay, const Array<TF,1>& tsfc,
        const Array<int,2>& jtemp, const Array<int,2>& jpress,
        const Array<int,4>& jeta, const Array<int,2>& tropo,
        const Array<TF,6>& fmajor,
        Source_func_lw<TF>& sources,
        const Array<TF,2>& tlev)
{}

#endif
