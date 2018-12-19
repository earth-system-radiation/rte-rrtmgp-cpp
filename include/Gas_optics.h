#ifndef GAS_OPTICS_H
#define GAS_OPTICS_H

#include "Optical_props.h"
#include "Array.h"

template<typename TF> class Gas_concs;

template<typename TF>
class Gas_optics : public Optical_props<TF>
{
    public:
        Gas_optics(
                std::vector<Gas_concs<TF>>& available_gases,
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

    private:
        Array<TF,2> totplnk;
        Array<TF,4> planck_frac;
        TF totplnk_delta;
        TF temp_ref_min, temp_ref_max;
        TF press_ref_min, press_ref_max;
        TF press_ref_trop, press_ref_trop_log;

        TF press_ref_log_delta;
        TF temp_ref_delta;

        Array<TF,1> press_ref, press_ref_log, temp_ref;

        Array<std::string,1> gas_names;

        Array<TF,3> vmr_ref;
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
        Array<int,2> flavor;

        int get_ngas() const { return this->gas_names.dim(0); }

        void init_abs_coeffs(
                std::vector<Gas_concs<TF>>& available_gases,
                Array<std::string,1>& gas_names,
                Array<int,3>& key_species,
                Array<int,2>& band2gpt,
                Array<TF,2>& band_lims_wavenum,
                Array<TF,1>& press_ref,
                Array<TF,1>& temp_ref,
                TF press_ref_trop,
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
                Array<TF,3>& rayl_lower,
                Array<TF,3>& rayl_upper);
};

namespace
{
    template<typename TF>
    void reduce_minor_arrays(
                std::vector<Gas_concs<TF>>& available_gases,
                Array<std::string,1>& gas_names,
                Array<std::string,1>& gas_minor,
                Array<std::string,1>& identifier_minor,
                Array<TF,3>& kminor_atm,
                Array<std::string,1>& minor_gases_atm,
                Array<int,2>& minor_limits_gpt_atm,
                Array<int,1>& minor_scales_with_density_atm, // CvH: logical bool or int?
                Array<std::string,1>& scaling_gas_atm,
                Array<int,1>& scale_by_complement_atm, // CvH: bool or int
                Array<int,1>& kminor_start_atm,
                Array<TF,3>& kminor_atm_red,
                Array<std::string,1>& minor_gases_atm_red,
                Array<int,2>& minor_limits_gpt_atm_red,
                Array<int,1>& minor_scales_with_density_atm_red, // CvH bool or int
                Array<std::string,1>& scaling_gas_atm_red,
                Array<int,1>& scale_by_complement_atm_red,
                Array<int,1>& kminor_start_atm_red)
    {}
            /*
    ! Local variables
    integer :: i, j
    integer :: idx_mnr, nm, tot_g, red_nm
    integer :: icnt, n_elim, ng
    logical, dimension(:), allocatable :: gas_is_present

    nm = size(minor_gases_atm)
    tot_g=0
    allocate(gas_is_present(nm))
    do i = 1, size(minor_gases_atm)
      idx_mnr = string_loc_in_array(minor_gases_atm(i), identifier_minor)
      gas_is_present(i) = string_in_array(gas_minor(idx_mnr),available_gases%gas_name)
      if(gas_is_present(i)) then
        tot_g = tot_g + (minor_limits_gpt_atm(2,i)-minor_limits_gpt_atm(1,i)+1)
      endif
    enddo
    red_nm = count(gas_is_present)

    if ((red_nm .eq. nm)) then
      kminor_atm_red = kminor_atm
      minor_gases_atm_red = minor_gases_atm
      minor_limits_gpt_atm_red = minor_limits_gpt_atm
      minor_scales_with_density_atm_red = minor_scales_with_density_atm
      scaling_gas_atm_red = scaling_gas_atm
      scale_by_complement_atm_red = scale_by_complement_atm
      kminor_start_atm_red = kminor_start_atm
    else
      minor_gases_atm_red= pack(minor_gases_atm, mask=gas_is_present)
      minor_scales_with_density_atm_red = pack(minor_scales_with_density_atm, &
        mask=gas_is_present)
      scaling_gas_atm_red = pack(scaling_gas_atm, &
        mask=gas_is_present)
      scale_by_complement_atm_red = pack(scale_by_complement_atm, &
        mask=gas_is_present)
      kminor_start_atm_red = pack(kminor_start_atm, &
        mask=gas_is_present)

      allocate(minor_limits_gpt_atm_red(2, red_nm))
      allocate(kminor_atm_red(tot_g, size(kminor_atm,2), size(kminor_atm,3)))

      icnt = 0
      n_elim = 0
      do i = 1, nm
        ng = minor_limits_gpt_atm(2,i)-minor_limits_gpt_atm(1,i)+1
        if(gas_is_present(i)) then
          icnt = icnt + 1
          minor_limits_gpt_atm_red(1:2,icnt) = minor_limits_gpt_atm(1:2,i)
          kminor_start_atm_red(icnt) = kminor_start_atm(i)-n_elim
          do j = 1, ng
            kminor_atm_red(kminor_start_atm_red(icnt)+j-1,:,:) = &
              kminor_atm(kminor_start_atm(i)+j-1,:,:)
          enddo
        else
          n_elim = n_elim + ng
        endif
      enddo
    endif*/

    void create_idx_minor(
            const Array<std::string,1>& gas_names,
            const Array<std::string,1>& gas_minor,
            const Array<std::string,1>& identifier_minor,
            const Array<std::string,1>& minor_gases_atm,
            Array<int,1> idx_minor_atm)
    {
        Array<int,1> idx_minor_atm_out({minor_gases_atm.dim(1)});

        for (int imnr=1; imnr<=minor_gases_atm.dim(1); ++imnr)
        {
            // Find identifying string for minor species in list of possible identifiers (e.g. h2o_slf)
            const int idx_mnr = identifier_minor.find_indices(minor_gases_atm({imnr}))[0];

            // Find name of gas associated with minor species identifier (e.g. h2o)
            idx_minor_atm_out({imnr}) = gas_names.find_indices(gas_minor({idx_mnr}))[0];
        }

        idx_minor_atm = idx_minor_atm_out;
    }

    void create_idx_minor_scaling(
            const Array<std::string,1>& gas_names,
            const Array<std::string,1>& scaling_gas_atm,
            Array<int,1> idx_minor_scaling_atm)
    {
        Array<int,1> idx_minor_scaling_atm_out({scaling_gas_atm.dim(1)});

        for (int imnr=1; imnr<=scaling_gas_atm.dim(1); ++imnr)
            idx_minor_scaling_atm_out({imnr}) = gas_names.find_indices(scaling_gas_atm({imnr}))[0];

        idx_minor_scaling_atm = idx_minor_scaling_atm_out;
    }
}

template<typename TF>
Gas_optics<TF>::Gas_optics(
        std::vector<Gas_concs<TF>>& available_gases,
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
        std::vector<Gas_concs<TF>>& available_gases,
        Array<std::string,1>& gas_names,
        Array<int,3>& key_species,
        Array<int,2>& band2gpt,
        Array<TF,2>& band_lims_wavenum,
        Array<TF,1>& press_ref,
        Array<TF,1>& temp_ref,
        TF press_ref_trop,
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
        Array<TF,3>& rayl_lower,
        Array<TF,3>& rayl_upper)
{
    // Which gases known to the gas optics are present in the host model (available_gases)?
    std::vector<std::string> gas_names_to_use;

    for (const std::string& s : gas_names.v())
    {
        auto it = std::find_if(
                available_gases.begin(), available_gases.end(),
                [&s](const auto& a){ return a.get_name() == s; } );
        if (it != available_gases.end())
            gas_names_to_use.push_back(s);
    }

    // Now the number of gases is the union of those known to the k-distribution and provided
    // by the host model.
    const int n_gas = gas_names_to_use.size();
    Array<std::string,1> gas_names_this(std::move(gas_names_to_use), {n_gas});
    this->gas_names = gas_names_this;

    // Initialize the gas optics object, keeping only those gases known to the
    // gas optics and also present in the host model.
    // Add an offset to the indexing to interface the negative ranging of fortran.
    Array<TF,3> vmr_ref_red({vmr_ref.dim(1), n_gas+1, vmr_ref.dim(3)});
    vmr_ref_red.set_offsets({0, -1, 0});

    // Gas 0 is used in single-key species method, set to 1.0 (col_dry)
    for (int i1=1; i1<=vmr_ref_red.dim(1); ++i1)
        for (int i3=1; i3<=vmr_ref_red.dim(3); ++i3)
            vmr_ref_red({i1,0,i3}) = vmr_ref({i1,1,i3});

    for (int i=1; i<=n_gas; ++i)
    {
        int idx = gas_names.find_indices(this->gas_names({i}))[0];
        for (int i1=1; i1<=vmr_ref_red.dim(1); ++i1)
            for (int i3=1; i3<=vmr_ref_red.dim(3); ++i3)
                vmr_ref_red({i1,i,i3}) = vmr_ref({i1,idx+1,i3}); // CvH: why +1?
    }

    this->vmr_ref = std::move(vmr_ref_red);

    // Reduce minor arrays so variables only contain minor gases that are available.
    // Reduce size of minor Arrays.
    Array<std::string,1> minor_gases_lower_red;
    Array<std::string,1> scaling_gas_lower_red;
    Array<std::string,1> minor_gases_upper_red;
    Array<std::string,1> scaling_gas_upper_red;

    reduce_minor_arrays(
            available_gases,
            gas_names,
            gas_minor,identifier_minor,
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
    for (int i1=1; i1<=this->press_ref_log.dim(1); ++i1)
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

    /*
    ! create flavor list
    ! Reduce (remap) key_species list; checks that all key gases are present in incoming
    call create_key_species_reduce(gas_names,this%gas_names, &
      key_species,key_species_red,key_species_present_init)
    err_message = check_key_species_present_init(gas_names,key_species_present_init)
    if(len_trim(err_message) /= 0) return
    ! create flavor list
    call create_flavor(key_species_red, this%flavor)
    ! create gpoint_flavor list
    call create_gpoint_flavor(key_species_red, this%get_gpoint_bands(), this%flavor, this%gpoint_flavor)
    */

    // minimum, maximum reference temperature, pressure -- assumes low-to-high ordering
    // for T, high-to-low ordering for p
    this->temp_ref_min = this->temp_ref({1});
    this->temp_ref_max = this->temp_ref({temp_ref.dim(1)});
    this->press_ref_min = this->press_ref({press_ref.dim(1)});
    this->press_ref_max = this->press_ref({1});

    // creates press_ref_log, temp_ref_delta
    this->press_ref_log_delta = (std::log(this->press_ref_min) - std::log(this->press_ref_max)) / (this->press_ref.dim(1)-1);
    this->temp_ref_delta = (this->temp_ref_max - this->temp_ref_min) / (this->temp_ref.dim(1)-1);

    // Which species are key in one or more bands?
    // this->flavor is an index into this->gas_names
    // if (allocated(this%is_key)) deallocate(this%is_key) ! Shouldn't ever happen...
    Array<int,1> is_key({get_ngas()}); // CvH bool, defaults to 0.?

    for (int j=1; j<=this->flavor.dim(2); ++j)
        for (int i=1; i<=this->flavor.dim(1); ++i)
        {
            if (this->flavor({i,j}) != 0)
                is_key({this->flavor({i,j})}) = true;
        }

    this->is_key = is_key;
}
#endif
