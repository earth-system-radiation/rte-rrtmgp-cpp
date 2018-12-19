#ifndef OPTICAL_PROPS_H
#define OPTICAL_PROPS_H

#include "Array.h"

template<typename TF>
class Optical_props
{
    public:
        Optical_props(
                Array<TF,2>& band_lims_wvn,
                Array<int,2>& band_lims_gpt)
        {
            // -------------------------
            // 
            // Error checking -- are the arrays the size we expect, contain positive values?
            // 
            /*
            err_message = ""
            if(size(band_lims_wvn,1) /= 2) &
              err_message = "optical_props%init(): band_lims_wvn 1st dim should be 2"
            if(any(band_lims_wvn < 0._wp) ) &
              err_message = "optical_props%init(): band_lims_wvn has values <  0., respectively"
            if(len_trim(err_message) > 0) return
            if(present(band_lims_gpt)) then
              if(size(band_lims_gpt, 1) /= 2)&
                err_message = "optical_props%init(): band_lims_gpt 1st dim should be 2"
              if(size(band_lims_gpt,2) /= size(band_lims_wvn,2)) &
                err_message = "optical_props%init(): band_lims_gpt and band_lims_wvn sized inconsistently"
              if(any(band_lims_gpt < 1) ) &
                err_message = "optical_props%init(): band_lims_gpt has values < 1"
              if(len_trim(err_message) > 0) return

              band_lims_gpt_lcl(:,:) = band_lims_gpt(:,:)
            else
              !
              ! Assume that values are defined by band, one g-point per band
              !
              do iband = 1, size(band_lims_wvn, 2)
                band_lims_gpt_lcl(1:2,iband) = iband
              end do
            end if
            !
            ! Assignment
            !
            if(allocated(this%band2gpt     )) deallocate(this%band2gpt)
            if(allocated(this%band_lims_wvn)) deallocate(this%band_lims_wvn)
            allocate(this%band2gpt     (2,size(band_lims_wvn,2)), &
                     this%band_lims_wvn(2,size(band_lims_wvn,2)))
            this%band2gpt      = band_lims_gpt_lcl
            this%band_lims_wvn = band_lims_wvn
            if(present(name)) this%name = trim(name)

            !
            ! Make a map between g-points and bands
            !   Efficient only when g-point indexes start at 1 and are contiguous.
            !
            if(allocated(this%gpt2band)) deallocate(this%gpt2band)
            allocate(this%gpt2band(maxval(band_lims_gpt_lcl)))
            do iband=1,size(band_lims_gpt_lcl,dim=2)
              this%gpt2band(band_lims_gpt_lcl(1,iband):band_lims_gpt_lcl(2,iband)) = iband
            end do
            */
        }

        virtual ~Optical_props() {};

    private:
        // Array<int,2> band2gpt;     // (begin g-point, end g-point) = band2gpt(2,band)
        // Array<int,1> gpt2band;     // band = gpt2band(g-point)
        // Array<TF,2> band_lims_wvn; // (upper and lower wavenumber by band) = band_lims_wvn(2,band)
};

/*
template<typename TF>
class Optical_props_arry : public Optical_props
{
    public:
        Optical_props_arry() {};
        virtual ~Optical_props_arry() {};
        virtual Array<TF,3>& get_tau() = 0;
        virtual Array<TF,3>& get_ssa() = 0;
        virtual Array<TF,3>& get_g() = 0;

        // virtual void get_subset(const int, const int, std::unique_ptr<Optical_props_arry>&) = 0;
};

template<typename TF>
class Optical_props_2str : public Optical_props
{
    public:
        Optical_props_2str(
                const int ncol, const int nlay, const int ngpt,
                Array_3d<TF>&& tau, Array_3d<TF>&& ssa, Array_3d<TF>&& g,
                const std::string name="") :
            tau_(std::move(tau)), ssa_(std::move(ssa)), g_(std::move(g)), name_(name)
        {};
        ~Optical_props_2str() {};

        void get_subset(const int col_start, const int block_size, std::unique_ptr<Optical_props>& optical_props)
        {
        }

    private:
        Array_3d<TF> tau_;
        Array_3d<TF> ssa_;
        Array_3d<TF> g_;
        const std::string name_;
};

template<typename TF>
class Optical_props_1scl : public Optical_props_arry<TF>
{
    public:
        // Initializer constructor.
        Optical_props_1scl(
                const int ncol, const int nlay, const int ngpt,
                Array<TF,3>&& tau,
                const std::string name="") :
            tau_(std::move(tau)),
            name_(name)
        {};

        // Subset constructor.
        Optical_props_1scl(
                std::unique_ptr<Optical_props_arry<TF>>& full,
                const int icol_start,
                const int ncol) :
            name_(full->get_name())
        {
            const int ncol_full = full->get_ncol();
            const int nlay = full->get_nlay();
            const int ngpt = full->get_ngpt();

            if ( (icol_start < 0) ||
                 ( (icol_start + ncol) > ncol_full) )
            {
                throw std::runtime_error("Optical_props::get_subset: out of range");
            }

            // Implement the class switching. Design is a little problematic at the moment.
            tau_.resize(ncol, nlay, ngpt);

            // Assign the sub-column.
            for (int igpt=0; igpt<ngpt; ++igpt)
                for (int ilay=0; ilay<nlay; ++ilay)
                    for (int icol=0; icol<ncol; ++icol)
                        tau_(icol, ilay, igpt) = full->get_tau()(icol+icol_start, ilay, igpt);
        }

        ~Optical_props_1scl() {};

        int get_ncol() const { return tau_.dim1(); }
        int get_nlay() const { return tau_.dim2(); }
        int get_ngpt() const { return tau_.dim3(); }
        std::string get_name() const { return name_; }

        Array<TF,3>& get_tau() { return tau_; }
        Array<TF,3>& get_ssa() { throw std::runtime_error("Not available in this class"); }
        Array<TF,3>& get_g  () { throw std::runtime_error("Not available in this class"); }

    private:
        Array<TF,3> tau_;
        const std::string name_;
};

*/
#endif
