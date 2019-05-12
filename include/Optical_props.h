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
            Array<int,2> band_lims_gpt_lcl(band_lims_gpt);
            this->band2gpt = band_lims_gpt_lcl;
            this->band_lims_wvn = band_lims_wvn;

            // Make a map between g-points and bands.
            this->gpt2band.set_dims({band_lims_gpt_lcl.max()});
            for (int iband=1; iband<=band_lims_gpt_lcl.dim(2); ++iband)
            {
                for (int i=band_lims_gpt_lcl({1,iband}); i<=band_lims_gpt_lcl({2,iband}); ++i)
                    this->gpt2band({i}) = iband;
            }
        }

        virtual ~Optical_props() {};

        Optical_props(const Optical_props&) = default;

        Array<int,1> get_gpoint_bands() const
        {
            Array<int,1> gpoint_bands(this->gpt2band);
            return gpoint_bands;
        }

        int get_nband() const { return this->band2gpt.dim(2); }
        int get_ngpt() const { return this->band2gpt.max(); }
        Array<int,2> get_band_lims_gpoint() const { return this->band2gpt; }
        Array<TF,2> get_band_lims_wavenumber() const { return this->band_lims_wvn; }

    private:
        Array<int,2> band2gpt;     // (begin g-point, end g-point) = band2gpt(2,band)
        Array<int,1> gpt2band;     // band = gpt2band(g-point)
        Array<TF,2> band_lims_wvn; // (upper and lower wavenumber by band) = band_lims_wvn(2,band)
};

template<typename TF>
class Optical_props_arry : public Optical_props<TF>
{
    public:
        Optical_props_arry(const Optical_props<TF>& optical_props) :
            Optical_props<TF>(optical_props)
        {}
        virtual ~Optical_props_arry() {};
        virtual Array<TF,3>& get_tau() = 0;
        virtual Array<TF,3>& get_ssa() = 0;
        virtual Array<TF,3>& get_g  () = 0;

        virtual void set_subset(
                const std::unique_ptr<Optical_props_arry<TF>>& optical_props_sub,
                const int col_s, const int col_e) = 0;

        virtual void get_subset(
                const std::unique_ptr<Optical_props_arry<TF>>& optical_props_sub,
                const int col_s, const int col_e) = 0;

        virtual int get_ncol() const = 0;
        virtual int get_nlay() const = 0;
};

template<typename TF>
class Optical_props_1scl : public Optical_props_arry<TF>
{
    public:
        // Initializer constructor.
        Optical_props_1scl(
                const int ncol,
                const int nlay,
                const Optical_props<TF>& optical_props) :
            Optical_props_arry<TF>(optical_props),
            tau({ncol, nlay, this->get_ngpt()})
        {}

        void set_subset(
                const std::unique_ptr<Optical_props_arry<TF>>& optical_props_sub,
                const int col_s, const int col_e)
        {
            for (int igpt=1; igpt<=tau.dim(3); ++igpt)
                for (int ilay=1; ilay<=tau.dim(2); ++ilay)
                    for (int icol=col_s; icol<=col_e; ++icol)
                        tau({icol, ilay, igpt}) = optical_props_sub->get_tau()({icol-col_s+1, ilay, igpt});
        }

        void get_subset(
                const std::unique_ptr<Optical_props_arry<TF>>& optical_props_sub,
                const int col_s, const int col_e)
        {
            for (int igpt=1; igpt<=tau.dim(3); ++igpt)
                for (int ilay=1; ilay<=tau.dim(2); ++ilay)
                    for (int icol=col_s; icol<=col_e; ++icol)
                        tau({icol-col_s+1, ilay, igpt}) = optical_props_sub->get_tau()({icol, ilay, igpt});
        }

        int get_ncol() const { return tau.dim(1); }
        int get_nlay() const { return tau.dim(2); }

        Array<TF,3>& get_tau() { return tau; }
        Array<TF,3>& get_ssa() { throw std::runtime_error("ssa is not available in this class"); }
        Array<TF,3>& get_g  () { throw std::runtime_error("g is available in this class"); }

    private:
        Array<TF,3> tau;
};

template<typename TF>
class Optical_props_2str : public Optical_props_arry<TF>
{
    public:
        // Initializer constructor.
        Optical_props_2str(
                const int ncol,
                const int nlay,
                const Optical_props<TF>& optical_props) :
            Optical_props_arry<TF>(optical_props),
            tau({ncol, nlay, this->get_ngpt()}),
            ssa({ncol, nlay, this->get_ngpt()}),
            g  ({ncol, nlay, this->get_ngpt()})
        {}

        void set_subset(
                const std::unique_ptr<Optical_props_arry<TF>>& optical_props_sub,
                const int col_s, const int col_e)
        {
            for (int igpt=1; igpt<=tau.dim(3); ++igpt)
                for (int ilay=1; ilay<=tau.dim(2); ++ilay)
                    for (int icol=col_s; icol<=col_e; ++icol)
                    {
                        tau({icol, ilay, igpt}) = optical_props_sub->get_tau()({icol-col_s+1, ilay, igpt});
                        ssa({icol, ilay, igpt}) = optical_props_sub->get_ssa()({icol-col_s+1, ilay, igpt});
                        g  ({icol, ilay, igpt}) = optical_props_sub->get_g  ()({icol-col_s+1, ilay, igpt});
                    }
        }

        void get_subset(
                const std::unique_ptr<Optical_props_arry<TF>>& optical_props_sub,
                const int col_s, const int col_e)
        {
            for (int igpt=1; igpt<=tau.dim(3); ++igpt)
                for (int ilay=1; ilay<=tau.dim(2); ++ilay)
                    for (int icol=col_s; icol<=col_e; ++icol)
                    {
                        tau({icol-col_s+1, ilay, igpt}) = optical_props_sub->get_tau()({icol, ilay, igpt});
                        ssa({icol-col_s+1, ilay, igpt}) = optical_props_sub->get_ssa()({icol, ilay, igpt});
                        g  ({icol-col_s+1, ilay, igpt}) = optical_props_sub->get_g  ()({icol, ilay, igpt});
                    }
        }

        int get_ncol() const { return tau.dim(1); }
        int get_nlay() const { return tau.dim(2); }

        Array<TF,3>& get_tau() { return tau; }
        Array<TF,3>& get_ssa() { return ssa; }
        Array<TF,3>& get_g  () { return g; }

    private:
        Array<TF,3> tau;
        Array<TF,3> ssa;
        Array<TF,3> g;
};
#endif
