#ifndef FLUXES_H
#define FLUXES_H

template<typename TF>
class Fluxes
{
    public:
        virtual void reduce(
                const Array<TF,3>& gpt_flux_up, const Array<TF,3>& gpt_flux_dn,
                const std::unique_ptr<Optical_props_arry<TF>>& optical_props,
                const int top_at_1) = 0;
};

template<typename TF>
class Fluxes_broadband : public Fluxes<TF>
{
    public:
        Fluxes_broadband();
        virtual void reduce(
                const Array<TF,3>& gpt_flux_up, const Array<TF,3>& gpt_flux_dn,
                const std::unique_ptr<Optical_props_arry<TF>>& optical_props,
                const int top_at_1);

        void set_flux_up(Array<TF,2>& flux_up_in, const int col_s)
        { flux_up = &flux_up_in.v()[col_s-1]; }

        void set_flux_dn(Array<TF,2>& flux_dn_in, const int col_s)
        { flux_dn = &flux_dn_in.v()[col_s-1]; }

        void set_flux_net(Array<TF,2>& flux_net_in, const int col_s)
        { flux_net = &flux_net_in.v()[col_s-1]; }

    private:
        TF* flux_up;
        TF* flux_dn;
        TF* flux_net;
};

template<typename TF>
class Fluxes_byband : public Fluxes_broadband<TF>
{
    public:
        Fluxes_byband();
        void reduce(
                const Array<TF,3>& gpt_flux_up, const Array<TF,3>& gpt_flux_dn,
                const std::unique_ptr<Optical_props_arry<TF>>& optical_props,
                const int top_at_1);

    private:
        TF* bnd_flux_up;
        TF* bnd_flux_dn;
        TF* bnd_flux_net;
};

namespace rrtmgp_kernels
{
    extern "C" void sum_broadband(
            int* ncol, int* nlev, int* ngpt,
            double* spectral_flux, double* broadband_flux);

    extern "C" void net_broadband_precalc(
            int* ncol, int* nlev,
            double* spectral_flux_dn, double* spectral_flux_up,
            double* broadband_flux_net);

    template<typename TF>
    void sum_broadband(
            int ncol, int nlev, int ngpt,
            const Array<TF,3>& spectral_flux, TF* broadband_flux)
    {
        sum_broadband(
                &ncol, &nlev, &ngpt,
                const_cast<TF*>(spectral_flux.v().data()),
                broadband_flux);
    }

    template<typename TF>
    void net_broadband(
            int ncol, int nlev,
            TF* spectral_flux_dn, TF* spectral_flux_up,
            TF* broadband_flux_net)
    {
        net_broadband_precalc(
                &ncol, &nlev,
                spectral_flux_dn, spectral_flux_up,
                broadband_flux_net);
    }
}

template<typename TF>
Fluxes_broadband<TF>::Fluxes_broadband() :
    flux_up(nullptr), flux_dn(nullptr), flux_net(nullptr)
{}

template<typename TF>
void Fluxes_broadband<TF>::reduce(
    const Array<TF,3>& gpt_flux_up, const Array<TF,3>& gpt_flux_dn,
    const std::unique_ptr<Optical_props_arry<TF>>& spectral_disc,
    const int top_at_1)
{
    const int ncol = gpt_flux_up.dim(1);
    const int nlev = gpt_flux_up.dim(2);
    const int ngpt = spectral_disc->get_ngpt();

    auto band_lims = spectral_disc->get_band_lims_gpoint();

    if (this->flux_up != nullptr)
        rrtmgp_kernels::sum_broadband(
                ncol, nlev, ngpt, gpt_flux_up, this->flux_up);

    if (this->flux_dn != nullptr)
        rrtmgp_kernels::sum_broadband(
                ncol, nlev, ngpt, gpt_flux_dn, this->flux_dn);

    if (this->flux_up != nullptr && this->flux_dn != nullptr)
        rrtmgp_kernels::net_broadband(
                ncol, nlev, this->flux_dn, this->flux_up, this->flux_net);
}

template<typename TF>
Fluxes_byband<TF>::Fluxes_byband() :
        bnd_flux_up(nullptr), bnd_flux_dn(nullptr), bnd_flux_net(nullptr)
{}

template<typename TF>
void Fluxes_byband<TF>::reduce(
    const Array<TF,3>& gpt_flux_up, const Array<TF,3>& gpt_flux_dn,
    const std::unique_ptr<Optical_props_arry<TF>>& spectral_disc,
    const int top_at_1)
{
    const int ncol = gpt_flux_up.dim(1);
    const int nlev = gpt_flux_up.dim(2);
    const int ngpt = spectral_disc->get_ngpt();
    const int nbnd = spectral_disc->get_nband();

    auto band_lims = spectral_disc->get_band_lims_gpoint();

    Fluxes_broadband<TF>::reduce(
            gpt_flux_up, gpt_flux_dn,
            spectral_disc, top_at_1);
}
#endif
