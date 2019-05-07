#ifndef FLUXES_H
#define FLUXES_H

template<typename TF>
class Fluxes
{
    public:
        virtual void reduce() = 0;
};

template<typename TF>
class Fluxes_broadband : public Fluxes<TF>
{
    public:
        Fluxes_broadband();
        void reduce() {}

    private:
        TF* flux_up;
        TF* flux_dn;
        TF* flux_net;
};

template<typename TF>
class Fluxes_byband : public Fluxes_broadband<TF>
{
    public:
        Fluxes_broadband();
        void reduce() {}

    private:
        TF* bnd_flux_up;
        TF* bnd_flux_dn;
        TF* bnd_flux_net;
};

template<typename TF>
Fluxes_broadband<TF>::Fluxes_broadband() :
    flux_up(nullptr), flux_dn(nullptr), flux_net(nullptr)
{}
#endif
