#ifndef RTE_LW_H
#define RTE_LW_H

template<typename TF>
class Rte_lw
{
    public:
        static void rte_lw(
                const std::unique_ptr<Optical_props_arry<TF>>& optical_props,
                const int top_at_1,
                const Source_func_lw<TF>& sources,
                const Array<TF,2>& sfc_emis,
                std::unique_ptr<Fluxes<TF>>& fluxes,
                const int n_gauss_angles)
        {}
};
#endif
