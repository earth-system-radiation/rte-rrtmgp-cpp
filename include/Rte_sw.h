#ifndef RTE_SW_H
#define RTE_SW_H

namespace rrtmgp_kernels
{
    extern "C" void apply_BC_0(
            int* ncol, int* nlay, int* ngpt,
            int* top_at_1, double* gpt_flux_dn);

    extern "C" void apply_BC_factor(
            int* ncol, int* nlay, int* ngpt,
            int* top_at_1, double* inc_flux,
            double* factor, double* flux_dn);

    extern "C" void sw_solver_2stream(
            int* ncol, int* nlay, int* ngpt, int* top_at_1,
            double* tau,
            double* ssa,
            double* g,
            double* mu0,
            double* sfc_alb_dir_gpt, double* sfc_alb_dif_gpt,
            double* gpt_flux_up, double* gpt_flux_dn, double* gpt_flux_dir);

    // CvH: This one is already defined in the LW solver.
    // template<typename TF>
    // void apply_BC(
    //         int ncol, int nlay, int ngpt,
    //         int top_at_1, Array<TF,3>& gpt_flux_dn)
    // {
    //     apply_BC_0(
    //             &ncol, &nlay, &ngpt,
    //             &top_at_1, gpt_flux_dn.ptr());
    // }

    template<typename TF>
    void apply_BC(
            int ncol, int nlay, int ngpt, int top_at_1,
            const Array<TF,2>& inc_flux,
            const Array<TF,1>& factor,
            Array<TF,3>& gpt_flux)
    {
        apply_BC_factor(
                &ncol, &nlay, &ngpt,
                &top_at_1,
                const_cast<TF*>(inc_flux.ptr()),
                const_cast<TF*>(factor.ptr()),
                gpt_flux.ptr());
    }

    template<typename TF>
    void sw_solver_2stream(
            int ncol, int nlay, int ngpt, int top_at_1,
            const Array<TF,3>& tau,
            const Array<TF,3>& ssa,
            const Array<TF,3>& g,
            const Array<TF,1>& mu0,
            const Array<TF,2>& sfc_alb_dir_gpt, const Array<TF,2>& sfc_alb_dif_gpt,
            Array<TF,3>& gpt_flux_up, Array<TF,3>& gpt_flux_dn, Array<TF,3>& gpt_flux_dir)
    {
        sw_solver_2stream(
                &ncol, &nlay, &ngpt, &top_at_1,
                const_cast<TF*>(tau.ptr()),
                const_cast<TF*>(ssa.ptr()),
                const_cast<TF*>(g  .ptr()),
                const_cast<TF*>(mu0.ptr()),
                const_cast<TF*>(sfc_alb_dir_gpt.ptr()),
                const_cast<TF*>(sfc_alb_dif_gpt.ptr()),
                gpt_flux_up.ptr(), gpt_flux_dn.ptr(), gpt_flux_dir.ptr());
    }
}

template<typename TF>
class Rte_sw
{
    public:
        static void rte_sw(
                const std::unique_ptr<Optical_props_arry<TF>>& optical_props,
                const int top_at_1,
                const Array<TF,1>& mu0,
                const Array<TF,2>& inc_flux,
                const Array<TF,2>& sfc_alb_dir,
                const Array<TF,2>& sfc_alb_dif,
                std::unique_ptr<Fluxes_broadband<TF>>& fluxes)
        {
            const int ncol = optical_props->get_ncol();
            const int nlay = optical_props->get_nlay();
            const int ngpt = optical_props->get_ngpt();
            // const int nband = optical_props->get_nband();

            Array<TF,3> gpt_flux_up ({ncol, nlay+1, ngpt});
            Array<TF,3> gpt_flux_dn ({ncol, nlay+1, ngpt});
            Array<TF,3> gpt_flux_dir({ncol, nlay+1, ngpt});

            Array<TF,2> sfc_alb_dir_gpt({ncol, ngpt});
            Array<TF,2> sfc_alb_dif_gpt({ncol, ngpt});

            expand_and_transpose(optical_props, sfc_alb_dir, sfc_alb_dir_gpt);
            expand_and_transpose(optical_props, sfc_alb_dif, sfc_alb_dif_gpt);

            // Upper boundary condition.
            rrtmgp_kernels::apply_BC(ncol, nlay, ngpt, top_at_1, inc_flux, mu0, gpt_flux_dir);
            rrtmgp_kernels::apply_BC(ncol, nlay, ngpt, top_at_1, gpt_flux_dn);

            // Run the radiative transfer solver
            // CvH: only two-stream solutions, I skipped the sw_solver_noscat
            rrtmgp_kernels::sw_solver_2stream(
                    ncol, nlay, ngpt, top_at_1,
                    optical_props->get_tau(),
                    optical_props->get_ssa(),
                    optical_props->get_g  (),
                    mu0,
                    sfc_alb_dir_gpt, sfc_alb_dif_gpt,
                    gpt_flux_up, gpt_flux_dn, gpt_flux_dir);

            fluxes->reduce(gpt_flux_up, gpt_flux_dn, gpt_flux_dir, optical_props, top_at_1);
        }

        static void expand_and_transpose(
                const std::unique_ptr<Optical_props_arry<TF>>& ops,
                const Array<TF,2> arr_in,
                Array<TF,2>& arr_out)
        {
            const int ncol = arr_in.dim(2);
            const int nband = ops->get_nband();
            Array<int,2> limits = ops->get_band_lims_gpoint();

            for (int iband=1; iband<=nband; ++iband)
                for (int icol=1; icol<=ncol; ++icol)
                    for (int igpt=limits({1, iband}); igpt<=limits({2, iband}); ++igpt)
                        arr_out({icol, igpt}) = arr_in({iband, icol});
        }
};
#endif
