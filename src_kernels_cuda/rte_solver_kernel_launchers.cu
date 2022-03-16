#include <chrono>

#include "rte_kernel_launcher_cuda.h"
#include "tools_gpu.h"
#include "Array.h"
#include "tuner.h"

#include <iomanip>


namespace
{
    #include "rte_solver_kernels.cu"
}


namespace rte_kernel_launcher_cuda
{
    void apply_BC(const int ncol, const int nlay, const int ngpt, const Bool top_at_1,
                  const Array_gpu<Float,2>& inc_flux_dir, const Array_gpu<Float,1>& mu0, Array_gpu<Float,3>& gpt_flux_dir)
    {
        const int block_col = 32;
        const int block_gpt = 32;

        const int grid_col = ncol/block_col + (ncol%block_col > 0);
        const int grid_gpt = ngpt/block_gpt + (ngpt%block_gpt > 0);

        dim3 grid_gpu(grid_col, grid_gpt);
        dim3 block_gpu(block_col, block_gpt);

        apply_BC_kernel<<<grid_gpu, block_gpu>>>(ncol, nlay, ngpt, top_at_1, inc_flux_dir.ptr(), mu0.ptr(), gpt_flux_dir.ptr());
    }


    void apply_BC(const int ncol, const int nlay, const int ngpt, const Bool top_at_1, Array_gpu<Float,3>& gpt_flux_dn)
    {
        const int block_col = 32;
        const int block_gpt = 32;

        const int grid_col = ncol/block_col + (ncol%block_col > 0);
        const int grid_gpt = ngpt/block_gpt + (ngpt%block_gpt > 0);

        dim3 grid_gpu(grid_col, grid_gpt);
        dim3 block_gpu(block_col, block_gpt);
        apply_BC_kernel<<<grid_gpu, block_gpu>>>(ncol, nlay, ngpt, top_at_1, gpt_flux_dn.ptr());
    }


    void apply_BC(const int ncol, const int nlay, const int ngpt, const Bool top_at_1, const Array_gpu<Float,2>& inc_flux_dif, Array_gpu<Float,3>& gpt_flux_dn)
    {
        const int block_col = 32;
        const int block_gpt = 32;

        const int grid_col = ncol/block_col + (ncol%block_col > 0);
        const int grid_gpt = ngpt/block_gpt + (ngpt%block_gpt > 0);

        dim3 grid_gpu(grid_col, grid_gpt);
        dim3 block_gpu(block_col, block_gpt);

        apply_BC_kernel<<<grid_gpu, block_gpu>>>(ncol, nlay, ngpt, top_at_1, inc_flux_dif.ptr(), gpt_flux_dn.ptr());
    }


    void lw_secants_array(
            const int ncol, const int ngpt, const int n_gauss_quad, const int max_gauss_pts,
            const Array_gpu<Float,2>& gauss_Ds, Array_gpu<Float,3>& secants)
    {
        const int block_col = 32;
        const int block_gpt = 32;

        const int grid_col = ncol/block_col + (ncol%block_col > 0);
        const int grid_gpt = ngpt/block_gpt + (ngpt%block_gpt > 0);

        dim3 grid_gpu(grid_col, grid_gpt, n_gauss_quad);
        dim3 block_gpu(block_col, block_gpt, 1);

        lw_secants_array_kernel<<<grid_gpu, block_gpu>>>(
                ncol, ngpt, n_gauss_quad, max_gauss_pts,
                gauss_Ds.ptr(), secants.ptr());
    }


    void lw_solver_noscat_gaussquad(
            const int ncol, const int nlay, const int ngpt, const Bool top_at_1, const int nmus,
            const Array_gpu<Float,3>& secants, const Array_gpu<Float,2>& weights,
            const Array_gpu<Float,3>& tau, const Array_gpu<Float,3> lay_source,
            const Array_gpu<Float,3>& lev_source_inc, const Array_gpu<Float,3>& lev_source_dec,
            const Array_gpu<Float,2>& sfc_emis, const Array_gpu<Float,2>& sfc_src,
            const Array_gpu<Float,2>& inc_flux,
            Array_gpu<Float,3>& flux_up, Array_gpu<Float,3>& flux_dn,
            const Bool do_broadband, Array_gpu<Float,3>& flux_up_loc, Array_gpu<Float,3>& flux_dn_loc,
            const Bool do_jacobians, const Array_gpu<Float,2>& sfc_src_jac, Array_gpu<Float,3>& flux_up_jac,
            void* calling_class_ptr)
    {
        Float eps = std::numeric_limits<Float>::epsilon();

        const int flx_size = flux_dn.size();
        const int opt_size = tau.size();
        const int sfc_size = sfc_src.size();

        Array_gpu<Float,2> source_sfc(sfc_src.get_dims());
        Array_gpu<Float,2> source_sfc_jac(sfc_src.get_dims());
        Array_gpu<Float,2> sfc_albedo(sfc_src.get_dims());
        Array_gpu<Float,3> tau_loc(tau.get_dims());
        Array_gpu<Float,3> trans(tau.get_dims());
        Array_gpu<Float,3> source_dn(tau.get_dims());
        Array_gpu<Float,3> source_up(tau.get_dims());
        Array_gpu<Float,3> radn_dn(flux_dn.get_dims());
        Array_gpu<Float,3> radn_up(flux_dn.get_dims());
        Array_gpu<Float,3> radn_up_jac(flux_dn.get_dims());

        const int block_col2d = 64;
        const int block_gpt2d = 2;

        const int grid_col2d = ncol/block_col2d + (ncol%block_col2d > 0);
        const int grid_gpt2d = ngpt/block_gpt2d + (ngpt%block_gpt2d > 0);

        dim3 grid_gpu2d(grid_col2d, grid_gpt2d);
        dim3 block_gpu2d(block_col2d, block_gpt2d);

        const int block_col3d = 96;
        const int block_lay3d = 1;
        const int block_gpt3d = 1;

        const int grid_col3d = ncol/block_col3d + (ncol%block_col3d > 0);
        const int grid_lay3d = (nlay+1)/block_lay3d + ((nlay+1)%block_lay3d > 0);
        const int grid_gpt3d = ngpt/block_gpt3d + (ngpt%block_gpt3d > 0);

        dim3 grid_gpu3d(grid_col3d, grid_lay3d, grid_gpt3d);
        dim3 block_gpu3d(block_col3d, block_lay3d, block_gpt3d);

        const int top_level = top_at_1 ? 0 : nlay;


        // Upper boundary condition.
        if (inc_flux.size() == 0)
            rte_kernel_launcher_cuda::apply_BC(ncol, nlay, ngpt, top_at_1, flux_dn);
        else
            rte_kernel_launcher_cuda::apply_BC(ncol, nlay, ngpt, top_at_1, inc_flux, flux_dn);


        // Step 1.
        Tuner_map& tunings = Tuner::get().get_map(calling_class_ptr);

        dim3 grid_1, block_1;

        if (tunings.count("lw_step_1") == 0)
        {
            std::tie(grid_1, block_1) = tune_kernel(
                    "lw_step_1",
                    dim3(ncol, nlay, ngpt),
                    {8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024}, {1, 2, 4, 8}, {1},
                    lw_solver_noscat_step_1_kernel,
                    ncol, nlay, ngpt, eps, top_at_1,
                    secants.ptr(), weights.ptr(), tau.ptr(), lay_source.ptr(),
                    lev_source_inc.ptr(), lev_source_dec.ptr(),
                    sfc_emis.ptr(), sfc_src.ptr(), flux_up.ptr(), flux_dn.ptr(), sfc_src_jac.ptr(),
                    flux_up_jac.ptr(), tau_loc.ptr(), trans.ptr(),
                    source_dn.ptr(), source_up.ptr(), source_sfc.ptr(), sfc_albedo.ptr(), source_sfc_jac.ptr());

            tunings["lw_step_1"].first = grid_1;
            tunings["lw_step_1"].second = block_1;
        }
        else
        {
            grid_1 = tunings["lw_step_1"].first;
            block_1 = tunings["lw_step_1"].second;
        }

        lw_solver_noscat_step_1_kernel<<<grid_1, block_1>>>(
                ncol, nlay, ngpt, eps, top_at_1,
                secants.ptr(), weights.ptr(), tau.ptr(), lay_source.ptr(),
                lev_source_inc.ptr(), lev_source_dec.ptr(),
                sfc_emis.ptr(), sfc_src.ptr(), flux_up.ptr(), flux_dn.ptr(), sfc_src_jac.ptr(),
                flux_up_jac.ptr(), tau_loc.ptr(), trans.ptr(),
                source_dn.ptr(), source_up.ptr(), source_sfc.ptr(), sfc_albedo.ptr(), source_sfc_jac.ptr());


        // Step 2.
        dim3 grid_2, block_2;

        if (tunings.count("lw_step_2") == 0)
        {
            std::tie(grid_2, block_2) = tune_kernel(
                    "lw_step_2",
                    dim3(ncol, ngpt),
                    {64, 128, 256, 384, 512, 768, 1024}, {1, 2, 4}, {1},
                    lw_solver_noscat_step_2_kernel,
                    ncol, nlay, ngpt, eps, top_at_1,
                    secants.ptr(), weights.ptr(), tau.ptr(), lay_source.ptr(),
                    lev_source_inc.ptr(), lev_source_dec.ptr(),
                    sfc_emis.ptr(), sfc_src.ptr(), flux_up.ptr(), flux_dn.ptr(), sfc_src_jac.ptr(),
                    flux_up_jac.ptr(), tau_loc.ptr(), trans.ptr(),
                    source_dn.ptr(), source_up.ptr(), source_sfc.ptr(), sfc_albedo.ptr(), source_sfc_jac.ptr());

            tunings["lw_step_2"].first = grid_2;
            tunings["lw_step_2"].second = block_2;
        }
        else
        {
            grid_2 = tunings["lw_step_2"].first;
            block_2 = tunings["lw_step_2"].second;
        }

        lw_solver_noscat_step_2_kernel<<<grid_2, block_2>>>(
                ncol, nlay, ngpt, eps, top_at_1,
                secants.ptr(), weights.ptr(), tau.ptr(), lay_source.ptr(),
                lev_source_inc.ptr(), lev_source_dec.ptr(),
                sfc_emis.ptr(), sfc_src.ptr(), flux_up.ptr(), flux_dn.ptr(), sfc_src_jac.ptr(),
                flux_up_jac.ptr(), tau_loc.ptr(), trans.ptr(),
                source_dn.ptr(), source_up.ptr(), source_sfc.ptr(), sfc_albedo.ptr(), source_sfc_jac.ptr());


        // Step 3.
        dim3 grid_3, block_3;

        if (tunings.count("lw_step_3") == 0)
        {
            std::tie(grid_3, block_3) = tune_kernel(
                    "lw_step_3",
                    dim3(ncol, nlay+1, ngpt),
                    {8, 16, 24, 32, 48, 64, 96, 128, 256}, {1, 2, 4, 8}, {1},
                    lw_solver_noscat_step_3_kernel,
                    ncol, nlay, ngpt, eps, top_at_1,
                    secants.ptr(), weights.ptr(), tau.ptr(), lay_source.ptr(),
                    lev_source_inc.ptr(), lev_source_dec.ptr(),
                    sfc_emis.ptr(), sfc_src.ptr(), flux_up.ptr(), flux_dn.ptr(), sfc_src_jac.ptr(),
                    flux_up_jac.ptr(), tau_loc.ptr(), trans.ptr(),
                    source_dn.ptr(), source_up.ptr(), source_sfc.ptr(), sfc_albedo.ptr(), source_sfc_jac.ptr());

            tunings["lw_step_3"].first = grid_3;
            tunings["lw_step_3"].second = block_3;
        }
        else
        {
            grid_3 = tunings["lw_step_3"].first;
            block_3 = tunings["lw_step_3"].second;
        }

        lw_solver_noscat_step_3_kernel<<<grid_3, block_3>>>(
                ncol, nlay, ngpt, eps, top_at_1,
                secants.ptr(), weights.ptr(), tau.ptr(), lay_source.ptr(),
                lev_source_inc.ptr(), lev_source_dec.ptr(),
                sfc_emis.ptr(), sfc_src.ptr(), flux_up.ptr(), flux_dn.ptr(), sfc_src_jac.ptr(),
                flux_up_jac.ptr(), tau_loc.ptr(), trans.ptr(),
                source_dn.ptr(), source_up.ptr(), source_sfc.ptr(), sfc_albedo.ptr(), source_sfc_jac.ptr());

        apply_BC_kernel_lw<<<grid_gpu2d, block_gpu2d>>>(top_level, ncol, nlay, ngpt, top_at_1, flux_dn.ptr(), radn_dn.ptr());

        if (nmus > 1)
        {
            for (int imu=1; imu<nmus; ++imu)
            {
                throw std::runtime_error("Not implemented due to lacking test case");
                /*
                lw_solver_noscat_step_1_kernel<<<grid_1, block_1>>>(
                        ncol, nlay, ngpt, eps, top_at_1,
                        secants.ptr()+imu, weights.ptr()+imu, tau.ptr(), lay_source.ptr(),
                        lev_source_inc.ptr(), lev_source_dec.ptr(),
                        sfc_emis.ptr(), sfc_src.ptr(), radn_up.ptr(), radn_dn.ptr(), sfc_src_jac.ptr(),
                        radn_up_jac.ptr(), tau_loc.ptr(), trans.ptr(),
                        source_dn.ptr(), source_up.ptr(), source_sfc.ptr(), sfc_albedo.ptr(), source_sfc_jac.ptr());

                lw_solver_noscat_step_2_kernel<<<grid_2, block_2>>>(
                        ncol, nlay, ngpt, eps, top_at_1,
                        secants.ptr()+imu, weights.ptr()+imu, tau.ptr(), lay_source.ptr(),
                        lev_source_inc.ptr(), lev_source_dec.ptr(),
                        sfc_emis.ptr(), sfc_src.ptr(),
                        radn_up.ptr(), radn_dn.ptr(), sfc_src_jac.ptr(),
                        radn_up_jac.ptr(), tau_loc.ptr(), trans.ptr(),
                        source_dn.ptr(), source_up.ptr(), source_sfc.ptr(), sfc_albedo.ptr(), source_sfc_jac.ptr());

                lw_solver_noscat_step_3_kernel<<<grid_3, block_3>>>(
                        ncol, nlay, ngpt, eps, top_at_1,
                        secants.ptr()+imu, weights.ptr()+imu, tau.ptr(), lay_source.ptr(),
                        lev_source_inc.ptr(), lev_source_dec.ptr(),
                        sfc_emis.ptr(), sfc_src.ptr(), radn_up.ptr(), radn_dn.ptr(), sfc_src_jac.ptr(),
                        radn_up_jac.ptr(), tau_loc.ptr(), trans.ptr(),
                        source_dn.ptr(), source_up.ptr(), source_sfc.ptr(), sfc_albedo.ptr(), source_sfc_jac.ptr());

                add_fluxes_kernel<<<grid_gpu3d, block_gpu3d>>>(
                        ncol, nlay+1, ngpt,
                        radn_up.ptr(), radn_dn.ptr(), radn_up_jac.ptr(),
                        flux_up.ptr(), flux_dn.ptr(), flux_up_jac.ptr());
                        */
            }
        }
    }


    void sw_solver_2stream(
            const int ncol, const int nlay, const int ngpt, const Bool top_at_1,
            const Array_gpu<Float,3>& tau, const Array_gpu<Float,3>& ssa, const Array_gpu<Float,3>& g,
            const Array_gpu<Float,1>& mu0,
            const Array_gpu<Float,2>& sfc_alb_dir, const Array_gpu<Float,2>& sfc_alb_dif,
            const Array_gpu<Float,2>& inc_flux_dir,
            Array_gpu<Float,3>& flux_up, Array_gpu<Float,3>& flux_dn, Array_gpu<Float,3>& flux_dir,
            const Bool has_dif_bc, const Array_gpu<Float,2>& inc_flux_dif,
            const Bool do_broadband, Array_gpu<Float,3>& flux_up_loc, Array_gpu<Float,3>& flux_dn_loc, Array_gpu<Float,3>& flux_dir_loc,
            void* calling_class_ptr)
    {
        const int opt_size = tau.size();
        const int alb_size = sfc_alb_dir.size();
        const int flx_size = flux_up.size();

        Float* r_dif = Tools_gpu::allocate_gpu<Float>(opt_size);
        Float* t_dif = Tools_gpu::allocate_gpu<Float>(opt_size);
        Float* r_dir = nullptr;
        Float* t_dir = nullptr;
        Float* t_noscat = nullptr;
        Float* source_up = Tools_gpu::allocate_gpu<Float>(opt_size);
        Float* source_dn = Tools_gpu::allocate_gpu<Float>(opt_size);
        Float* source_sfc = Tools_gpu::allocate_gpu<Float>(alb_size);
        Float* albedo = Tools_gpu::allocate_gpu<Float>(flx_size);
        Float* src = Tools_gpu::allocate_gpu<Float>(flx_size);
        Float* denom = Tools_gpu::allocate_gpu<Float>(opt_size);

        dim3 grid_source(ncol, ngpt), block_source;

        // Step0. Upper boundary condition. At this stage, flux_dn contains the diffuse radiation only.
        rte_kernel_launcher_cuda::apply_BC(ncol, nlay, ngpt, top_at_1, inc_flux_dir, mu0, flux_dir);
        if (inc_flux_dif.size() == 0)
            rte_kernel_launcher_cuda::apply_BC(ncol, nlay, ngpt, top_at_1, flux_dn);
        else
            rte_kernel_launcher_cuda::apply_BC(ncol, nlay, ngpt, top_at_1, inc_flux_dif, flux_dn);


        // Step 1.
        Tuner_map& tunings = Tuner::get().get_map(calling_class_ptr);

        if (tunings.count("sw_source_2stream_kernel") == 0)
        {
            if (top_at_1)
            {
                std::tie(grid_source, block_source) = tune_kernel(
                        "sw_source_2stream_kernel",
                        dim3(ncol, ngpt),
                        {8, 16, 32, 64, 96, 128, 256, 384, 512, 768, 1024}, {1, 2, 4, 8, 16}, {1},
                        sw_source_2stream_kernel<1>,
                        ncol, nlay, ngpt, tau.ptr(), ssa.ptr(), g.ptr(), mu0.ptr(), r_dif, t_dif,
                        sfc_alb_dir.ptr(), source_up, source_dn, source_sfc, flux_dir.ptr());
            }
            else
            {
                std::tie(grid_source, block_source) = tune_kernel(
                        "sw_source_2stream_kernel",
                        dim3(ncol, ngpt),
                        {8, 16, 32, 64, 96, 128, 256, 384, 512, 768, 1024}, {1, 2, 4, 8, 16}, {1},
                        sw_source_2stream_kernel<0>,
                        ncol, nlay, ngpt, tau.ptr(), ssa.ptr(), g.ptr(), mu0.ptr(), r_dif, t_dif,
                        sfc_alb_dir.ptr(), source_up, source_dn, source_sfc, flux_dir.ptr());
            }

            tunings["sw_source_2stream_kernel"].first = grid_source;
            tunings["sw_source_2stream_kernel"].second = block_source;
        }
        else
        {
            grid_source = tunings["sw_source_2stream_kernel"].first;
            block_source = tunings["sw_source_2stream_kernel"].second;
        }

        if (top_at_1)
        {
            sw_source_2stream_kernel<1><<<grid_source, block_source>>>(
                    ncol, nlay, ngpt, tau.ptr(), ssa.ptr(), g.ptr(), mu0.ptr(), r_dif, t_dif,
                    sfc_alb_dir.ptr(), source_up, source_dn, source_sfc, flux_dir.ptr());
        }
        else
        {
            sw_source_2stream_kernel<0><<<grid_source, block_source>>>(
                    ncol, nlay, ngpt, tau.ptr(), ssa.ptr(), g.ptr(), mu0.ptr(), r_dif, t_dif,
                    sfc_alb_dir.ptr(), source_up, source_dn, source_sfc, flux_dir.ptr());
        }


        // Step 2.
        dim3 grid_adding, block_adding;

        if (tunings.count("sw_adding") == 0)
        {
            if (top_at_1)
            {
                std::tie(grid_adding, block_adding) = tune_kernel(
                        "sw_adding",
                        dim3(ncol, ngpt),
                        {8, 16, 32, 64, 96, 128, 256, 384, 512, 768, 1024}, {1, 2, 4, 8, 16}, {1},
                        sw_adding_kernel<1>,
                        ncol, nlay, ngpt, top_at_1,
                        sfc_alb_dif.ptr(), r_dif, t_dif,
                        source_dn, source_up, source_sfc,
                        flux_up.ptr(), flux_dn.ptr(), flux_dir.ptr(), albedo, src, denom);
            }
            else
            {
                std::tie(grid_adding, block_adding) = tune_kernel(
                        "sw_adding",
                        dim3(ncol, ngpt),
                        {8, 16, 32, 64, 96, 128, 256, 384, 512, 768, 1024}, {1, 2, 4, 8, 16}, {1},
                        sw_adding_kernel<0>,
                        ncol, nlay, ngpt, top_at_1,
                        sfc_alb_dif.ptr(), r_dif, t_dif,
                        source_dn, source_up, source_sfc,
                        flux_up.ptr(), flux_dn.ptr(), flux_dir.ptr(), albedo, src, denom);
            }

            tunings["sw_adding"].first = grid_adding;
            tunings["sw_adding"].second = block_adding;
        }
        else
        {
            grid_adding = tunings["sw_adding"].first;
            block_adding = tunings["sw_adding"].second;
        }

        if (top_at_1)
        {
            sw_adding_kernel<1><<<grid_adding, block_adding>>>(
                ncol, nlay, ngpt, top_at_1,
                sfc_alb_dif.ptr(), r_dif, t_dif,
                source_dn, source_up, source_sfc,
                flux_up.ptr(), flux_dn.ptr(), flux_dir.ptr(), albedo, src, denom);
        }
        else
        {
            sw_adding_kernel<0><<<grid_adding, block_adding>>>(
                        ncol, nlay, ngpt, top_at_1,
                        sfc_alb_dif.ptr(), r_dif, t_dif,
                        source_dn, source_up, source_sfc,
                        flux_up.ptr(), flux_dn.ptr(), flux_dir.ptr(), albedo, src, denom);
        }

        Tools_gpu::free_gpu(r_dif);
        Tools_gpu::free_gpu(t_dif);
        Tools_gpu::free_gpu(source_up);
        Tools_gpu::free_gpu(source_dn);
        Tools_gpu::free_gpu(source_sfc);
        Tools_gpu::free_gpu(albedo);
        Tools_gpu::free_gpu(src);
        Tools_gpu::free_gpu(denom);
    }
}
