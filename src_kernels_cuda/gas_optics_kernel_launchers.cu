#include <chrono>
#include <functional>
#include <iostream>
#include <iomanip>

#include "rrtmgp_kernel_launcher_cuda.h"
#include "tools_gpu.h"
#include "Array.h"
#include "tuner.h"


namespace
{
    #include "gas_optics_kernels.cu"
}


namespace rrtmgp_kernel_launcher_cuda
{
    template<typename TF>
    void reorder123x321(const int ni, const int nj, const int nk,
                        const Array_gpu<TF,3>& arr_in, Array_gpu<TF,3>& arr_out)
    {
        const int block_i = 8;
        const int block_j = 4;
        const int block_k = 16;

        const int grid_i = ni/block_i + (ni%block_i > 0);
        const int grid_j = nj/block_j + (nj%block_j > 0);
        const int grid_k = nk/block_k + (nk%block_k > 0);

        dim3 grid_gpu(grid_i, grid_j, grid_k);
        dim3 block_gpu(block_i, block_j, block_k);

        reorder123x321_kernel<<<grid_gpu, block_gpu>>>(
                ni, nj, nk, arr_in.ptr(), arr_out.ptr());
    }

    template<typename TF>
    void reorder12x21(const int ni, const int nj,
                      const Array_gpu<TF,2>& arr_in, Array_gpu<TF,2>& arr_out)
    {
        const int block_i = 32;
        const int block_j = 16;

        const int grid_i = ni/block_i + (ni%block_i > 0);
        const int grid_j = nj/block_j + (nj%block_j > 0);

        dim3 grid_gpu(grid_i, grid_j);
        dim3 block_gpu(block_i, block_j);

        reorder12x21_kernel<<<grid_gpu, block_gpu>>>(
                ni, nj, arr_in.ptr(), arr_out.ptr());
    }

    template<typename TF>
    void zero_array(const int ni, const int nj, const int nk, Array_gpu<TF,3>& arr)
    {
        const int block_i = 32;
        const int block_j = 16;
        const int block_k = 1;

        const int grid_i = ni/block_i + (ni%block_i > 0);
        const int grid_j = nj/block_j + (nj%block_j > 0);
        const int grid_k = nk/block_k + (nk%block_k > 0);

        dim3 grid_gpu(grid_i, grid_j, grid_k);
        dim3 block_gpu(block_i, block_j, block_k);

        zero_array_kernel<<<grid_gpu, block_gpu>>>(
                ni, nj, nk, arr.ptr());

    }

    template<typename TF>
    void interpolation(
            const int ncol, const int nlay,
            const int ngas, const int nflav, const int neta, const int npres, const int ntemp,
            const Array_gpu<int,2>& flavor,
            const Array_gpu<TF,1>& press_ref_log,
            const Array_gpu<TF,1>& temp_ref,
            TF press_ref_log_delta,
            TF temp_ref_min,
            TF temp_ref_delta,
            TF press_ref_trop_log,
            const Array_gpu<TF,3>& vmr_ref,
            const Array_gpu<TF,2>& play,
            const Array_gpu<TF,2>& tlay,
            Array_gpu<TF,3>& col_gas,
            Array_gpu<int,2>& jtemp,
            Array_gpu<TF,6>& fmajor, Array_gpu<TF,5>& fminor,
            Array_gpu<TF,4>& col_mix,
            Array_gpu<BOOL_TYPE,2>& tropo,
            Array_gpu<int,4>& jeta,
            Array_gpu<int,2>& jpress)
    {
        const int block_flav = 16;
        const int block_lay  = 2;
        const int block_col  = 4;

        const int grid_flav = nflav/block_flav + (nflav%block_flav > 0);
        const int grid_lay  = nlay /block_lay  + (nlay%block_lay   > 0);
        const int grid_col  = ncol /block_col  + (ncol%block_col   > 0);

        dim3 grid_gpu(grid_flav, grid_col, grid_lay);
        dim3 block_gpu(block_flav, block_col, block_lay);

        TF tmin = std::numeric_limits<TF>::min();
        interpolation_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, ngas, nflav, neta, npres, ntemp, tmin,
                flavor.ptr(), press_ref_log.ptr(), temp_ref.ptr(),
                press_ref_log_delta, temp_ref_min,
                temp_ref_delta, press_ref_trop_log,
                vmr_ref.ptr(), play.ptr(), tlay.ptr(),
                col_gas.ptr(), jtemp.ptr(), fmajor.ptr(),
                fminor.ptr(), col_mix.ptr(), tropo.ptr(),
                jeta.ptr(), jpress.ptr());
    }

    template<typename TF>
    void combine_and_reorder_2str(
            const int ncol, const int nlay, const int ngpt,
            const Array_gpu<TF,3>& tau_abs, const Array_gpu<TF,3>& tau_rayleigh,
            Array_gpu<TF,3>& tau, Array_gpu<TF,3>& ssa, Array_gpu<TF,3>& g)
    {
        const int block_col = 32;
        const int block_gpt = 32;
        const int block_lay = 1;

        const int grid_col = ncol/block_col + (ncol%block_col > 0);
        const int grid_gpt = ngpt/block_gpt + (ngpt%block_gpt > 0);
        const int grid_lay = nlay/block_lay + (nlay%block_lay > 0);

        dim3 grid_gpu(grid_col, grid_gpt, grid_lay);
        dim3 block_gpu(block_col, block_gpt, block_lay);

        TF tmin = std::numeric_limits<TF>::min();
        combine_and_reorder_2str_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, ngpt, tmin,
                tau_abs.ptr(), tau_rayleigh.ptr(),
                tau.ptr(), ssa.ptr(), g.ptr());
    }

    template<typename TF>
    void compute_tau_rayleigh(
            const int ncol, const int nlay, const int nbnd, const int ngpt,
            const int ngas, const int nflav, const int neta, const int npres, const int ntemp,
            const Array_gpu<int,2>& gpoint_flavor,
            const Array_gpu<int,2>& band_lims_gpt,
            const Array_gpu<TF,4>& krayl,
            int idx_h2o, const Array_gpu<TF,2>& col_dry, const Array_gpu<TF,3>& col_gas,
            const Array_gpu<TF,5>& fminor, const Array_gpu<int,4>& jeta,
            const Array_gpu<BOOL_TYPE,2>& tropo, const Array_gpu<int,2>& jtemp,
            Array_gpu<TF,3>& tau_rayleigh)
    {
        TF* k = Tools_gpu::allocate_gpu<TF>(ncol*nlay*ngpt);

        // Call the kernel.
        const int block_bnd = 1;
        const int block_lay = 1;
        const int block_col = 8;

        const int grid_bnd = nbnd/block_bnd + (nbnd%block_bnd > 0);
        const int grid_lay = nlay/block_lay + (nlay%block_lay > 0);
        const int grid_col = ncol/block_col + (ncol%block_col > 0);

        dim3 grid_gpu(grid_bnd, grid_lay, grid_col);
        dim3 block_gpu(block_bnd, block_lay, block_col);

        compute_tau_rayleigh_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, nbnd, ngpt,
                ngas, nflav, neta, npres, ntemp,
                gpoint_flavor.ptr(),
                band_lims_gpt.ptr(),
                krayl.ptr(),
                idx_h2o, col_dry.ptr(), col_gas.ptr(),
                fminor.ptr(), jeta.ptr(),
                tropo.ptr(), jtemp.ptr(),
                tau_rayleigh.ptr(), k);

        Tools_gpu::free_gpu(k);
    }

    template<typename TF>
    struct Compute_tau_minor_absorption_kernel
    {
        template<int I, int J, int K, class... Args>
        static void launch(dim3 grid, dim3 block, Args... args)
        {
            compute_tau_minor_absorption_kernel<TF, I, J, K><<<grid, block>>>(args...);
        }
    };

    template<typename TF>
    void compute_tau_absorption(
            const int ncol, const int nlay, const int nband, const int ngpt,
            const int ngas, const int nflav, const int neta, const int npres, const int ntemp,
            const int nminorlower, const int nminorklower,
            const int nminorupper, const int nminorkupper,
            const int idx_h2o,
            const Array_gpu<int,2>& gpoint_flavor,
            const Array_gpu<int,2>& band_lims_gpt,
            const Array_gpu<TF,4>& kmajor,
            const Array_gpu<TF,3>& kminor_lower,
            const Array_gpu<TF,3>& kminor_upper,
            const Array_gpu<int,2>& minor_limits_gpt_lower,
            const Array_gpu<int,2>& minor_limits_gpt_upper,
            const Array_gpu<BOOL_TYPE,1>& minor_scales_with_density_lower,
            const Array_gpu<BOOL_TYPE,1>& minor_scales_with_density_upper,
            const Array_gpu<BOOL_TYPE,1>& scale_by_complement_lower,
            const Array_gpu<BOOL_TYPE,1>& scale_by_complement_upper,
            const Array_gpu<int,1>& idx_minor_lower,
            const Array_gpu<int,1>& idx_minor_upper,
            const Array_gpu<int,1>& idx_minor_scaling_lower,
            const Array_gpu<int,1>& idx_minor_scaling_upper,
            const Array_gpu<int,1>& kminor_start_lower,
            const Array_gpu<int,1>& kminor_start_upper,
            const Array_gpu<BOOL_TYPE,2>& tropo,
            const Array_gpu<TF,4>& col_mix, const Array_gpu<TF,6>& fmajor,
            const Array_gpu<TF,5>& fminor, const Array_gpu<TF,2>& play,
            const Array_gpu<TF,2>& tlay, const Array_gpu<TF,3>& col_gas,
            const Array_gpu<int,4>& jeta, const Array_gpu<int,2>& jtemp,
            const Array_gpu<int,2>& jpress,
            Array_gpu<TF,3>& tau,
            Tuner_map& tunings)
    {
        Array_gpu<TF,3> tau_major(tau.get_dims());
        Array_gpu<TF,3> tau_minor(tau.get_dims());

        dim3 grid_gpu_maj{ncol, nlay, nband}, block_gpu_maj;


        if (tunings.count("compute_tau_major_absorption_kernel") == 0)
        {
            std::tie(grid_gpu_maj, block_gpu_maj) = tune_kernel(
                    "compute_tau_major_absorption_kernel",
                    {ncol, nlay, nband}, {1, 2, 4, 8, 16}, {1, 2, 4, 8, 16}, {1, 2, 4, 8, 16},
                    compute_tau_major_absorption_kernel<TF>,
                    ncol, nlay, nband, ngpt,
                    nflav, neta, npres, ntemp,
                    gpoint_flavor.ptr(), band_lims_gpt.ptr(),
                    kmajor.ptr(), col_mix.ptr(), fmajor.ptr(), jeta.ptr(),
                    tropo.ptr(), jtemp.ptr(), jpress.ptr(),
                    Array_gpu<TF,3>(tau).ptr(), Array_gpu<TF,3>(tau_major).ptr());

            tunings["compute_tau_major_absorption_kernel"].first = grid_gpu_maj;
            tunings["compute_tau_major_absorption_kernel"].second = block_gpu_maj;
        }
        else
        {
            grid_gpu_maj = tunings["compute_tau_major_absorption_kernel"].first;
            block_gpu_maj = tunings["compute_tau_major_absorption_kernel"].second;
        }

        compute_tau_major_absorption_kernel<<<grid_gpu_maj, block_gpu_maj>>>(
                ncol, nlay, nband, ngpt,
                nflav, neta, npres, ntemp,
                gpoint_flavor.ptr(), band_lims_gpt.ptr(),
                kmajor.ptr(), col_mix.ptr(), fmajor.ptr(), jeta.ptr(),
                tropo.ptr(), jtemp.ptr(), jpress.ptr(),
                tau.ptr(), tau_major.ptr());

        const int nscale_lower = scale_by_complement_lower.dim(1);
        const int nscale_upper = scale_by_complement_upper.dim(1);


        // Lower
        int idx_tropo = 1;

        dim3 grid_gpu_min_1{ncol, nlay, ngpt}, block_gpu_min_1;


        if (tunings.count("compute_tau_minor_absorption_kernel_lower") == 0)
        {
            std::tie(grid_gpu_min_1, block_gpu_min_1) =
                tune_kernel_compile_time<Compute_tau_minor_absorption_kernel<TF>>(
                        "compute_tau_minor_absorption_kernel_lower",
                        {ncol, nlay},
                        std::integer_sequence<int, 1, 2, 4, 8, 16>{},
                        std::integer_sequence<int, 1, 2, 4, 8, 16>{},
                        std::integer_sequence<int, 1>{},
                        ncol, nlay, ngpt,
                        ngas, nflav, ntemp, neta,
                        nscale_lower,
                        nminorlower,
                        nminorklower,
                        idx_h2o, idx_tropo,
                        gpoint_flavor.ptr(),
                        kminor_lower.ptr(),
                        minor_limits_gpt_lower.ptr(),
                        minor_scales_with_density_lower.ptr(),
                        scale_by_complement_lower.ptr(),
                        idx_minor_lower.ptr(),
                        idx_minor_scaling_lower.ptr(),
                        kminor_start_lower.ptr(),
                        play.ptr(), tlay.ptr(), col_gas.ptr(),
                        fminor.ptr(), jeta.ptr(), jtemp.ptr(),
                        tropo.ptr(), tau.ptr(), tau_minor.ptr());

            tunings["compute_tau_minor_absorption_kernel_lower"].first = grid_gpu_min_1;
            tunings["compute_tau_minor_absorption_kernel_lower"].second = block_gpu_min_1;
        }
        else
        {
            grid_gpu_min_1 = tunings["compute_tau_minor_absorption_kernel_lower"].first;
            block_gpu_min_1 = tunings["compute_tau_minor_absorption_kernel_lower"].second;
        }

        run_kernel_compile_time<Compute_tau_minor_absorption_kernel<TF>>(
                std::integer_sequence<int, 1, 2, 4, 8, 16>{},
                std::integer_sequence<int, 1, 2, 4, 8, 16>{},
                std::integer_sequence<int, 1>{},
                grid_gpu_min_1, block_gpu_min_1,
                ncol, nlay, ngpt,
                ngas, nflav, ntemp, neta,
                nscale_lower,
                nminorlower,
                nminorklower,
                idx_h2o, idx_tropo,
                gpoint_flavor.ptr(),
                kminor_lower.ptr(),
                minor_limits_gpt_lower.ptr(),
                minor_scales_with_density_lower.ptr(),
                scale_by_complement_lower.ptr(),
                idx_minor_lower.ptr(),
                idx_minor_scaling_lower.ptr(),
                kminor_start_lower.ptr(),
                play.ptr(), tlay.ptr(), col_gas.ptr(),
                fminor.ptr(), jeta.ptr(), jtemp.ptr(),
                tropo.ptr(), tau.ptr(), tau_minor.ptr());


        // Upper
        idx_tropo = 0;

        dim3 grid_gpu_min_2{ncol, nlay, ngpt}, block_gpu_min_2;


        if (tunings.count("compute_tau_minor_absorption_kernel_upper") == 0)
        {
            std::tie(grid_gpu_min_2, block_gpu_min_2) =
                tune_kernel_compile_time<Compute_tau_minor_absorption_kernel<TF>>(
                        "compute_tau_minor_absorption_kernel_upper",
                        {ncol, nlay},
                        std::integer_sequence<int, 1, 2, 4, 8, 16>{},
                        std::integer_sequence<int, 1, 2, 4, 8, 16>{},
                        std::integer_sequence<int, 1>{},
                        ncol, nlay, ngpt,
                        ngas, nflav, ntemp, neta,
                        nscale_upper,
                        nminorupper,
                        nminorkupper,
                        idx_h2o, idx_tropo,
                        gpoint_flavor.ptr(),
                        kminor_upper.ptr(),
                        minor_limits_gpt_upper.ptr(),
                        minor_scales_with_density_upper.ptr(),
                        scale_by_complement_upper.ptr(),
                        idx_minor_upper.ptr(),
                        idx_minor_scaling_upper.ptr(),
                        kminor_start_upper.ptr(),
                        play.ptr(), tlay.ptr(), col_gas.ptr(),
                        fminor.ptr(), jeta.ptr(), jtemp.ptr(),
                        tropo.ptr(), tau.ptr(), tau_minor.ptr());

            tunings["compute_tau_minor_absorption_kernel_upper"].first = grid_gpu_min_2;
            tunings["compute_tau_minor_absorption_kernel_upper"].second = block_gpu_min_2;
        }
        else
        {
            grid_gpu_min_2 = tunings["compute_tau_minor_absorption_kernel_upper"].first;
            block_gpu_min_2 = tunings["compute_tau_minor_absorption_kernel_upper"].second;
        }

        run_kernel_compile_time<Compute_tau_minor_absorption_kernel<TF>>(
                std::integer_sequence<int, 1, 2, 4, 8, 16>{},
                std::integer_sequence<int, 1, 2, 4, 8, 16>{},
                std::integer_sequence<int, 1>{},
                grid_gpu_min_2, block_gpu_min_2,
                ncol, nlay, ngpt,
                ngas, nflav, ntemp, neta,
                nscale_upper,
                nminorupper,
                nminorkupper,
                idx_h2o, idx_tropo,
                gpoint_flavor.ptr(),
                kminor_upper.ptr(),
                minor_limits_gpt_upper.ptr(),
                minor_scales_with_density_upper.ptr(),
                scale_by_complement_upper.ptr(),
                idx_minor_upper.ptr(),
                idx_minor_scaling_upper.ptr(),
                kminor_start_upper.ptr(),
                play.ptr(), tlay.ptr(), col_gas.ptr(),
                fminor.ptr(), jeta.ptr(), jtemp.ptr(),
                tropo.ptr(), tau.ptr(), tau_minor.ptr());

    }

    template<typename TF>
    void Planck_source(
            const int ncol, const int nlay, const int nbnd, const int ngpt,
            const int nflav, const int neta, const int npres, const int ntemp,
            const int nPlanckTemp,
            const Array_gpu<TF,2>& tlay,
            const Array_gpu<TF,2>& tlev,
            const Array_gpu<TF,1>& tsfc,
            const int sfc_lay,
            const Array_gpu<TF,6>& fmajor,
            const Array_gpu<int,4>& jeta,
            const Array_gpu<BOOL_TYPE,2>& tropo,
            const Array_gpu<int,2>& jtemp,
            const Array_gpu<int,2>& jpress,
            const Array_gpu<int,1>& gpoint_bands,
            const Array_gpu<int,2>& band_lims_gpt,
            const Array_gpu<TF,4>& pfracin,
            const TF temp_ref_min, const TF totplnk_delta,
            const Array_gpu<TF,2>& totplnk,
            const Array_gpu<int,2>& gpoint_flavor,
            Array_gpu<TF,2>& sfc_src,
            Array_gpu<TF,3>& lay_src,
            Array_gpu<TF,3>& lev_src_inc,
            Array_gpu<TF,3>& lev_src_dec,
            Array_gpu<TF,2>& sfc_src_jac,
            Tuner_map& tunings)
    {
        TF ones_cpu[2] = {TF(1.), TF(1.)};
        const TF delta_Tsurf = TF(1.);

        TF* pfrac = Tools_gpu::allocate_gpu<TF>(lay_src.size());
        TF* ones = Tools_gpu::allocate_gpu<TF>(2);

        // Copy the data to the GPU.
        cuda_safe_call(cudaMemcpy(ones, ones_cpu, 2*sizeof(TF), cudaMemcpyHostToDevice));

        // Call the kernel.
        dim3 grid_gpu{nbnd, nlay, ncol}, block_gpu;


        if (tunings.count("Planck_source_kernel") == 0)
        {
            std::tie(grid_gpu, block_gpu) = tune_kernel(
                    "Planck_source_kernel",
                    {nbnd, nlay, ncol}, {1, 2, 3, 4, 8}, {1, 2, 3, 4, 8}, {1, 2, 3, 4, 8},
                    Planck_source_kernel<TF>,
                    ncol, nlay, nbnd, ngpt,
                    nflav, neta, npres, ntemp, nPlanckTemp,
                    tlay.ptr(), tlev.ptr(), tsfc.ptr(), sfc_lay,
                    fmajor.ptr(), jeta.ptr(), tropo.ptr(), jtemp.ptr(),
                    jpress.ptr(), gpoint_bands.ptr(), band_lims_gpt.ptr(),
                    pfracin.ptr(), temp_ref_min, totplnk_delta,
                    totplnk.ptr(), gpoint_flavor.ptr(), ones,
                    delta_Tsurf, sfc_src.ptr(), lay_src.ptr(),
                    lev_src_inc.ptr(), lev_src_dec.ptr(),
                    sfc_src_jac.ptr(), pfrac);

            tunings["Planck_source_kernel"].first = grid_gpu;
            tunings["Planck_source_kernel"].second = block_gpu;
        }
        else
        {
            grid_gpu = tunings["Planck_source_kernel"].first;
            block_gpu = tunings["Planck_source_kernel"].second;
        }

        Planck_source_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, nbnd, ngpt,
                nflav, neta, npres, ntemp, nPlanckTemp,
                tlay.ptr(), tlev.ptr(), tsfc.ptr(), sfc_lay,
                fmajor.ptr(), jeta.ptr(), tropo.ptr(), jtemp.ptr(),
                jpress.ptr(), gpoint_bands.ptr(), band_lims_gpt.ptr(),
                pfracin.ptr(), temp_ref_min, totplnk_delta,
                totplnk.ptr(), gpoint_flavor.ptr(), ones,
                delta_Tsurf, sfc_src.ptr(), lay_src.ptr(),
                lev_src_inc.ptr(), lev_src_dec.ptr(),
                sfc_src_jac.ptr(), pfrac);

        Tools_gpu::free_gpu(pfrac);
        Tools_gpu::free_gpu(ones);
    }
}


#ifdef RTE_RRTMGP_SINGLE_PRECISION
template void rrtmgp_kernel_launcher_cuda::reorder123x321<float>(const int, const int, const int, const Array_gpu<float,3>&, Array_gpu<float,3>&);
template void rrtmgp_kernel_launcher_cuda::reorder12x21<float>(const int, const int, const Array_gpu<float,2>&, Array_gpu<float,2>&);

template void rrtmgp_kernel_launcher_cuda::zero_array<float>(const int, const int, const int, Array_gpu<float,3>&);

template void rrtmgp_kernel_launcher_cuda::interpolation<float>(
        const int, const int, const int, const int, const int, const int, const int,
        const Array_gpu<int,2>&, const Array_gpu<float,1>&, const Array_gpu<float,1>&,
        float, float, float, float, const Array_gpu<float,3>&, const Array_gpu<float,2>&,
        const Array_gpu<float,2>&, Array_gpu<float,3>&, Array_gpu<int,2>&, Array_gpu<float,6>&, Array_gpu<float,5>&,
        Array_gpu<float,4>&, Array_gpu<BOOL_TYPE,2>&, Array_gpu<int,4>&, Array_gpu<int,2>&);

template void rrtmgp_kernel_launcher_cuda::combine_and_reorder_2str<float>(
        const int, const int, const int, const Array_gpu<float,3>&, const Array_gpu<float,3>&, Array_gpu<float,3>&, Array_gpu<float,3>&, Array_gpu<float,3>&);

template void rrtmgp_kernel_launcher_cuda::compute_tau_rayleigh<float>(
        const int, const int, const int, const int, const int, const int, const int, const int, const int,
        const Array_gpu<int,2>&, const Array_gpu<int,2>&, const Array_gpu<float,4>&, int, const Array_gpu<float,2>&,
        const Array_gpu<float,3>&, const Array_gpu<float,5>&, const Array_gpu<int,4>&, const Array_gpu<BOOL_TYPE,2>&,
        const Array_gpu<int,2>&, Array_gpu<float,3>&);

template void rrtmgp_kernel_launcher_cuda::compute_tau_absorption<float>(const int, const int, const int, const int, const int, const int,
        const int, const int, const int, const int, const int, const int, const int, const int,
        const Array_gpu<int,2>&, const Array_gpu<int,2>&, const Array_gpu<float,4>&, const Array_gpu<float,3>&, const Array_gpu<float,3>&,
        const Array_gpu<int,2>&, const Array_gpu<int,2>&, const Array_gpu<BOOL_TYPE,1>&, const Array_gpu<BOOL_TYPE,1>&,
        const Array_gpu<BOOL_TYPE,1>&, const Array_gpu<BOOL_TYPE,1>&, const Array_gpu<int,1>&, const Array_gpu<int,1>&,
        const Array_gpu<int,1>&, const Array_gpu<int,1>&, const Array_gpu<int,1>&, const Array_gpu<int,1>&, const Array_gpu<BOOL_TYPE,2>& tropo,
        const Array_gpu<float,4>&, const Array_gpu<float,6>&, const Array_gpu<float,5>&, const Array_gpu<float,2>&, const Array_gpu<float,2>&, const Array_gpu<float,3>&,
        const Array_gpu<int,4>&, const Array_gpu<int,2>&, const Array_gpu<int,2>&, Array_gpu<float,3>&, Tuner_map& tunings);

template void rrtmgp_kernel_launcher_cuda::Planck_source<float>(const int ncol, const int nlay, const int nbnd, const int ngpt,
        const int nflav, const int neta, const int npres, const int ntemp,
        const int nPlanckTemp, const Array_gpu<float,2>& tlay, const Array_gpu<float,2>& tlev,
        const Array_gpu<float,1>& tsfc, const int sfc_lay, const Array_gpu<float,6>& fmajor,
        const Array_gpu<int,4>& jeta, const Array_gpu<BOOL_TYPE,2>& tropo, const Array_gpu<int,2>& jtemp,
        const Array_gpu<int,2>& jpress, const Array_gpu<int,1>& gpoint_bands, const Array_gpu<int,2>& band_lims_gpt,
        const Array_gpu<float,4>& pfracin, const float temp_ref_min, const float totplnk_delta,
        const Array_gpu<float,2>& totplnk, const Array_gpu<int,2>& gpoint_flavor,
        Array_gpu<float,2>& sfc_src,  Array_gpu<float,3>& lay_src, Array_gpu<float,3>& lev_src_inc,
        Array_gpu<float,3>& lev_src_dec, Array_gpu<float,2>& sfc_src_jac, Tuner_map& tunings);

#else
template void rrtmgp_kernel_launcher_cuda::reorder123x321<double>(const int, const int, const int, const Array_gpu<double,3>&, Array_gpu<double,3>&);

template void rrtmgp_kernel_launcher_cuda::reorder12x21<double>(const int, const int, const Array_gpu<double,2>&, Array_gpu<double,2>&);

template void rrtmgp_kernel_launcher_cuda::zero_array<double>(const int, const int, const int, Array_gpu<double,3>&);

template void rrtmgp_kernel_launcher_cuda::interpolation<double>(
        const int, const int, const int, const int, const int, const int, const int,
        const Array_gpu<int,2>&, const Array_gpu<double,1>&, const Array_gpu<double,1>&,
        double, double, double, double, const Array_gpu<double,3>&, const Array_gpu<double,2>&,
        const Array_gpu<double,2>&, Array_gpu<double,3>&, Array_gpu<int,2>&, Array_gpu<double,6>&, Array_gpu<double,5>&,
        Array_gpu<double,4>&, Array_gpu<BOOL_TYPE,2>&, Array_gpu<int,4>&, Array_gpu<int,2>&);

template void rrtmgp_kernel_launcher_cuda::combine_and_reorder_2str<double>(
        const int, const int, const int, const Array_gpu<double,3>&, const Array_gpu<double,3>&, Array_gpu<double,3>&, Array_gpu<double,3>&, Array_gpu<double,3>&);

template void rrtmgp_kernel_launcher_cuda::compute_tau_rayleigh<double>(
        const int, const int, const int, const int, const int, const int, const int, const int, const int,
        const Array_gpu<int,2>&, const Array_gpu<int,2>&, const Array_gpu<double,4>&, int, const Array_gpu<double,2>&,
        const Array_gpu<double,3>&, const Array_gpu<double,5>&, const Array_gpu<int,4>&, const Array_gpu<BOOL_TYPE,2>&,
        const Array_gpu<int,2>&, Array_gpu<double,3>&);

template void rrtmgp_kernel_launcher_cuda::compute_tau_absorption<double>(const int, const int, const int, const int, const int, const int,
        const int, const int, const int, const int, const int, const int, const int, const int,
        const Array_gpu<int,2>&, const Array_gpu<int,2>&, const Array_gpu<double,4>&, const Array_gpu<double,3>&, const Array_gpu<double,3>&,
        const Array_gpu<int,2>&, const Array_gpu<int,2>&, const Array_gpu<BOOL_TYPE,1>&, const Array_gpu<BOOL_TYPE,1>&,
        const Array_gpu<BOOL_TYPE,1>&, const Array_gpu<BOOL_TYPE,1>&, const Array_gpu<int,1>&, const Array_gpu<int,1>&,
        const Array_gpu<int,1>&, const Array_gpu<int,1>&, const Array_gpu<int,1>&, const Array_gpu<int,1>&, const Array_gpu<BOOL_TYPE,2>& tropo,
        const Array_gpu<double,4>&, const Array_gpu<double,6>&, const Array_gpu<double,5>&, const Array_gpu<double,2>&, const Array_gpu<double,2>&, const Array_gpu<double,3>&,
        const Array_gpu<int,4>&, const Array_gpu<int,2>&, const Array_gpu<int,2>&, Array_gpu<double,3>&, Tuner_map& tunings);

template void rrtmgp_kernel_launcher_cuda::Planck_source<double>(const int ncol, const int nlay, const int nbnd, const int ngpt,
        const int nflav, const int neta, const int npres, const int ntemp,
        const int nPlanckTemp, const Array_gpu<double,2>& tlay, const Array_gpu<double,2>& tlev,
        const Array_gpu<double,1>& tsfc, const int sfc_lay, const Array_gpu<double,6>& fmajor,
        const Array_gpu<int,4>& jeta, const Array_gpu<BOOL_TYPE,2>& tropo, const Array_gpu<int,2>& jtemp,
        const Array_gpu<int,2>& jpress, const Array_gpu<int,1>& gpoint_bands, const Array_gpu<int,2>& band_lims_gpt,
        const Array_gpu<double,4>& pfracin, const double temp_ref_min, const double totplnk_delta,
        const Array_gpu<double,2>& totplnk, const Array_gpu<int,2>& gpoint_flavor,
        Array_gpu<double,2>& sfc_src,  Array_gpu<double,3>& lay_src, Array_gpu<double,3>& lev_src_inc,
        Array_gpu<double,3>& lev_src_dec, Array_gpu<double,2>& sfc_src_jac, Tuner_map& tunings);
#endif