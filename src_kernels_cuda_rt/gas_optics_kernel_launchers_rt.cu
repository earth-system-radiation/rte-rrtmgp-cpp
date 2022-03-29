#include <chrono>
#include <functional>
#include <iostream>
#include <iomanip>

#include "rrtmgp_kernel_launcher_cuda_rt.h"
#include "tools_gpu.h"
#include "Array.h"
#include "tuner.h"


namespace
{
    #include "gas_optics_kernels_rt.cu"
}


namespace rrtmgp_kernel_launcher_cuda_rt
{
    void reorder123x321(
            const int ni, const int nj, const int nk,
            const Array_gpu<Float,3>& arr_in, Array_gpu<Float,3>& arr_out)
    {
        Tuner_map& tunings = Tuner::get_map();
        dim3 grid{ni, nj, nk}, block;

        if (tunings.count("reorder123x321_kernel") == 0)
        {
            std::tie(grid, block) = tune_kernel(
                "reorder123x321_kernel",
                dim3{ni, nj, nk},
                {1, 2, 4, 8, 16, 24, 32, 48, 64, 96},
                {1, 2, 4, 8, 16, 24, 32, 48, 64, 96},
                {1, 2, 4, 8, 16, 24, 32, 48, 64, 96},
                reorder123x321_kernel,
                ni, nj, nk, arr_in.ptr(), arr_out.ptr());

            tunings["reorder123x321_kernel"].first = grid;
            tunings["reorder123x321_kernel"].second = block;
        }
        else
        {
            grid = tunings["reorder123x321_kernel"].first;
            block = tunings["reorder123x321_kernel"].second;
        }

        reorder123x321_kernel<<<grid, block>>>(
                ni, nj, nk, arr_in.ptr(), arr_out.ptr());
    }

    
    void reorder12x21(const int ni, const int nj,
                      const Array_gpu<Float,2>& arr_in, Array_gpu<Float,2>& arr_out)
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

    
    void zero_array(const int ni, const int nj, const int nk, const int nn, Array_gpu<Float,4>& arr)
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
                ni, nj, nk, nn, arr.ptr());
    }

    
    void zero_array(const int ni, const int nj, const int nk, Array_gpu<Float,3>& arr)
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
    
    
    void zero_array(const int ni, const int nj, Array_gpu<Float,2>& arr)
    {
        const int block_i = 32;
        const int block_j = 16;

        const int grid_i = ni/block_i + (ni%block_i > 0);
        const int grid_j = nj/block_j + (nj%block_j > 0);

        dim3 grid_gpu(grid_i, grid_j, 1);
        dim3 block_gpu(block_i, block_j, 1);

        zero_array_kernel<<<grid_gpu, block_gpu>>>(
                ni, nj, arr.ptr());

    }

    void zero_array(const int ni, Array_gpu<int, 1>& arr)
    {
        const int block_i = 32;

        const int grid_i = ni/block_i + (ni%block_i > 0);

        dim3 grid_gpu(grid_i);
        dim3 block_gpu(block_i);

        zero_array_kernel<<<grid_gpu, block_gpu>>>(
                ni, arr.ptr());
    }
    
    void interpolation(
            const int ncol, const int nlay,
            const int ngas, const int nflav, const int neta, const int npres, const int ntemp,
            const Array_gpu<int,2>& flavor,
            const Array_gpu<Float,1>& press_ref_log,
            const Array_gpu<Float,1>& temp_ref,
            Float press_ref_log_delta,
            Float temp_ref_min,
            Float temp_ref_delta,
            Float press_ref_trop_log,
            const Array_gpu<Float,3>& vmr_ref,
            const Array_gpu<Float,2>& play,
            const Array_gpu<Float,2>& tlay,
            Array_gpu<Float,3>& col_gas,
            Array_gpu<int,2>& jtemp,
            Array_gpu<Float,6>& fmajor, Array_gpu<Float,5>& fminor,
            Array_gpu<Float,4>& col_mix,
            Array_gpu<Bool,2>& tropo,
            Array_gpu<int,4>& jeta,
            Array_gpu<int,2>& jpress)
    {
        Tuner_map& tunings = Tuner::get_map();
        Float tmin = std::numeric_limits<Float>::min();
        
        dim3 grid(ncol, nlay, nflav), block;
        if (tunings.count("interpolation_kernel") == 0)
        {
            std::tie(grid, block) = tune_kernel(
                    "interpolation_kernel",
                    dim3{ncol, nlay, nflav}, 
                    {1, 2, 4, 8, 16, 32, 64, 128, 256}, {1}, {1, 2, 4, 8, 16, 32, 64, 128, 256},
                    interpolation_kernel,
                    ncol, nlay, ngas, nflav, neta, npres, ntemp, tmin,
                    flavor.ptr(), press_ref_log.ptr(), temp_ref.ptr(),
                    press_ref_log_delta, temp_ref_min,
                    temp_ref_delta, press_ref_trop_log,
                    vmr_ref.ptr(), play.ptr(), tlay.ptr(),
                    col_gas.ptr(), jtemp.ptr(), fmajor.ptr(),
                    fminor.ptr(), col_mix.ptr(), tropo.ptr(),
                    jeta.ptr(), jpress.ptr());
            tunings["interpolation_kernel"].first = grid;
            tunings["interpolation_kernel"].second = block;
        }
        else
        {
            grid = tunings["interpolation_kernel"].first;
            block = tunings["interpolation_kernel"].second;
        }

        interpolation_kernel<<<grid, block>>>(
                ncol, nlay, ngas, nflav, neta, npres, ntemp, tmin,
                flavor.ptr(), press_ref_log.ptr(), temp_ref.ptr(),
                press_ref_log_delta, temp_ref_min,
                temp_ref_delta, press_ref_trop_log,
                vmr_ref.ptr(), play.ptr(), tlay.ptr(),
                col_gas.ptr(), jtemp.ptr(), fmajor.ptr(),
                fminor.ptr(), col_mix.ptr(), tropo.ptr(),
                jeta.ptr(), jpress.ptr());

    }

    
    void minor_scalings(
            const int ncol, const int nlay, const int nflav, const int ngpt,
            const int nminorlower, const int nminorupper,
            const int idx_h2o,
            const Array_gpu<int,2>& gpoint_flavor,
            const Array_gpu<int,2>& minor_limits_gpt_lower,
            const Array_gpu<int,2>& minor_limits_gpt_upper,
            const Array_gpu<Bool,1>& minor_scales_with_density_lower,
            const Array_gpu<Bool,1>& minor_scales_with_density_upper,
            const Array_gpu<Bool,1>& scale_by_complement_lower,
            const Array_gpu<Bool,1>& scale_by_complement_upper,
            const Array_gpu<int,1>& idx_minor_lower,
            const Array_gpu<int,1>& idx_minor_upper,
            const Array_gpu<int,1>& idx_minor_scaling_lower,
            const Array_gpu<int,1>& idx_minor_scaling_upper,
            const Array_gpu<Float,2>& play,
            const Array_gpu<Float,2>& tlay,
            const Array_gpu<Float,3>& col_gas,
            const Array_gpu<Bool,2>& tropo,
            Array_gpu<Float,3>& scalings_lower,
            Array_gpu<Float,3>& scalings_upper)
    {
        Tuner_map& tunings = Tuner::get_map();
        
        // lower atmosphere
        int idx_tropo=1;

        dim3 grid_1(ncol, nlay, nminorlower), block_1;
        if (tunings.count("minor_scalings_lower_kernel") == 0)
        {
            std::tie(grid_1, block_1) = tune_kernel(
                    "minor_scalings_lower_kernel",
                    dim3{ncol, nlay, nminorlower}, 
                    {8, 16, 24, 32, 48, 64, 128}, {1}, {1, 2, 4, 8, 16, 32},
                    scaling_kernel,
                    ncol, nlay, nflav, nminorlower,
                    idx_h2o, idx_tropo,
                    gpoint_flavor.ptr(),
                    minor_limits_gpt_lower.ptr(),
                    minor_scales_with_density_lower.ptr(),
                    scale_by_complement_lower.ptr(),
                    idx_minor_lower.ptr(),
                    idx_minor_scaling_lower.ptr(),
                    play.ptr(), tlay.ptr(), col_gas.ptr(),
                    tropo.ptr(), scalings_lower.ptr());

            tunings["minor_scalings_lower_kernel"].first = grid_1;
            tunings["minor_scalings_lower_kernel"].second = block_1;
        }
        else
        {
            grid_1 =  tunings["minor_scalings_lower_kernel"].first;
            block_1 = tunings["minor_scalings_lower_kernel"].second;
        }

        scaling_kernel<<<grid_1, block_1>>>(
                ncol, nlay, nflav, nminorlower,
                idx_h2o, idx_tropo,
                gpoint_flavor.ptr(),
                minor_limits_gpt_lower.ptr(),
                minor_scales_with_density_lower.ptr(),
                scale_by_complement_lower.ptr(),
                idx_minor_lower.ptr(),
                idx_minor_scaling_lower.ptr(),
                play.ptr(), tlay.ptr(), col_gas.ptr(),
                tropo.ptr(), scalings_lower.ptr());

        // upper atmosphere
        idx_tropo=0;

        dim3 grid_2(ncol, nlay, nminorupper), block_2;
        if (tunings.count("minor_scalings_upper_kernel") == 0)
        {
            std::tie(grid_2, block_2) = tune_kernel(
                    "minor_scalings_upper_kernel",
                    dim3{ncol, nlay, nminorupper}, 
                    {8, 16, 24, 32, 48, 64, 128}, {1}, {1, 2, 4, 8, 16, 32},
                    scaling_kernel,
                    ncol, nlay, nflav, nminorupper,
                    idx_h2o, idx_tropo,
                    gpoint_flavor.ptr(),
                    minor_limits_gpt_upper.ptr(),
                    minor_scales_with_density_upper.ptr(),
                    scale_by_complement_upper.ptr(),
                    idx_minor_upper.ptr(),
                    idx_minor_scaling_upper.ptr(),
                    play.ptr(), tlay.ptr(), col_gas.ptr(),
                    tropo.ptr(), scalings_upper.ptr());

            tunings["minor_scalings_upper_kernel"].first = grid_2;
            tunings["minor_scalings_upper_kernel"].second = block_2;
        }
        else
        {
            grid_2 =  tunings["minor_scalings_upper_kernel"].first;
            block_2 = tunings["minor_scalings_upper_kernel"].second;
        }

        scaling_kernel<<<grid_2, block_2>>>(
                ncol, nlay, nflav, nminorupper,
                idx_h2o, idx_tropo,
                gpoint_flavor.ptr(),
                minor_limits_gpt_upper.ptr(),
                minor_scales_with_density_upper.ptr(),
                scale_by_complement_upper.ptr(),
                idx_minor_upper.ptr(),
                idx_minor_scaling_upper.ptr(),
                play.ptr(), tlay.ptr(), col_gas.ptr(),
                tropo.ptr(), scalings_upper.ptr());
    }

    
    void combine_abs_and_rayleigh(
            const int ncol, const int nlay,
            const Array_gpu<Float,2>& tau_abs, const Array_gpu<Float,2>& tau_rayleigh,
            Array_gpu<Float,2>& tau, Array_gpu<Float,2>& ssa, Array_gpu<Float,2>& g)
    {
        Tuner_map& tunings = Tuner::get_map();
        
        Float tmin = std::numeric_limits<Float>::epsilon();

        dim3 grid{ncol, nlay, 1}, block;
        if (tunings.count("combine_abs_and_rayleigh_kernel") == 0)
        {
            std::tie(grid, block) = tune_kernel(
                "combine_abs_and_rayleigh_kernel",
                dim3{ncol, nlay, 1},
                {24, 32, 48, 64, 96, 128, 256, 512}, {1, 2, 4}, {1},
                combine_abs_and_rayleigh_kernel,
                ncol, nlay, tmin,
                tau_abs.ptr(), tau_rayleigh.ptr(),
                tau.ptr(), ssa.ptr(), g.ptr());

            tunings["combine_abs_and_rayleigh_kernel"].first = grid;
            tunings["combine_abs_and_rayleigh_kernel"].second = block;
        }
        else
        {
            grid = tunings["combine_abs_and_rayleigh_kernel"].first;
            block = tunings["combine_abs_and_rayleigh_kernel"].second;
        }

        combine_abs_and_rayleigh_kernel<<<grid, block>>>(
                ncol, nlay, tmin,
                tau_abs.ptr(), tau_rayleigh.ptr(),
                tau.ptr(), ssa.ptr(), g.ptr());
    }

    
    void compute_tau_rayleigh(
            const int ncol, const int nlay, const int nbnd, const int ngpt, const int igpt,
            const int ngas, const int nflav, const int neta, const int npres, const int ntemp,
            const Array_gpu<int,2>& gpoint_flavor,
            const Array_gpu<int,1>& gpoint_bands,
            const Array_gpu<int,2>& band_lims_gpt,
            const Array_gpu<Float,4>& krayl,
            int idx_h2o, const Array_gpu<Float,2>& col_dry, const Array_gpu<Float,3>& col_gas,
            const Array_gpu<Float,5>& fminor, const Array_gpu<int,4>& jeta,
            const Array_gpu<Bool,2>& tropo, const Array_gpu<int,2>& jtemp,
            Array_gpu<Float,2>& tau_rayleigh)
    {
        Tuner_map& tunings = Tuner::get_map();

        dim3 grid{ncol, nlay, 1}, block;
        if (tunings.count("compute_tau_rayleigh_kernel") == 0)
        {
            std::tie(grid, block) = tune_kernel(
                "compute_tau_rayleigh_kernel",
                dim3{ncol, nlay, 1},
                {24, 32, 64, 128, 256, 512}, {1, 2}, {1},
                compute_tau_rayleigh_kernel,
                ncol, nlay, nbnd, ngpt,
                ngas, nflav, neta, npres, ntemp,
                igpt,
                gpoint_flavor.ptr(),
                gpoint_bands.ptr(),
                band_lims_gpt.ptr(),
                krayl.ptr(),
                idx_h2o, col_dry.ptr(), col_gas.ptr(),
                fminor.ptr(), jeta.ptr(),
                tropo.ptr(), jtemp.ptr(),
                tau_rayleigh.ptr());

            tunings["compute_tau_rayleigh_kernel"].first = grid;
            tunings["compute_tau_rayleigh_kernel"].second = block;
        }
        else
        {
            grid = tunings["compute_tau_rayleigh_kernel"].first;
            block = tunings["compute_tau_rayleigh_kernel"].second;
        }
        
        compute_tau_rayleigh_kernel<<<grid, block>>>(
                ncol, nlay, nbnd, ngpt,
                ngas, nflav, neta, npres, ntemp,
                igpt,
                gpoint_flavor.ptr(),
                gpoint_bands.ptr(),
                band_lims_gpt.ptr(),
                krayl.ptr(),
                idx_h2o, col_dry.ptr(), col_gas.ptr(),
                fminor.ptr(), jeta.ptr(),
                tropo.ptr(), jtemp.ptr(),
                tau_rayleigh.ptr());
    }

    
    void compute_tau_absorption(
            const int ncol, const int nlay, const int nband, const int ngpt, const int igpt,
            const int ngas, const int nflav, const int neta, const int npres, const int ntemp,
            const int nminorlower, const int nminorklower,
            const int nminorupper, const int nminorkupper,
            const int idx_h2o,
            const Array_gpu<int,2>& gpoint_flavor,
            const Array_gpu<int,2>& band_lims_gpt,
            const Array_gpu<Float,4>& kmajor,
            const Array_gpu<Float,3>& kminor_lower,
            const Array_gpu<Float,3>& kminor_upper,
            const Array_gpu<int,2>& minor_limits_gpt_lower,
            const Array_gpu<int,2>& minor_limits_gpt_upper,
            const Array_gpu<int,2>& first_last_minor_lower,
            const Array_gpu<int,2>& first_last_minor_upper,
            const Array_gpu<Bool,1>& minor_scales_with_density_lower,
            const Array_gpu<Bool,1>& minor_scales_with_density_upper,
            const Array_gpu<Bool,1>& scale_by_complement_lower,
            const Array_gpu<Bool,1>& scale_by_complement_upper,
            const Array_gpu<int,1>& idx_minor_lower,
            const Array_gpu<int,1>& idx_minor_upper,
            const Array_gpu<int,1>& idx_minor_scaling_lower,
            const Array_gpu<int,1>& idx_minor_scaling_upper,
            const Array_gpu<int,1>& kminor_start_lower,
            const Array_gpu<int,1>& kminor_start_upper,
            const Array_gpu<Bool,2>& tropo,
            const Array_gpu<Float,4>& col_mix, const Array_gpu<Float,6>& fmajor,
            const Array_gpu<Float,5>& fminor, const Array_gpu<Float,2>& play,
            const Array_gpu<Float,2>& tlay, const Array_gpu<Float,3>& col_gas,
            const Array_gpu<int,4>& jeta, const Array_gpu<int,2>& jtemp,
            const Array_gpu<int,2>& jpress,
            const Array_gpu<Float,3>& scalings_lower,
            const Array_gpu<Float,3>& scalings_upper,
            Array_gpu<Float,2>& tau)
    {
        Tuner_map& tunings = Tuner::get_map();
        
        dim3 grid_maj{nlay, ncol, 1}, block_maj;
        if (tunings.count("gas_optical_depths_major_kernel") == 0)
        {
            std::tie(grid_maj, block_maj) = tune_kernel(
                    "gas_optical_depths_major_kernel",
                    dim3{nlay, ncol, 1}, {1, 2}, {64, 96, 128, 256, 512, 768, 1024}, {1, 2},
                    gas_optical_depths_major_kernel,
                    ncol, nlay, nband, ngpt,
                    nflav, neta, npres, ntemp,
                    igpt,
                    gpoint_flavor.ptr(), band_lims_gpt.ptr(),
                    kmajor.ptr(), col_mix.ptr(), fmajor.ptr(), jeta.ptr(),
                    tropo.ptr(), jtemp.ptr(), jpress.ptr(),
                    Array_gpu<Float,2>(tau).ptr());

            tunings["gas_optical_depths_major_kernel"].first = grid_maj;
            tunings["gas_optical_depths_major_kernel"].second = block_maj;
        }
        else
        {
            grid_maj = tunings["gas_optical_depths_major_kernel"].first;
            block_maj = tunings["gas_optical_depths_major_kernel"].second;
        }
        
        gas_optical_depths_major_kernel<<<grid_maj, block_maj>>>(
            ncol, nlay, nband, ngpt,
            nflav, neta, npres, ntemp,
            igpt,
            gpoint_flavor.ptr(), band_lims_gpt.ptr(),
            kmajor.ptr(), col_mix.ptr(), fmajor.ptr(), jeta.ptr(),
            tropo.ptr(), jtemp.ptr(), jpress.ptr(),
            tau.ptr());
        

        const int nscale_lower = scale_by_complement_lower.dim(1);
        const int nscale_upper = scale_by_complement_upper.dim(1);

        // Lower
        int idx_tropo = 1;

        dim3 grid_min_1(nlay, ncol, 1), block_min_1;
        if (tunings.count("gas_optical_depths_minor_kernel_lower") == 0)
        {
            std::tie(grid_min_1, block_min_1) = tune_kernel(
                        "gas_optical_depths_minor_kernel_lower",
                        dim3{nlay, ncol, 1},
                        {1}, {32, 48, 64, 96, 128, 256, 384, 512}, {1},
                        gas_optical_depths_minor_kernel,
                        ncol, nlay, ngpt, igpt,
                        ngas, nflav, ntemp, neta,
                        nscale_lower,
                        nminorlower,
                        nminorklower,
                        idx_h2o, idx_tropo,
                        gpoint_flavor.ptr(),
                        kminor_lower.ptr(),
                        minor_limits_gpt_lower.ptr(),
                        first_last_minor_lower.ptr(),
                        minor_scales_with_density_lower.ptr(),
                        scale_by_complement_lower.ptr(),
                        idx_minor_lower.ptr(),
                        idx_minor_scaling_lower.ptr(),
                        kminor_start_lower.ptr(),
                        play.ptr(), tlay.ptr(), col_gas.ptr(),
                        fminor.ptr(), jeta.ptr(), jtemp.ptr(),
                        tropo.ptr(), scalings_lower.ptr(),
                        Array_gpu<Float,2>(tau).ptr());

            tunings["gas_optical_depths_minor_kernel_lower"].first = grid_min_1;
            tunings["gas_optical_depths_minor_kernel_lower"].second = block_min_1;
        }
        else
        {
            grid_min_1 = tunings["gas_optical_depths_minor_kernel_lower"].first;
            block_min_1 = tunings["gas_optical_depths_minor_kernel_lower"].second;
        }
        
        gas_optical_depths_minor_kernel<<<grid_min_1, block_min_1>>>(
                ncol, nlay, ngpt, igpt,
                ngas, nflav, ntemp, neta,
                nscale_lower,
                nminorlower,
                nminorklower,
                idx_h2o, idx_tropo,
                gpoint_flavor.ptr(),
                kminor_lower.ptr(),
                minor_limits_gpt_lower.ptr(),
                first_last_minor_lower.ptr(),
                minor_scales_with_density_lower.ptr(),
                scale_by_complement_lower.ptr(),
                idx_minor_lower.ptr(),
                idx_minor_scaling_lower.ptr(),
                kminor_start_lower.ptr(),
                play.ptr(), tlay.ptr(), col_gas.ptr(),
                fminor.ptr(), jeta.ptr(), jtemp.ptr(),
                tropo.ptr(), scalings_lower.ptr(), tau.ptr());

        // Upper
        idx_tropo = 0;

        dim3 grid_min_2(nlay, ncol, 1), block_min_2;
        if (tunings.count("gas_optical_depths_minor_kernel_upper") == 0)
        {
            std::tie(grid_min_2, block_min_2) = tune_kernel(
                   "gas_optical_depths_minor_kernel_upper",
                   dim3{nlay, ncol, 1},
                   {1}, {32, 48, 64, 96, 128, 256, 384, 512}, {1},
                   gas_optical_depths_minor_kernel,
                   ncol, nlay, ngpt, igpt,
                   ngas, nflav, ntemp, neta,
                   nscale_upper,
                   nminorupper,
                   nminorkupper,
                   idx_h2o, idx_tropo,
                   gpoint_flavor.ptr(),
                   kminor_upper.ptr(),
                   minor_limits_gpt_upper.ptr(),
                   first_last_minor_upper.ptr(),
                   minor_scales_with_density_upper.ptr(),
                   scale_by_complement_upper.ptr(),
                   idx_minor_upper.ptr(),
                   idx_minor_scaling_upper.ptr(),
                   kminor_start_upper.ptr(),
                   play.ptr(), tlay.ptr(), col_gas.ptr(),
                   fminor.ptr(), jeta.ptr(), jtemp.ptr(),
                   tropo.ptr(), scalings_upper.ptr(), Array_gpu<Float,2>(tau).ptr());

            tunings["gas_optical_depths_minor_kernel_upper"].first = grid_min_2;
            tunings["gas_optical_depths_minor_kernel_upper"].second = block_min_2;
        }
        else
        {
            grid_min_2 = tunings["gas_optical_depths_minor_kernel_upper"].first;
            block_min_2 = tunings["gas_optical_depths_minor_kernel_upper"].second;
        }

        gas_optical_depths_minor_kernel<<<grid_min_2, block_min_2>>>(
                ncol, nlay, ngpt, igpt,
                ngas, nflav, ntemp, neta,
                nscale_upper,
                nminorupper,
                nminorkupper,
                idx_h2o, idx_tropo,
                gpoint_flavor.ptr(),
                kminor_upper.ptr(),
                minor_limits_gpt_upper.ptr(),
                first_last_minor_upper.ptr(),
                minor_scales_with_density_upper.ptr(),
                scale_by_complement_upper.ptr(),
                idx_minor_upper.ptr(),
                idx_minor_scaling_upper.ptr(),
                kminor_start_upper.ptr(),
                play.ptr(), tlay.ptr(), col_gas.ptr(),
                fminor.ptr(), jeta.ptr(), jtemp.ptr(),
                tropo.ptr(), scalings_upper.ptr(), tau.ptr());
        
    }


    
    void Planck_source(
            const int ncol, const int nlay, const int nbnd, const int ngpt, const int igpt,
            const int nflav, const int neta, const int npres, const int ntemp,
            const int nPlanckTemp,
            const Array_gpu<Float,2>& tlay,
            const Array_gpu<Float,2>& tlev,
            const Array_gpu<Float,1>& tsfc,
            const int sfc_lay,
            const Array_gpu<Float,6>& fmajor,
            const Array_gpu<int,4>& jeta,
            const Array_gpu<Bool,2>& tropo,
            const Array_gpu<int,2>& jtemp,
            const Array_gpu<int,2>& jpress,
            const Array_gpu<int,1>& gpoint_bands,
            const Array_gpu<int,2>& band_lims_gpt,
            const Array_gpu<Float,4>& pfracin,
            const Float temp_ref_min, const Float totplnk_delta,
            const Array_gpu<Float,2>& totplnk,
            const Array_gpu<int,2>& gpoint_flavor,
            Array_gpu<Float,1>& sfc_src,
            Array_gpu<Float,2>& lay_src,
            Array_gpu<Float,2>& lev_src_inc,
            Array_gpu<Float,2>& lev_src_dec,
            Array_gpu<Float,1>& sfc_src_jac)
    {
        Tuner_map& tunings = Tuner::get_map();

        const Float delta_Tsurf = Float(1.);
        
        dim3 grid(ncol, nlay, 1), block;
        if (tunings.count("Planck_source_kernel") == 0)
        {
            std::tie(grid, block) = tune_kernel(
                    "Planck_source_kernel",
                    dim3{ncol, nlay, 1},
                    {16, 32, 48, 64, 96, 128, 256, 512}, {1, 2, 4, 8}, {1},
                    Planck_source_kernel,
                    ncol, nlay, nbnd, ngpt,
                    nflav, neta, npres, ntemp, nPlanckTemp, igpt,
                    tlay.ptr(), tlev.ptr(), tsfc.ptr(), sfc_lay,
                    fmajor.ptr(), jeta.ptr(), tropo.ptr(), jtemp.ptr(),
                    jpress.ptr(), gpoint_bands.ptr(), band_lims_gpt.ptr(),
                    pfracin.ptr(), temp_ref_min, totplnk_delta,
                    totplnk.ptr(), gpoint_flavor.ptr(),
                    delta_Tsurf, sfc_src.ptr(), lay_src.ptr(),
                    lev_src_inc.ptr(), lev_src_dec.ptr(),
                    sfc_src_jac.ptr());

            tunings["Planck_source_kernel"].first = grid;
            tunings["Planck_source_kernel"].second = block;
        }
        else
        {
            grid = tunings["Planck_source_kernel"].first;
            block = tunings["Planck_source_kernel"].second;
        }

        Planck_source_kernel<<<grid, block>>>(
                ncol, nlay, nbnd, ngpt,
                nflav, neta, npres, ntemp, nPlanckTemp, igpt,
                tlay.ptr(), tlev.ptr(), tsfc.ptr(), sfc_lay,
                fmajor.ptr(), jeta.ptr(), tropo.ptr(), jtemp.ptr(),
                jpress.ptr(), gpoint_bands.ptr(), band_lims_gpt.ptr(),
                pfracin.ptr(), temp_ref_min, totplnk_delta,
                totplnk.ptr(), gpoint_flavor.ptr(),
                delta_Tsurf,
                sfc_src.ptr(), lay_src.ptr(),
                lev_src_inc.ptr(), lev_src_dec.ptr(),
                sfc_src_jac.ptr());
    }
}

