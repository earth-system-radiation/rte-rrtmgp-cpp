#ifndef RAYTRACER_KERNELS_H
#define RAYTRACER_KERNELS_H
#include "raytracer_functions.h"

using Raytracer_functions::Vector;
using Raytracer_functions::Optics_scat;

#ifdef RTE_RRTMGP_SINGLE_PRECISION
constexpr int block_size = 512;
constexpr int grid_size = 1024;
#else
constexpr int block_size = 512;
constexpr int grid_size = 256;
#endif

constexpr Float k_null_gas_min = Float(1.e-3);

__global__
void ray_tracer_kernel(
        const Int photons_to_shoot,
        const Int qrng_grid_x,
        const Int qrng_grid_y,
        const Int qrng_gpt_offset,
        const Float* __restrict__ k_null_grid,
        Float* __restrict__ toa_down_count,
        Float* __restrict__ tod_up_count,
        Float* __restrict__ surface_down_direct_count,
        Float* __restrict__ surface_down_diffuse_count,
        Float* __restrict__ surface_up_count,
        Float* __restrict__ atmos_direct_count,
        Float* __restrict__ atmos_diffuse_count,
        const Float* __restrict__ k_ext,
        const Optics_scat* __restrict__ scat_asy,
        const Float tod_inc_direct,
        const Float tod_inc_diffuse,
        const Float* __restrict__ surface_albedo,
        const Float x_size, const Float y_size, const Float z_size,
        const Float dx_grid, const Float dy_grid, const Float dz_grid,
        const int ngrid_x, const int ngrid_y, const int ngrid_z,
        const Float dir_x, const Float dir_y, const Float dir_z,
        const int itot, const int jtot, const int ktot,
        curandDirectionVectors32_t* qrng_vectors, unsigned int* qrng_constants);
#endif
