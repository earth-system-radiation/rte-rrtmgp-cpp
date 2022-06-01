#ifndef RAYTRACER_KERNELS_H
#define RAYTRACER_KERNELS_H
#include <curand_kernel.h>
#include "Types.h"



#ifdef RTE_RRTMGP_SINGLE_PRECISION
constexpr int block_size = 256;
constexpr int grid_size = 512;
#else
constexpr int block_size = 512;
constexpr int grid_size = 256;
#endif
using Int = unsigned long long;
constexpr int ngrid_x = 20;//90;
constexpr int ngrid_y = 20;//90;
constexpr int ngrid_z = 35;//71
constexpr Float k_null_gas_min = Float(1.e-3);


struct Optics_ext
{
    Float gas;
    Float cloud;
};


struct Optics_scat
{
    Float ssa;
    Float asy;
};

__global__
void ray_tracer_kernel(
        const Int photons_to_shoot,
        const Float* __restrict__ k_null_grid,
        Float* __restrict__ toa_down_count,
        Float* __restrict__ tod_up_count,
        Float* __restrict__ surface_down_direct_count,
        Float* __restrict__ surface_down_diffuse_count,
        Float* __restrict__ surface_up_count,
        Float* __restrict__ atmos_direct_count,
        Float* __restrict__ atmos_diffuse_count,
        const Optics_ext* __restrict__ k_ext, const Optics_scat* __restrict__ ssa_asy,
        const Float tod_inc_direct,
        const Float tod_inc_diffuse,
        const Float* __restrict__ surface_albedo,
        const Float x_size, const Float y_size, const Float z_size,
        const Float dx_grid, const Float dy_grid, const Float dz_grid,
        const Float dir_x, const Float dir_y, const Float dir_z,
        const int itot, const int jtot, const int ktot,
        curandDirectionVectors32_t* qrng_vectors, unsigned int* qrng_constants);
#endif
