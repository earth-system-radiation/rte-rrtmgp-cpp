#ifndef RAYTRACER_KERNELS_BW_H
#define RAYTRACER_KERNELS_BW_H
#include <curand_kernel.h>
#include "Types.h"



#ifdef RTE_RRTMGP_SINGLE_PRECISION
//using Float = float;
constexpr int block_size= 512;
constexpr int grid_size = 1024;
#else
//using Float = double;
constexpr int block_size = 256;
constexpr int grid_size = 256;
#endif
using Int = unsigned long long;
constexpr int ngrid_x = 96;
constexpr int ngrid_y = 96;
constexpr int ngrid_z = 24;
constexpr Float k_null_gas_min = Float(1.e-3);

struct Optics_scat
{
    Float k_sca_gas;
    Float k_sca_cld;
    Float k_sca_aer;
    Float asy_cld;
    Float asy_aer;
};

struct Grid_knull
{
    Float k_max;
    Float k_min;
};

__global__
void ray_tracer_kernel_bw(
        const Int photons_to_shoot,
        const Grid_knull* __restrict__ k_null_grid,
        Float* __restrict__ camera_count,
        Float* __restrict__ camera_shot,
        int* __restrict__ counter,
        const int cam_nx, const int cam_ny, const Float* __restrict__ cam_data,
        const Float* __restrict__ k_ext, const Optics_scat* __restrict__ scat_asy,
        const Float* __restrict__ k_ext_bg, const Optics_scat* __restrict__ scat_asy_bg,
        const Float* __restrict__ z_lev_bg,
        const Float* __restrict__ surface_albedo,
        const Float* __restrict__ land_use_map,
        const Float mu,
        const Float x_size, const Float y_size, const Float z_size,
        const Float dx_grid, const Float dy_grid, const Float dz_grid,
        const int ngrid_x, const int ngrid_y, const int ngrid_z,
        const Float dir_x, const Float dir_y, const Float dir_z,
        const int itot, const int jtot, const int ktot, const int nbg);
#endif
