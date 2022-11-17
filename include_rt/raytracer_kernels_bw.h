#ifndef RAYTRACER_KERNELS_BW_H
#define RAYTRACER_KERNELS_BW_H
#include <curand_kernel.h>
#include "Types.h"
#include "raytracer_functions.h"

using Raytracer_functions::Vector;
using Raytracer_functions::Optics_scat;

#ifdef RTE_RRTMGP_SINGLE_PRECISION
constexpr int bw_kernel_block= 512;
constexpr int bw_kernel_grid = 1024;
#else
constexpr int bw_kernel_block = 256;
constexpr int bw_kernel_grid = 256;
#endif
constexpr Float k_null_gas_min = Float(1.e-3);

struct Grid_knull
{
    Float k_max;
    Float k_min;
};

struct Camera
{
    Vector<Float> position;

    // rotation matrix
    Vector<Float> mx;
    Vector<Float> my;
    Vector<Float> mz;

    Float f_zoom;
    void setup_rotation_matrix(const Float yaw_deg, const Float pitch_deg, const Float roll_deg)
    {
        const Float yaw = yaw_deg / Float(180.) * M_PI;
        const Float pitch = pitch_deg / Float(180.) * M_PI;
        const Float roll = roll_deg / Float(180.) * M_PI;
        mx = {cos(yaw)*sin(pitch), (cos(yaw)*cos(pitch)*sin(roll)-sin(yaw)*cos(roll)), (cos(yaw)*cos(pitch)*cos(roll)+sin(yaw)*sin(roll))};
        my = {sin(yaw)*sin(pitch), (sin(yaw)*cos(pitch)*sin(roll)+cos(yaw)*cos(roll)), (sin(yaw)*cos(pitch)*cos(roll)-cos(yaw)*sin(roll))};
        mz = {-cos(pitch), sin(pitch)*sin(roll), sin(pitch)*cos(roll)};
    }

    // size of output arrays, either number of horizontal and vertical pixels, or number of zenith/azimuth angles of fisheye lens
    int ny = 1024;
    int nx = 1024;
};

__global__
void ray_tracer_kernel_bw(
        const Int photons_to_shoot,
        const Grid_knull* __restrict__ k_null_grid,
        Float* __restrict__ camera_count,
        Float* __restrict__ camera_shot,
        int* __restrict__ counter,
        const Float* __restrict__ k_ext, const Optics_scat* __restrict__ scat_asy,
        const Float* __restrict__ k_ext_bg, const Optics_scat* __restrict__ scat_asy_bg,
        const Float* __restrict__ z_lev_bg,
        const Float* __restrict__ surface_albedo,
        const Float* __restrict__ land_use_map,
        const Float mu,
        const Vector<Float> grid_size,
        const Vector<Float> grid_d,
        const Vector<int> grid_cells,
        const Vector<int> kn_grid,
        const Vector<Float> sun_direction,
        const Camera camera,
        const int nbg);
#endif
