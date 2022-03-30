#ifndef RAYTRACER_BW_H
#define RAYTRACER_BW_H

#include <memory>
#include "Types.h"
#include <curand_kernel.h>
#include "raytracer_kernels_bw.h"
#include "Optical_props_rt.h"

// Forward declarations.
template<typename, int> class Array_gpu;
class Optical_props_rt;
class Optical_props_arry_rt;

#ifdef __CUDACC__
class Raytracer_bw
{
    public:
        Raytracer_bw();

        void trace_rays(
                const Int photons_to_shoot,
                const int n_col_x, const int n_col_y, const int n_z, const int n_lay,
                const Float dx_grid, const Float dy_grid, const Float dz_grid,
                const Array_gpu<Float,1>& z_lev,
                const Optical_props_2str_rt& optical_props,
                const Optical_props_2str_rt& cloud_optical_props,
                const Array_gpu<Float,2>& surface_albedo,
                const Float zenith_angle,
                const Float azimuth_angle,
                const Array_gpu<Float,1>& toa_src,
                const Float toa_factor,
                const Float rayleigh,
                const Array_gpu<Float,2>& col_dry,
                const Array_gpu<Float,2>& vmr_h2o,
                const Array_gpu<Float,1>& cam_data,
                Array_gpu<Float,2>& flux_camera);

        void add_xyz_camera(
                const int cam_nx, const int cam_ny,
                const Array_gpu<Float,1>& xyz_factor,
                const Array_gpu<Float,2>& flux_camera,
                Array_gpu<Float,3>& XYZ);
        
        void normalize_xyz_camera(
                const int cam_nx, const int cam_ny,
                const Float total_source,
                Array_gpu<Float,3>& XYZ);
                
    private:

};
#endif

#endif
