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
                const int nlay,
                const Vector<int>& grid_cells,
                const Vector<Float>& grid_d,
                const Vector<int>& kn_grid,
                const Array_gpu<Float,1>& z_lev,
                const Array_gpu<Float,2>& tau_total,
                const Array_gpu<Float,2>& ssa_total,
                const Array_gpu<Float,2>& tau_cloud,
                const Array_gpu<Float,2>& ssa_cloud,
                const Array_gpu<Float,2>& asy_cloud,
                const Array_gpu<Float,2>& tau_aeros,
                const Array_gpu<Float,2>& ssa_aeros,
                const Array_gpu<Float,2>& asy_aeros,
                const Array_gpu<Float,2>& surface_albedo,
                const Array_gpu<Float,1>& land_use_map,
                const Float zenith_angle,
                const Float azimuth_angle,
                const Float toa_src,
                const Float toa_factor,
                const Float rayleigh,
                const Array_gpu<Float,2>& col_dry,
                const Array_gpu<Float,2>& vmr_h2o,
                const Camera& camera,
                Array_gpu<Float,2>& flux_camera);

        void trace_rays_bb(
                const Int photons_to_shoot,
                const int nlay,
                const Vector<int>& grid_cells,
                const Vector<Float>& grid_d,
                const Vector<int>& kn_grid,
                const Array_gpu<Float,1>& z_lev,
                const Array_gpu<Float,2>& tau_total,
                const Array_gpu<Float,2>& ssa_total,
                const Array_gpu<Float,2>& tau_cloud,
                const Array_gpu<Float,2>& ssa_cloud,
                const Array_gpu<Float,2>& asy_cloud,
                const Array_gpu<Float,2>& tau_aeros,
                const Array_gpu<Float,2>& ssa_aeros,
                const Array_gpu<Float,2>& asy_aeros,
                const Array_gpu<Float,2>& surface_albedo,
                const Array_gpu<Float,1>& land_use_map,
                const Float zenith_angle,
                const Float azimuth_angle,
                const Float toa_src,
                const Camera& camera,
                Array_gpu<Float,2>& flux_camera);

        void add_camera(
                const Camera& camera,
                const Array_gpu<Float,2>& flux_camera,
                Array_gpu<Float,2>& radiance);

        void add_xyz_camera(
                const Camera& camera,
                const Array_gpu<Float,1>& xyz_factor,
                const Array_gpu<Float,2>& flux_camera,
                Array_gpu<Float,3>& XYZ);

        void normalize_xyz_camera(
                const Camera& camera,
                const Float total_source,
                Array_gpu<Float,3>& XYZ);

    private:

};
#endif

#endif
