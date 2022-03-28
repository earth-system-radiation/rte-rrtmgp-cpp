#ifndef RAYTRACER_RT_H
#define RAYTRACER_RT_H

#include <memory>
#include "Types.h"
#include <curand_kernel.h>
#include "raytracer_kernels.h"
#include "Optical_props_rt.h"

// Forward declarations.
template<typename, int> class Array_gpu;
class Optical_props_rt;
class Optical_props_arry_rt;

#ifdef __CUDACC__
class Raytracer
{
    public:
        Raytracer();

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
                Array_gpu<Float,2>& flux_tod_up,
                Array_gpu<Float,2>& flux_sfc_dir,
                Array_gpu<Float,2>& flux_sfc_dif,
                Array_gpu<Float,2>& flux_sfc_up,
                Array_gpu<Float,3>& flux_abs_dir,
                Array_gpu<Float,3>& flux_abs_dif);

    private:
        curandDirectionVectors32_t* qrng_vectors_gpu;
        unsigned int* qrng_constants_gpu;

};
#endif

#endif
