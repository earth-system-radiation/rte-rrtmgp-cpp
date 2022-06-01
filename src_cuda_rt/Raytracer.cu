#include "Raytracer.h"
#include "Array.h"
#include <curand_kernel.h>
#include "rrtmgp_kernel_launcher_cuda_rt.h"
#include "raytracer_kernels.h"
#include "Optical_props_rt.h"
namespace
{
    inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort=true)
    {
        if (code != cudaSuccess)
        {
            fprintf(stderr,"CUDA_SAFE_CALL: %s %s %d\n", cudaGetErrorString(code), file, line);
            if (abort) exit(code);
        }
    }


    template<typename T>
    T* allocate_gpu(const int length)
    {
        T* data_ptr = Tools_gpu::allocate_gpu<T>(length);

        return data_ptr;
    }

    template<typename T>
    void copy_to_gpu(T* gpu_data, const T* cpu_data, const int length)
    {
        cuda_safe_call(cudaMemcpy(gpu_data, cpu_data, length*sizeof(T), cudaMemcpyHostToDevice));
    }



    template<typename T>
    void copy_from_gpu(T* cpu_data, const T* gpu_data, const int length)
    {
        cuda_safe_call(cudaMemcpy(cpu_data, gpu_data, length*sizeof(T), cudaMemcpyDeviceToHost));
    }

    __global__
    void create_knull_grid(
            const int ncol_x, const int ncol_y, const int nlay, const Float k_ext_null_min,
            const Optics_ext* __restrict__ k_ext, Float* __restrict__ k_null_grid)
    {
        const int grid_x = blockIdx.x*blockDim.x + threadIdx.x;
        const int grid_y = blockIdx.y*blockDim.y + threadIdx.y;
        const int grid_z = blockIdx.z*blockDim.z + threadIdx.z;
        if ( ( grid_x < ngrid_h) && ( grid_y < ngrid_h) && ( grid_z < ngrid_v))
        {
            const Float fx = Float(ncol_x) / Float(ngrid_h);
            const Float fy = Float(ncol_y) / Float(ngrid_h);
            const Float fz = Float(nlay) / Float(ngrid_v);

            const int x0 = grid_x*fx;
            const int x1 = floor((grid_x+1)*fx);
            const int y0 = grid_y*fy;
            const int y1 = floor((grid_y+1)*fy);
            const int z0 = grid_z*fz;
            const int z1 = floor((grid_z+1)*fz);

            const int ijk_grid = grid_x +grid_y*ngrid_h + grid_z*ngrid_h*ngrid_h;
            Float k_null = k_ext_null_min;

            for (int k=z0; k<z1; ++k)
                for (int j=y0; j<y1; ++j)
                    for (int i=x0; i<x1; ++i)
                    {
                        const int ijk_in = i + j*ncol_x + k*ncol_x*ncol_y;
                        const Float k_ext_tot = k_ext[ijk_in].gas + k_ext[ijk_in].cloud;
                        k_null = max(k_null, k_ext_tot);
                    }
            k_null_grid[ijk_grid] = k_null;
        }
    }


    __global__
    void bundles_optical_props(
            const int ncol_x, const int ncol_y, const int nlay, const Float dz_grid,
            const Float* __restrict__ tau_tot, const Float* __restrict__ ssa,
            const Float* __restrict__ asy, const Float* __restrict__ tau_cld,
            Optics_ext* __restrict__ k_ext, Optics_scat* __restrict__ ssa_asy)
    {
        const int icol_x = blockIdx.x*blockDim.x + threadIdx.x;
        const int icol_y = blockIdx.y*blockDim.y + threadIdx.y;
        const int iz = blockIdx.z*blockDim.z + threadIdx.z;
        if ( ( icol_x < ncol_x) && ( icol_y < ncol_y) && ( iz < nlay))
        {
            const int idx = icol_x + icol_y*ncol_x + iz*ncol_y*ncol_x;
            const Float kext_cld = tau_cld[idx] / dz_grid;
            const Float kext_gas = tau_tot[idx] / dz_grid - kext_cld;
            k_ext[idx].cloud = kext_cld;
            k_ext[idx].gas = kext_gas;
            ssa_asy[idx].ssa = ssa[idx];
            ssa_asy[idx].asy = asy[idx];
        }
    }

    __global__
    void count_to_flux_2d(
            const int ncol_x, const int ncol_y, const Float photons_per_col, const Float toa_src,
            const Float* __restrict__ count_1, const Float* __restrict__ count_2, const Float* __restrict__ count_3, const Float* __restrict__ count_4,
            Float* __restrict__ flux_1, Float* __restrict__ flux_2, Float* __restrict__ flux_3, Float* __restrict__ flux_4)
    {
        const int icol_x = blockIdx.x*blockDim.x + threadIdx.x;
        const int icol_y = blockIdx.y*blockDim.y + threadIdx.y;

        if ( ( icol_x < ncol_x) && ( icol_y < ncol_y) )
        {
            const int idx = icol_x + icol_y*ncol_x;
            const Float flux_per_ray = toa_src / photons_per_col;
            flux_1[idx] = count_1[idx] * flux_per_ray;
            flux_2[idx] = count_2[idx] * flux_per_ray;
            flux_3[idx] = count_3[idx] * flux_per_ray;
            flux_4[idx] = count_4[idx] * flux_per_ray;
        }
    }

    __global__
    void count_to_flux_3d(
            const int ncol_x, const int ncol_y, const int nlay, const Float photons_per_col, const Float toa_src,
            const Float* __restrict__ count_1, const Float* __restrict__ count_2,
            Float* __restrict__ flux_1, Float* __restrict__ flux_2)
    {
        const int icol_x = blockIdx.x*blockDim.x + threadIdx.x;
        const int icol_y = blockIdx.y*blockDim.y + threadIdx.y;
        const int iz = blockIdx.z*blockDim.z + threadIdx.z;

        if ( ( icol_x < ncol_x) && ( icol_y < ncol_y) && ( iz < nlay))
        {
            const int idx = icol_x + icol_y*ncol_x + iz*ncol_x*ncol_y;
            const Float flux_per_ray = toa_src / photons_per_col;
            flux_1[idx] = count_1[idx] * flux_per_ray;
            flux_2[idx] = count_2[idx] * flux_per_ray;
        }
    }
}


Raytracer::Raytracer()
{
    curandDirectionVectors32_t* qrng_vectors;
    curandGetDirectionVectors32(
                &qrng_vectors,
                CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6);
    unsigned int* qrng_constants;
    curandGetScrambleConstants32(&qrng_constants);

    this->qrng_vectors_gpu = allocate_gpu<curandDirectionVectors32_t>(2);
    this->qrng_constants_gpu = allocate_gpu<unsigned int>(2);

    copy_to_gpu(qrng_vectors_gpu, qrng_vectors, 2);
    copy_to_gpu(qrng_constants_gpu, qrng_constants, 2);
}


void Raytracer::trace_rays(
        const Int photons_to_shoot,
        const int ncol_x, const int ncol_y, const int nlay,
        const Float dx_grid, const Float dy_grid, const Float dz_grid,
        const Optical_props_2str_rt& optical_props,
        const Optical_props_2str_rt& cloud_optical_props,
        const Array_gpu<Float,2>& surface_albedo,
        const Float zenith_angle,
        const Float azimuth_angle,
        const Float tod_inc_direct,
        const Float tod_inc_diffuse,
        Array_gpu<Float,2>& flux_tod_up,
        Array_gpu<Float,2>& flux_sfc_dir,
        Array_gpu<Float,2>& flux_sfc_dif,
        Array_gpu<Float,2>& flux_sfc_up,
        Array_gpu<Float,3>& flux_abs_dir,
        Array_gpu<Float,3>& flux_abs_dif)
{
    // set of block and grid dimensions used in data processing kernels - requires some proper tuning later
    const int block_col_x = 8;
    const int block_col_y = 8;
    const int block_z = 4;

    const int grid_col_x  = ncol_x/block_col_x + (ncol_x%block_col_x > 0);
    const int grid_col_y  = ncol_y/block_col_y + (ncol_y%block_col_y > 0);
    const int grid_z  = nlay/block_z + (nlay%block_z > 0);

    dim3 grid_2d(grid_col_x, grid_col_y);
    dim3 block_2d(block_col_x, block_col_y);
    dim3 grid_3d(grid_col_x, grid_col_y, grid_z);
    dim3 block_3d(block_col_x, block_col_y, block_z);

    // bundle optical properties in struct
    Array_gpu<Optics_ext,3> k_ext({ncol_x, ncol_y, nlay});
    Array_gpu<Optics_scat,3> ssa_asy({ncol_x, ncol_y, nlay});

    bundles_optical_props<<<grid_3d, block_3d>>>(
            ncol_x, ncol_y, nlay, dz_grid,
            optical_props.get_tau().ptr(), optical_props.get_ssa().ptr(),
            optical_props.get_g().ptr(), cloud_optical_props.get_tau().ptr(),
            k_ext.ptr(), ssa_asy.ptr());

    // create k_null_grid
    const int block_kn_x = 8;
    const int block_kn_y = 8;
    const int block_kn_z = 4;

    const int grid_kn_x  = ngrid_h/block_kn_x + (ngrid_h%block_kn_x > 0);
    const int grid_kn_y  = ngrid_h/block_kn_y + (ngrid_h%block_kn_y > 0);
    const int grid_kn_z  = ngrid_v/block_kn_z + (ngrid_v%block_kn_z > 0);

    dim3 grid_kn(grid_kn_x, grid_kn_y, grid_kn_z);
    dim3 block_kn(block_kn_x, block_kn_y, block_kn_z);

    Array_gpu<Float,3> k_null_grid({ngrid_h, ngrid_h, ngrid_v});
    const Float k_ext_null_min = Float(1e-3);

    create_knull_grid<<<grid_kn, block_kn>>>(
            ncol_x, ncol_y, nlay, k_ext_null_min,
            k_ext.ptr(), k_null_grid.ptr());

    // initialise output arrays and set to 0
    Array_gpu<Float,2> toa_down_count({ncol_x, ncol_y});
    Array_gpu<Float,2> tod_up_count({ncol_x, ncol_y});
    Array_gpu<Float,2> surface_down_direct_count({ncol_x, ncol_y});
    Array_gpu<Float,2> surface_down_diffuse_count({ncol_x, ncol_y});
    Array_gpu<Float,2> surface_up_count({ncol_x, ncol_y});
    Array_gpu<Float,3> atmos_direct_count({ncol_x, ncol_y, nlay});
    Array_gpu<Float,3> atmos_diffuse_count({ncol_x, ncol_y, nlay});

    rrtmgp_kernel_launcher_cuda_rt::zero_array(ncol_x, ncol_y, toa_down_count.ptr());
    rrtmgp_kernel_launcher_cuda_rt::zero_array(ncol_x, ncol_y, tod_up_count.ptr());
    rrtmgp_kernel_launcher_cuda_rt::zero_array(ncol_x, ncol_y, surface_down_direct_count.ptr());
    rrtmgp_kernel_launcher_cuda_rt::zero_array(ncol_x, ncol_y, surface_down_diffuse_count.ptr());
    rrtmgp_kernel_launcher_cuda_rt::zero_array(ncol_x, ncol_y, surface_up_count.ptr());
    rrtmgp_kernel_launcher_cuda_rt::zero_array(ncol_x, ncol_y, nlay, atmos_direct_count.ptr());
    rrtmgp_kernel_launcher_cuda_rt::zero_array(ncol_x, ncol_y, nlay, atmos_diffuse_count.ptr());

    // domain sizes
    const Float x_size = ncol_x * dx_grid;
    const Float y_size = ncol_y * dy_grid;
    const Float z_size = nlay * dz_grid;

    // direction of direct rays. Take into account that azimuth is 0 north and increases clockwise
    const Float dir_x = -std::sin(zenith_angle) * std::cos(Float(0.5*M_PI) - azimuth_angle);
    const Float dir_y = -std::sin(zenith_angle) * std::sin(Float(0.5*M_PI) - azimuth_angle);
    const Float dir_z = -std::cos(zenith_angle);


    dim3 grid{grid_size}, block{block_size};
    Int photons_per_thread = photons_to_shoot / (grid_size * block_size);

    ray_tracer_kernel<<<grid, block>>>(
            photons_per_thread, k_null_grid.ptr(),
            toa_down_count.ptr(),
            tod_up_count.ptr(),
            surface_down_direct_count.ptr(),
            surface_down_diffuse_count.ptr(),
            surface_up_count.ptr(),
            atmos_direct_count.ptr(),
            atmos_diffuse_count.ptr(),
            k_ext.ptr(), ssa_asy.ptr(),
            tod_inc_direct,
            tod_inc_diffuse,
            surface_albedo.ptr(),
            x_size, y_size, z_size,
            dx_grid, dy_grid, dz_grid,
            dir_x, dir_y, dir_z,
            ncol_x, ncol_y, nlay,
            this->qrng_vectors_gpu, this->qrng_constants_gpu);

    // convert counts to fluxes
    const Float photons_per_col = Float(photons_to_shoot) / (ncol_x * ncol_y);

    const Float toa_src = tod_inc_direct + tod_inc_diffuse;
    count_to_flux_2d<<<grid_2d, block_2d>>>(
            ncol_x, ncol_y, photons_per_col,
            toa_src,
            tod_up_count.ptr(),
            surface_down_direct_count.ptr(),
            surface_down_diffuse_count.ptr(),
            surface_up_count.ptr(),
            flux_tod_up.ptr(),
            flux_sfc_dir.ptr(),
            flux_sfc_dif.ptr(),
            flux_sfc_up.ptr());

    count_to_flux_3d<<<grid_3d, block_3d>>>(
            ncol_x, ncol_y, nlay, photons_per_col,
            toa_src,
            atmos_direct_count.ptr(),
            atmos_diffuse_count.ptr(),
            flux_abs_dir.ptr(),
            flux_abs_dif.ptr());
}

