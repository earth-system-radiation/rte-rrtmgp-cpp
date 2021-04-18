#include <chrono>
#include <iomanip>

#include "subset_kernel_launcher_cuda.h"
#include "tools_gpu.h"
#include "Array.h"

namespace
{
    template<typename TF>__global__
    void get_from_subset_kernel(const int ncol, const int nbnd, const int ncol_in, const int col_s_in,
                  TF* __restrict__ var_full, const TF* __restrict__ var_sub)
    {
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;
        const int ibnd = blockIdx.y*blockDim.y + threadIdx.y;
        if ( (icol < ncol_in) && (ibnd < nbnd) )
        {
            const int idx_full = icol+col_s_in-1 + ibnd*ncol;
            const int idx_sub = icol + ibnd*ncol_in;
            var_full[idx_full] = var_sub[idx_sub];
        }
    }

    template<typename TF>__global__
    void get_from_subset_kernel(const int ncol, const int nlay, const int ncol_in, const int col_s_in,
                  TF* __restrict__ var1_full, TF* __restrict__ var2_full, TF* __restrict__ var3_full,  TF* __restrict__ var4_full,
                  const TF* __restrict__ var1_sub, const TF* __restrict__ var2_sub, const TF* __restrict__ var3_sub, const TF* __restrict__ var4_sub)
    {
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;
        const int ilay = blockIdx.y*blockDim.y + threadIdx.y;
        if ( (icol < ncol_in) && (ilay < nlay) )
        {
            const int idx_full = icol+col_s_in-1 + ilay*ncol;
            const int idx_sub = icol + ilay*ncol_in;
            var1_full[idx_full] = var1_sub[idx_sub];
            var2_full[idx_full] = var2_sub[idx_sub];
            var3_full[idx_full] = var3_sub[idx_sub];
            var4_full[idx_full] = var4_sub[idx_sub];
        }
    }

    template<typename TF>__global__
    void get_from_subset_kernel(const int ncol, const int nlay, const int ncol_in, const int col_s_in,
                  TF* __restrict__ var1_full, TF* __restrict__ var2_full, TF* __restrict__ var3_full,
                  const TF* __restrict__ var1_sub, const TF* __restrict__ var2_sub, const TF* __restrict__ var3_sub)
    {
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;
        const int ilay = blockIdx.y*blockDim.y + threadIdx.y;
        if ( (icol < ncol_in) && (ilay < nlay) )
        {
            const int idx_full = icol+col_s_in-1 + ilay*ncol;
            const int idx_sub = icol + ilay*ncol_in;
            var1_full[idx_full] = var1_sub[idx_sub];
            var2_full[idx_full] = var2_sub[idx_sub];
            var3_full[idx_full] = var3_sub[idx_sub];
        }
    }

    template<typename TF>__global__
    void get_from_subset_kernel(const int ncol, const int nlay, const int nbnd, const int ncol_in, const int col_s_in,
                  TF* __restrict__ var1_full, TF* __restrict__ var2_full, TF* __restrict__ var3_full,  TF* __restrict__ var4_full,
                  const TF* __restrict__ var1_sub, const TF* __restrict__ var2_sub, const TF* __restrict__ var3_sub, const TF* __restrict__ var4_sub)
    {
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;
        const int ilay = blockIdx.y*blockDim.y + threadIdx.y;
        const int ibnd = blockIdx.z*blockDim.z + threadIdx.z;
        if ( (icol < ncol_in) && (ilay < nlay) && (ibnd < nbnd) )
        {
            const int idx_full = icol+col_s_in-1 + ilay*ncol + ibnd*nlay*ncol;
            const int idx_sub = icol + ilay*ncol_in + ibnd*nlay*ncol_in;
            var1_full[idx_full] = var1_sub[idx_sub];
            var2_full[idx_full] = var2_sub[idx_sub];
            var3_full[idx_full] = var3_sub[idx_sub];
            var4_full[idx_full] = var4_sub[idx_sub];
        }
    }

    template<typename TF>__global__
    void get_from_subset_kernel(const int ncol, const int nlay, const int nbnd, const int ncol_in, const int col_s_in,
                  TF* __restrict__ var1_full, TF* __restrict__ var2_full, TF* __restrict__ var3_full,
                  const TF* __restrict__ var1_sub, const TF* __restrict__ var2_sub, const TF* __restrict__ var3_sub)
    {
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;
        const int ilay = blockIdx.y*blockDim.y + threadIdx.y;
        const int ibnd = blockIdx.z*blockDim.z + threadIdx.z;
        if ( (icol < ncol_in) && (ilay < nlay) && (ibnd < nbnd) )
        {
            const int idx_full = icol+col_s_in-1 + ilay*ncol + ibnd*nlay*ncol;
            const int idx_sub = icol + ilay*ncol_in + ibnd*nlay*ncol_in;
            var1_full[idx_full] = var1_sub[idx_sub];
            var2_full[idx_full] = var2_sub[idx_sub];
            var3_full[idx_full] = var3_sub[idx_sub];
        }
    }
}

namespace subset_kernel_launcher_cuda
{
    template<typename TF>
    void get_from_subset(const int ncol, const int nlay, const int ncol_in, const int col_s_in,
                  Array_gpu<TF,2>& var1_full, Array_gpu<TF,2>& var2_full, Array_gpu<TF,2>& var3_full,  Array_gpu<TF,2>& var4_full,
                  const Array_gpu<TF,2>& var1_sub, const Array_gpu<TF,2>& var2_sub, const Array_gpu<TF,2>& var3_sub, const Array_gpu<TF,2>& var4_sub)
    {
        const int block_col = 16;
        const int block_lay = 16;

        const int grid_col  = ncol_in/block_col + (ncol_in%block_col > 0);
        const int grid_lay  = nlay/block_lay + (nlay%block_lay > 0);

        dim3 grid_gpu(grid_col, grid_lay);
        dim3 block_gpu(block_col, block_lay);
        get_from_subset_kernel<<<grid_gpu, block_gpu>>>(ncol, nlay, ncol_in, col_s_in, var1_full.ptr(), var2_full.ptr(),
                                                        var3_full.ptr(), var4_full.ptr(), var1_sub.ptr(), var2_sub.ptr(),
                                                        var3_sub.ptr(), var4_sub.ptr());
    }

    template<typename TF>
    void get_from_subset(const int ncol, const int nlay, const int ncol_in, const int col_s_in,
                  Array_gpu<TF,2>& var1_full, Array_gpu<TF,2>& var2_full, Array_gpu<TF,2>& var3_full,
                  const Array_gpu<TF,2>& var1_sub, const Array_gpu<TF,2>& var2_sub, const Array_gpu<TF,2>& var3_sub)
    {
        const int block_col = 16;
        const int block_lay = 16;

        const int grid_col  = ncol_in/block_col + (ncol_in%block_col > 0);
        const int grid_lay  = nlay/block_lay + (nlay%block_lay > 0);

        dim3 grid_gpu(grid_col, grid_lay);
        dim3 block_gpu(block_col, block_lay);
        get_from_subset_kernel<<<grid_gpu, block_gpu>>>(ncol, nlay, ncol_in, col_s_in, var1_full.ptr(), var2_full.ptr(),
                                                        var3_full.ptr(), var1_sub.ptr(), var2_sub.ptr(),
                                                        var3_sub.ptr());
    }

    template<typename TF>
    void get_from_subset(const int ncol, const int nlay, const int nbnd, const int ncol_in, const int col_s_in,
                  Array_gpu<TF,3>& var1_full, Array_gpu<TF,3>& var2_full, Array_gpu<TF,3>& var3_full,  Array_gpu<TF,3>& var4_full,
                  const Array_gpu<TF,3>& var1_sub, const Array_gpu<TF,3>& var2_sub, const Array_gpu<TF,3>& var3_sub, const Array_gpu<TF,3>& var4_sub)
    {
        const int block_col = 16;
        const int block_lay = 16;
        const int block_bnd = 1;

        const int grid_col  = ncol_in/block_col + (ncol_in%block_col > 0);
        const int grid_lay  = nlay/block_lay + (nlay%block_lay > 0);
        const int grid_bnd  = nbnd/block_bnd + (nbnd%block_bnd > 0);

        dim3 grid_gpu(grid_col, grid_lay, grid_bnd);
        dim3 block_gpu(block_col, block_lay, block_bnd);
        get_from_subset_kernel<<<grid_gpu, block_gpu>>>(ncol, nlay, nbnd, ncol_in, col_s_in, var1_full.ptr(), var2_full.ptr(),
                                                        var3_full.ptr(), var4_full.ptr(), var1_sub.ptr(), var2_sub.ptr(),
                                                        var3_sub.ptr(), var4_sub.ptr());
    }

    template<typename TF>
    void get_from_subset(const int ncol, const int nlay, const int nbnd, const int ncol_in, const int col_s_in,
                  Array_gpu<TF,3>& var1_full, Array_gpu<TF,3>& var2_full, Array_gpu<TF,3>& var3_full,
                  const Array_gpu<TF,3>& var1_sub, const Array_gpu<TF,3>& var2_sub, const Array_gpu<TF,3>& var3_sub)
    {
        const int block_col = 16;
        const int block_lay = 16;
        const int block_bnd = 1;

        const int grid_col  = ncol_in/block_col + (ncol_in%block_col > 0);
        const int grid_lay  = nlay/block_lay + (nlay%block_lay > 0);
        const int grid_bnd  = nbnd/block_bnd + (nbnd%block_bnd > 0);

        dim3 grid_gpu(grid_col, grid_lay, grid_bnd);
        dim3 block_gpu(block_col, block_lay, block_bnd);
        get_from_subset_kernel<<<grid_gpu, block_gpu>>>(ncol, nlay, nbnd, ncol_in, col_s_in, var1_full.ptr(), var2_full.ptr(),
                                                        var3_full.ptr(), var1_sub.ptr(), var2_sub.ptr(), var3_sub.ptr());
    }

    template<typename TF>
    void copy_to_subset(const int ncol, const int nlay, const int nbnd, const int ncol_in, const int col_s_in,
                  Array_gpu<TF,3>& var1_full, Array_gpu<TF,3>& var2_full, Array_gpu<TF,3>& var3_full,
                  const Array_gpu<TF,3>& var1_sub, const Array_gpu<TF,3>& var2_sub, const Array_gpu<TF,3>& var3_sub)
    {
        const int block_col = 16;
        const int block_lay = 16;
        const int block_bnd = 1;

        const int grid_col  = ncol_in/block_col + (ncol_in%block_col > 0);
        const int grid_lay  = nlay/block_lay + (nlay%block_lay > 0);
        const int grid_bnd  = nbnd/block_bnd + (nbnd%block_bnd > 0);

        dim3 grid_gpu(grid_col, grid_lay, grid_bnd);
        dim3 block_gpu(block_col, block_lay, block_bnd);
        copy_to_subset_kernel<<<grid_gpu, block_gpu>>>(ncol, nlay, nbnd, ncol_in, col_s_in, var1_full.ptr(), var2_full.ptr(),
                                                        var3_full.ptr(), var1_sub.ptr(), var2_sub.ptr(), var3_sub.ptr());
    }

    template<typename TF>
    void get_from_subset(const int ncol, const int nbnd, const int ncol_in, const int col_s_in,
                  Array_gpu<TF,2>& var_full, const Array_gpu<TF,2>& var_sub)
    {
        const int block_col = 16;
        const int block_bnd = 14;

        const int grid_col  = ncol_in/block_col + (ncol_in%block_col > 0);
        const int grid_bnd  = nbnd/block_bnd + (nbnd%block_bnd > 0);

        dim3 grid_gpu(grid_col, grid_bnd);
        dim3 block_gpu(block_col, block_bnd);
        get_from_subset_kernel<<<grid_gpu, block_gpu>>>(ncol, nbnd, ncol_in, col_s_in, var_full.ptr(), var_sub.ptr());
    }
}

#ifdef RTE_RRTMGP_SINGLE_PRECISION
template void subset_kernel_launcher_cuda::get_from_subset(const int ncol, const int nlay, const int ncol_in, const int col_s_in,
              Array_gpu<float,2>&, Array_gpu<float,2>&, Array_gpu<float,2>&, Array_gpu<float,2>&,
              const Array_gpu<float,2>&, const Array_gpu<float,2>&, const Array_gpu<float,2>&, const Array_gpu<float,2>&);

template void subset_kernel_launcher_cuda::get_from_subset(const int ncol, const int nlay, const int ncol_in, const int col_s_in,
              Array_gpu<float,2>&, Array_gpu<float,2>&, Array_gpu<float,2>&,
              const Array_gpu<float,2>&, const Array_gpu<float,2>&, const Array_gpu<float,2>&);

template void subset_kernel_launcher_cuda::get_from_subset(const int ncol, const int nlay, const int nbnd, const int ncol_in, const int col_s_in,
              Array_gpu<float,3>&, Array_gpu<float,3>&, Array_gpu<float,3>&, Array_gpu<float,3>&,
              const Array_gpu<float,3>&, const Array_gpu<float,3>&, const Array_gpu<float,3>&, const Array_gpu<float,3>&);

template void subset_kernel_launcher_cuda::get_from_subset(const int ncol, const int nlay, const int nbnd, const int ncol_in, const int col_s_in,
              Array_gpu<float,3>&, Array_gpu<float,3>&, Array_gpu<float,3>&, const Array_gpu<float,3>&, const Array_gpu<float,3>&,
              const Array_gpu<float,3>&);

template void subset_kernel_launcher_cuda::get_from_subset(const int ncol, const int nbnd, const int ncol_in, const int col_s_in,
              Array_gpu<float,2>&, const Array_gpu<float,2>&);

#else
template void subset_kernel_launcher_cuda::get_from_subset(const int ncol, const int nlay, const int ncol_in, const int col_s_in,
              Array_gpu<double,2>&, Array_gpu<double,2>&, Array_gpu<double,2>&, Array_gpu<double,2>&,
              const Array_gpu<double,2>&, const Array_gpu<double,2>&, const Array_gpu<double,2>&, const Array_gpu<double,2>&);

template void subset_kernel_launcher_cuda::get_from_subset(const int ncol, const int nlay, const int ncol_in, const int col_s_in,
              Array_gpu<double,2>&, Array_gpu<double,2>&, Array_gpu<double,2>&,
              const Array_gpu<double,2>&, const Array_gpu<double,2>&, const Array_gpu<double,2>&);

template void subset_kernel_launcher_cuda::get_from_subset(const int ncol, const int nlay, const int nbnd, const int ncol_in, const int col_s_in,
              Array_gpu<double,3>&, Array_gpu<double,3>&, Array_gpu<double,3>&, Array_gpu<double,3>&,
              const Array_gpu<double,3>&, const Array_gpu<double,3>&, const Array_gpu<double,3>&, const Array_gpu<double,3>&);

template void subset_kernel_launcher_cuda::get_from_subset(const int ncol, const int nlay, const int nbnd, const int ncol_in, const int col_s_in,
              Array_gpu<double,3>&, Array_gpu<double,3>&, Array_gpu<double,3>&, const Array_gpu<double,3>&, const Array_gpu<double,3>&,
              const Array_gpu<double,3>&);

template void subset_kernel_launcher_cuda::get_from_subset(const int ncol, const int nbnd, const int ncol_in, const int col_s_in,
              Array_gpu<double,2>&, const Array_gpu<double,2>&);
#endif
