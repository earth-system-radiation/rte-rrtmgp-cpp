#include <chrono>
#include <iomanip>

#include "subset_kernel_launcher_cuda.h"
#include "tools_gpu.h"
#include "Array.h"

namespace
{
    #include "subset_kernels.cu"
}

namespace subset_kernel_launcher_cuda
{
    void get_from_subset(const int ncol, const int nlay, const int ncol_in, const int col_s_in,
                  Array_gpu<Float,2>& var1_full, Array_gpu<Float,2>& var2_full, Array_gpu<Float,2>& var3_full,  Array_gpu<Float,2>& var4_full,
                  const Array_gpu<Float,2>& var1_sub, const Array_gpu<Float,2>& var2_sub, const Array_gpu<Float,2>& var3_sub, const Array_gpu<Float,2>& var4_sub)
    {
        const int block_col = 16;
        const int block_lay = 16;

        const int grid_col = ncol_in/block_col + (ncol_in%block_col > 0);
        const int grid_lay = nlay/block_lay + (nlay%block_lay > 0);

        dim3 grid_gpu(grid_col, grid_lay);
        dim3 block_gpu(block_col, block_lay);
        get_from_subset_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, ncol_in, col_s_in, var1_full.ptr(), var2_full.ptr(),
                var3_full.ptr(), var4_full.ptr(), var1_sub.ptr(), var2_sub.ptr(),
                var3_sub.ptr(), var4_sub.ptr());
    }


    void get_from_subset(const int ncol, const int nlay, const int ncol_in, const int col_s_in,
                  Array_gpu<Float,2>& var1_full, Array_gpu<Float,2>& var2_full, Array_gpu<Float,2>& var3_full,
                  const Array_gpu<Float,2>& var1_sub, const Array_gpu<Float,2>& var2_sub, const Array_gpu<Float,2>& var3_sub)
    {
        const int block_col = 16;
        const int block_lay = 16;

        const int grid_col = ncol_in/block_col + (ncol_in%block_col > 0);
        const int grid_lay = nlay/block_lay + (nlay%block_lay > 0);

        dim3 grid_gpu(grid_col, grid_lay);
        dim3 block_gpu(block_col, block_lay);
        get_from_subset_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, ncol_in, col_s_in, var1_full.ptr(), var2_full.ptr(),
                var3_full.ptr(), var1_sub.ptr(), var2_sub.ptr(),
                var3_sub.ptr());
    }


    void get_from_subset(const int ncol, const int nlay, const int nbnd, const int ncol_in, const int col_s_in,
                  Array_gpu<Float,3>& var1_full, Array_gpu<Float,3>& var2_full, Array_gpu<Float,3>& var3_full,  Array_gpu<Float,3>& var4_full,
                  const Array_gpu<Float,3>& var1_sub, const Array_gpu<Float,3>& var2_sub, const Array_gpu<Float,3>& var3_sub, const Array_gpu<Float,3>& var4_sub)
    {
        const int block_col = 16;
        const int block_lay = 16;
        const int block_bnd = 1;

        const int grid_col = ncol_in/block_col + (ncol_in%block_col > 0);
        const int grid_lay = nlay/block_lay + (nlay%block_lay > 0);
        const int grid_bnd = nbnd/block_bnd + (nbnd%block_bnd > 0);

        dim3 grid_gpu(grid_col, grid_lay, grid_bnd);
        dim3 block_gpu(block_col, block_lay, block_bnd);
        get_from_subset_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, nbnd, ncol_in, col_s_in, var1_full.ptr(), var2_full.ptr(),
                var3_full.ptr(), var4_full.ptr(), var1_sub.ptr(), var2_sub.ptr(),
                var3_sub.ptr(), var4_sub.ptr());
    }


    void get_from_subset(const int ncol, const int nlay, const int nbnd, const int ncol_in, const int col_s_in,
                  Array_gpu<Float,3>& var1_full, Array_gpu<Float,3>& var2_full, Array_gpu<Float,3>& var3_full,
                  const Array_gpu<Float,3>& var1_sub, const Array_gpu<Float,3>& var2_sub, const Array_gpu<Float,3>& var3_sub)
    {
        const int block_col = 16;
        const int block_lay = 16;
        const int block_bnd = 1;

        const int grid_col = ncol_in/block_col + (ncol_in%block_col > 0);
        const int grid_lay = nlay/block_lay + (nlay%block_lay > 0);
        const int grid_bnd = nbnd/block_bnd + (nbnd%block_bnd > 0);

        dim3 grid_gpu(grid_col, grid_lay, grid_bnd);
        dim3 block_gpu(block_col, block_lay, block_bnd);
        get_from_subset_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, nbnd, ncol_in, col_s_in, var1_full.ptr(), var2_full.ptr(),
                var3_full.ptr(), var1_sub.ptr(), var2_sub.ptr(), var3_sub.ptr());
    }


    /*
    void copy_to_subset(
            const int ncol, const int nlay, const int nbnd, const int ncol_in, const int col_s_in,
            Array_gpu<Float,3>& var1_full, Array_gpu<Float,3>& var2_full, Array_gpu<Float,3>& var3_full,
            const Array_gpu<Float,3>& var1_sub, const Array_gpu<Float,3>& var2_sub, const Array_gpu<Float,3>& var3_sub)
    {
        const int block_col = 16;
        const int block_lay = 16;
        const int block_bnd = 1;

        const int grid_col = ncol_in/block_col + (ncol_in%block_col > 0);
        const int grid_lay = nlay/block_lay + (nlay%block_lay > 0);
        const int grid_bnd = nbnd/block_bnd + (nbnd%block_bnd > 0);

        dim3 grid_gpu(grid_col, grid_lay, grid_bnd);
        dim3 block_gpu(block_col, block_lay, block_bnd);

        copy_to_subset_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, nbnd, ncol_in, col_s_in, var1_full.ptr(), var2_full.ptr(),
                var3_full.ptr(), var1_sub.ptr(), var2_sub.ptr(), var3_sub.ptr());
    }
    */


    void get_from_subset(
            const int ncol, const int nbnd, const int ncol_in, const int col_s_in,
            Array_gpu<Float,2>& var_full, const Array_gpu<Float,2>& var_sub)
    {
        const int block_col = 16;
        const int block_bnd = 14;

        const int grid_col = ncol_in/block_col + (ncol_in%block_col > 0);
        const int grid_bnd = nbnd/block_bnd + (nbnd%block_bnd > 0);

        dim3 grid_gpu(grid_col, grid_bnd);
        dim3 block_gpu(block_col, block_bnd);
        get_from_subset_kernel<<<grid_gpu, block_gpu>>>(ncol, nbnd, ncol_in, col_s_in, var_full.ptr(), var_sub.ptr());
    }
}
