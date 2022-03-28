#include <chrono>
#include <iomanip>

#include "gpt_combine_kernel_launcher_cuda_rt.h"
#include "tools_gpu.h"
#include "Array.h"

namespace
{
    #include "gpt_combine_kernels_rt.cu"
}

namespace gpt_combine_kernel_launcher_cuda_rt
{
    
    void add_from_gpoint(const int ncol, const int nlay,
                  Array_gpu<Float,2>& var1_full, Array_gpu<Float,2>& var2_full, Array_gpu<Float,2>& var3_full,  Array_gpu<Float,2>& var4_full,
                  const Array_gpu<Float,2>& var1_sub, const Array_gpu<Float,2>& var2_sub, const Array_gpu<Float,2>& var3_sub, const Array_gpu<Float,2>& var4_sub )
    {
        const int block_col = 16;
        const int block_lay = 16;

        const int grid_col = ncol/block_col + (ncol%block_col > 0);
        const int grid_lay = nlay/block_lay + (nlay%block_lay > 0);

        dim3 grid_gpu(grid_col, grid_lay);
        dim3 block_gpu(block_col, block_lay);

        add_from_gpoint_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, var1_full.ptr(), var2_full.ptr(),
                var3_full.ptr(), var4_full.ptr(), var1_sub.ptr(), var2_sub.ptr(),
                var3_sub.ptr(), var4_sub.ptr());
    }

    
    void add_from_gpoint(const int ncol, const int nlay,
                  Array_gpu<Float,3>& var1_full, Array_gpu<Float,3>& var2_full,
                  const Array_gpu<Float,3>& var1_sub, const Array_gpu<Float,3>& var2_sub)
    {
        const int block_col = 16;
        const int block_lay = 16;

        const int grid_col = ncol/block_col + (ncol%block_col > 0);
        const int grid_lay = nlay/block_lay + (nlay%block_lay > 0);

        dim3 grid_gpu(grid_col, grid_lay);
        dim3 block_gpu(block_col, block_lay);

        add_from_gpoint_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, var1_full.ptr(), var2_full.ptr(),
                var1_sub.ptr(), var2_sub.ptr());
    }
    
    
    void add_from_gpoint(const int ncol, const int nlay,
                  Array_gpu<Float,2>& var1_full, Array_gpu<Float,2>& var2_full, Array_gpu<Float,2>& var3_full,
                  const Array_gpu<Float,2>& var1_sub, const Array_gpu<Float,2>& var2_sub, const Array_gpu<Float,2>& var3_sub)
    {
        const int block_col = 16;
        const int block_lay = 16;

        const int grid_col = ncol/block_col + (ncol%block_col > 0);
        const int grid_lay = nlay/block_lay + (nlay%block_lay > 0);

        dim3 grid_gpu(grid_col, grid_lay);
        dim3 block_gpu(block_col, block_lay);

        add_from_gpoint_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, var1_full.ptr(), var2_full.ptr(),
                var3_full.ptr(), var1_sub.ptr(), var2_sub.ptr(),
                var3_sub.ptr());
    }
    
    
    void get_from_gpoint(const int ncol, const int nlay, const int igpt,
                  Array_gpu<Float,3>& var1_full, Array_gpu<Float,3>& var2_full, Array_gpu<Float,3>& var3_full,  Array_gpu<Float,3>& var4_full,
                  const Array_gpu<Float,2>& var1_sub, const Array_gpu<Float,2>& var2_sub, const Array_gpu<Float,2>& var3_sub, const Array_gpu<Float,2>& var4_sub)
    {
        const int block_col = 16;
        const int block_lay = 16;

        const int grid_col = ncol/block_col + (ncol%block_col > 0);
        const int grid_lay = nlay/block_lay + (nlay%block_lay > 0);

        dim3 grid_gpu(grid_col, grid_lay);
        dim3 block_gpu(block_col, block_lay);
        get_from_gpoint_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, igpt, var1_full.ptr(), var2_full.ptr(),
                var3_full.ptr(), var4_full.ptr(), var1_sub.ptr(), var2_sub.ptr(),
                var3_sub.ptr(), var4_sub.ptr());
    }

    
    void get_from_gpoint(const int ncol, const int nlay, const int igpt,
                  Array_gpu<Float,3>& var1_full, Array_gpu<Float,3>& var2_full, Array_gpu<Float,3>& var3_full,
                  const Array_gpu<Float,2>& var1_sub, const Array_gpu<Float,2>& var2_sub, const Array_gpu<Float,2>& var3_sub)
    {
        const int block_col = 16;
        const int block_lay = 16;

        const int grid_col = ncol/block_col + (ncol%block_col > 0);
        const int grid_lay = nlay/block_lay + (nlay%block_lay > 0);

        dim3 grid_gpu(grid_col, grid_lay);
        dim3 block_gpu(block_col, block_lay);
        get_from_gpoint_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, igpt, var1_full.ptr(), var2_full.ptr(),
                var3_full.ptr(), var1_sub.ptr(), var2_sub.ptr(), var3_sub.ptr());
    }

    
    void get_from_gpoint(const int ncol, const int igpt,
                  Array_gpu<Float,2>& var_full, const Array_gpu<Float,1>& var_sub)
    {
        const int block_col = 16;

        const int grid_col = ncol/block_col + (ncol%block_col > 0);

        dim3 grid_gpu(grid_col, 1);
        dim3 block_gpu(block_col, 1);
        get_from_gpoint_kernel<<<grid_gpu, block_gpu>>>(ncol, igpt, var_full.ptr(), var_sub.ptr());
    }
}
