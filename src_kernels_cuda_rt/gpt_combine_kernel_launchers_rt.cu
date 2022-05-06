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
                  Float* var1_full, Float* var2_full, Float* var3_full,  Float* var4_full,
                  const Float* var1_sub, const Float* var2_sub, const Float* var3_sub, const Float* var4_sub )
    {
        const int block_col = 16;
        const int block_lay = 16;

        const int grid_col = ncol/block_col + (ncol%block_col > 0);
        const int grid_lay = nlay/block_lay + (nlay%block_lay > 0);

        dim3 grid_gpu(grid_col, grid_lay);
        dim3 block_gpu(block_col, block_lay);

        add_from_gpoint_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, var1_full, var2_full,
                var3_full, var4_full, var1_sub, var2_sub,
                var3_sub, var4_sub);
    }

    
    void add_from_gpoint(const int ncol, const int nlay,
                  Float* var1_full, Float* var2_full,
                  const Float* var1_sub, const Float* var2_sub)
    {
        const int block_col = 16;
        const int block_lay = 16;

        const int grid_col = ncol/block_col + (ncol%block_col > 0);
        const int grid_lay = nlay/block_lay + (nlay%block_lay > 0);

        dim3 grid_gpu(grid_col, grid_lay);
        dim3 block_gpu(block_col, block_lay);

        add_from_gpoint_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, var1_full, var2_full,
                var1_sub, var2_sub);
    }
    
    
    void add_from_gpoint(const int ncol, const int nlay,
                  Float* var1_full, Float* var2_full, Float* var3_full,
                  const Float* var1_sub, const Float* var2_sub, const Float* var3_sub)
    {
        const int block_col = 16;
        const int block_lay = 16;

        const int grid_col = ncol/block_col + (ncol%block_col > 0);
        const int grid_lay = nlay/block_lay + (nlay%block_lay > 0);

        dim3 grid_gpu(grid_col, grid_lay);
        dim3 block_gpu(block_col, block_lay);

        add_from_gpoint_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, var1_full, var2_full,
                var3_full, var1_sub, var2_sub,
                var3_sub);
    }
    
    
    void get_from_gpoint(const int ncol, const int nlay, const int igpt,
                  Float* var1_full, Float* var2_full, Float* var3_full,  Float* var4_full,
                  const Float* var1_sub, const Float* var2_sub, const Float* var3_sub, const Float* var4_sub)
    {
        const int block_col = 16;
        const int block_lay = 16;

        const int grid_col = ncol/block_col + (ncol%block_col > 0);
        const int grid_lay = nlay/block_lay + (nlay%block_lay > 0);

        dim3 grid_gpu(grid_col, grid_lay);
        dim3 block_gpu(block_col, block_lay);
        get_from_gpoint_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, igpt, var1_full, var2_full,
                var3_full, var4_full, var1_sub, var2_sub,
                var3_sub, var4_sub);
    }

    
    void get_from_gpoint(const int ncol, const int nlay, const int igpt,
                  Float* var1_full, Float* var2_full, Float* var3_full,
                  const Float* var1_sub, const Float* var2_sub, const Float* var3_sub)
    {
        const int block_col = 16;
        const int block_lay = 16;

        const int grid_col = ncol/block_col + (ncol%block_col > 0);
        const int grid_lay = nlay/block_lay + (nlay%block_lay > 0);

        dim3 grid_gpu(grid_col, grid_lay);
        dim3 block_gpu(block_col, block_lay);
        get_from_gpoint_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, igpt, var1_full, var2_full,
                var3_full, var1_sub, var2_sub, var3_sub);
    }

    
    void get_from_gpoint(const int ncol, const int igpt,
                  Float* var_full, const Float* var_sub)
    {
        const int block_col = 16;

        const int grid_col = ncol/block_col + (ncol%block_col > 0);

        dim3 grid_gpu(grid_col, 1);
        dim3 block_gpu(block_col, 1);
        get_from_gpoint_kernel<<<grid_gpu, block_gpu>>>(ncol, igpt, var_full, var_sub);
    }
}
