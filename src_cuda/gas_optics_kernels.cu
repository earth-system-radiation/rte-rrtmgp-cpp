#include "rrtmgp_kernel_launcher_cuda.h"
#include "tools_gpu.h"

#include "Array.h"

namespace
{
    // Add the kernel here.
    template<typename TF>__global__
    void combine_and_reorder_2str_kernel(
            const int ncol, const int nlay, const int ngpt,
            const TF* __restrict__ tau_abs, const TF* __restrict__ tau_rayleigh,
            TF* __restrict__ tau, TF* __restrict__ ssa, TF* __restrict__ g)
    {
        // Fetch the three coordinates.
        const int igpt = blockIdx.x*blockDim.x + threadIdx.x;
        const int ilay = blockIdx.y*blockDim.y + threadIdx.y;
        const int icol = blockIdx.z*blockDim.z + threadIdx.z;

        if ( (icol < ncol) && (ilay < nlay) && (igpt < ngpt) )
        {
            const int idx_in  = igpt + ilay*ngpt + icol*(ngpt*nlay);
            const int idx_out = icol + ilay*ncol + igpt*(ncol*nlay);

            tau[idx_out] = tau_abs[idx_in] + TF(100.)*(ilay+1);
        }
    }
}

namespace rrtmgp_kernel_launcher_cuda
{
    template<typename TF>
    void combine_and_reorder_2str(
            const int ncol, const int nlay, const int ngpt,
            const Array<TF,3>& tau_abs, const Array<TF,3>& tau_rayleigh,
            Array<TF,3>& tau, Array<TF,3>& ssa, Array<TF,3>& g)
    {
        // Store the sizes, all the same
        const int array_size = tau_abs.size()*sizeof(TF);

        TF* tau_abs_gpu;
        TF* tau_rayleigh_gpu;
        TF* tau_gpu;
        TF* ssa_gpu;
        TF* g_gpu;

        // Allocate a CUDA array.
        cuda_safe_call(cudaMalloc((void**)&tau_abs_gpu, array_size));
        cuda_safe_call(cudaMalloc((void**)&tau_rayleigh_gpu, array_size));
        cuda_safe_call(cudaMalloc((void**)&tau_gpu, array_size));
        cuda_safe_call(cudaMalloc((void**)&ssa_gpu, array_size));
        cuda_safe_call(cudaMalloc((void**)&g_gpu, array_size));

        // Copy the data to the GPU.
        cuda_safe_call(cudaMemcpy(tau_abs_gpu, tau_abs.ptr(), array_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(tau_rayleigh_gpu, tau_rayleigh.ptr(), array_size, cudaMemcpyHostToDevice));

        // Call the kernel.
        // CvH: THIS KERNEL IS JUST SOME RANDOM CODE, IT NEEDS TO BE IMPLEMENTED.
        const int block_gpt = 1;
        const int block_lay = 1;
        const int block_col = 1;

        const int grid_gpt  = ngpt/block_gpt + (ngpt%block_gpt > 0);
        const int grid_lay  = nlay/block_lay + (nlay%block_lay > 0);
        const int grid_col  = ncol/block_col + (ncol%block_col > 0);

        dim3 grid_gpu(grid_gpt, grid_lay, grid_col);
        dim3 block_gpu(block_gpt, block_lay, block_col);

        combine_and_reorder_2str_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, ngpt,
                tau_abs_gpu, tau_rayleigh_gpu,
                tau_gpu, ssa_gpu, g_gpu);
        cuda_check_error();

        // Copy back the results.
        cuda_safe_call(cudaMemcpy(tau.ptr(), tau_gpu, array_size, cudaMemcpyDeviceToHost));
        cuda_safe_call(cudaMemcpy(ssa.ptr(), ssa_gpu, array_size, cudaMemcpyDeviceToHost));
        cuda_safe_call(cudaMemcpy(g.ptr(), g_gpu, array_size, cudaMemcpyDeviceToHost));

        // Deallocate a CUDA array.
        cuda_safe_call(cudaFree(tau_abs_gpu));
        cuda_safe_call(cudaFree(tau_rayleigh_gpu));
        cuda_safe_call(cudaFree(tau_gpu));
        cuda_safe_call(cudaFree(ssa_gpu));
        cuda_safe_call(cudaFree(g_gpu));
    }
}

#ifdef FLOAT_SINGLE_RRTMGP
template void rrtmgp_kernel_launcher_cuda::combine_and_reorder_2str<float>(
        const int, const int, const int, const Array<float,3>&, const Array<float,3>&, Array<float,3>&, Array<float,3>&, Array<float,3>&);
#else
template void rrtmgp_kernel_launcher_cuda::combine_and_reorder_2str<double>(
        const int, const int, const int, const Array<double,3>&, const Array<double,3>&, Array<double,3>&, Array<double,3>&, Array<double,3>&);
#endif
