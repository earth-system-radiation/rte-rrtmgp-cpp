#include "rrtmgp_kernel_launcher_cuda.h"
#include "tools_gpu.h"

#include "Array.h"

namespace
{
    // Add the kernel here.
}

namespace rrtmgp_kernel_launcher_cuda
{
    template<typename TF>
    void combine_and_reorder_2str(
            const int ncol, const int nlay, const int ngpt,
            const Array<TF,3>& tau_local, const Array<TF,3>& tau_rayleigh,
            Array<TF,3>& tau, Array<TF,3>& ssa, Array<TF,3>& g)
    {
        // Store the sizes.
        const int tau_local_size = tau_local.size()*sizeof(TF);
        const int tau_rayleigh_size = tau_rayleigh.size()*sizeof(TF);
        const int tau_size = tau.size()*sizeof(TF);
        const int ssa_size = ssa.size()*sizeof(TF);
        const int g_size = g.size()*sizeof(TF);

        TF* tau_local_gpu;
        TF* tau_rayleigh_gpu;
        TF* tau_gpu;
        TF* ssa_gpu;
        TF* g_gpu;

        // Allocate a CUDA array.
        cuda_safe_call(cudaMalloc((void**)&tau_local_gpu, tau_local_size));
        cuda_safe_call(cudaMalloc((void**)&tau_rayleigh_gpu, tau_rayleigh_size));
        cuda_safe_call(cudaMalloc((void**)&tau_gpu, tau_size));
        cuda_safe_call(cudaMalloc((void**)&ssa_gpu, ssa_size));
        cuda_safe_call(cudaMalloc((void**)&g_gpu, g_size));

        // Copy the data to the GPU.
        cuda_safe_call(cudaMemcpy(tau_local_gpu, tau_local.ptr(), tau_local_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(tau_rayleigh_gpu, tau_rayleigh.ptr(), tau_rayleigh_size, cudaMemcpyHostToDevice));

        // Call the kernel.
        // TO BE IMPLEMENTED...

        // Copy back the results.
        cuda_safe_call(cudaMemcpy(tau.ptr(), tau_gpu, tau_size, cudaMemcpyDeviceToHost));
        cuda_safe_call(cudaMemcpy(ssa.ptr(), ssa_gpu, ssa_size, cudaMemcpyDeviceToHost));
        cuda_safe_call(cudaMemcpy(g.ptr(), g_gpu, g_size, cudaMemcpyDeviceToHost));
    }
}

#ifdef FLOAT_SINGLE_RRTMGP
template void rrtmgp_kernel_launcher_cuda::combine_and_reorder_2str<float>(
        const int, const int, const int, const Array<float,3>&, const Array<float,3>&, Array<float,3>&, Array<float,3>&, Array<float,3>&);
#else
template void rrtmgp_kernel_launcher_cuda::combine_and_reorder_2str<double>(
        const int, const int, const int, const Array<double,3>&, const Array<double,3>&, Array<double,3>&, Array<double,3>&, Array<double,3>&);
#endif
