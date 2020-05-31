#include "rrtmgp_kernel_launcher_cuda.h"
#include "Array.h"


// CvH: MOVE THIS OUT FOR REUSE LATER.
#define cuda_safe_call(err) ::__cuda_safe_call(err, __FILE__, __LINE__)
#define cuda_check_error()  ::__cuda_check_error(__FILE__, __LINE__)

namespace
{
    /* CUDA error checking, from: http://choorucode.com/2011/03/02/how-to-do-error-checking-in-cuda/
       In debug mode, CUDACHECKS is defined and all kernel calls are checked with cudaCheckError().
       All CUDA api calls are always checked with cudaSafeCall() */

    // Wrapper to check for errors in CUDA api calls (e.g. cudaMalloc)
    inline void __cuda_safe_call(cudaError err, const char *file, const int line)
    {
        if (cudaSuccess != err)
        {
            printf("cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
            throw 1;
        }
    }

    // Function to check for errors in CUDA kernels. Call directly after kernel.
    inline void __cuda_check_error(const char *file, const int line)
    {
        #ifdef CUDACHECKS
        cudaError err = cudaGetLastError();
        if (cudaSuccess != err)
        {
            printf("cudaCheckError() failed at %s:%i : %s\n", file, line, cudaGetErrorString( err ) );
            throw 1;
        }

        err = cudaDeviceSynchronize();
        if(cudaSuccess != err)
        {
            printf("cudaCheckError() with sync failed at %s:%i : %s\n", file, line, cudaGetErrorString( err ) );
            throw 1;
        }
        #endif
    }
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
