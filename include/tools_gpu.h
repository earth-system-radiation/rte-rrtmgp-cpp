#ifndef TOOLS_GPU_H
#define TOOLS_GPU_H

#define cuda_safe_call(err) Tools_gpu::__cuda_safe_call(err, __FILE__, __LINE__)
#define cuda_check_error()  Tools_gpu::__cuda_check_error(__FILE__, __LINE__)

namespace Tools_gpu
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
#endif
