#ifndef TOOLS_GPU_H
#define TOOLS_GPU_H

#define cuda_safe_call(err) Tools_gpu::__cuda_safe_call(err, __FILE__, __LINE__)
#define cuda_check_error()  Tools_gpu::__cuda_check_error(__FILE__, __LINE__)
#define cuda_check_memory() Tools_gpu::__cuda_check_memory(__FILE__, __LINE__)


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
        if (cudaSuccess != err)
        {
            printf("cudaCheckError() with sync failed at %s:%i : %s\n", file, line, cudaGetErrorString( err ) );
            throw 1;
        }
        #endif
    }

    // Check the memory usage.
    inline void __cuda_check_memory(const char *file, const int line)
    {
        #ifdef CUDACHECKS
        size_t free_byte, total_byte ;

        cudaError err = cudaMemGetInfo( &free_byte, &total_byte ) ;

        if ( cudaSuccess != err ){

            printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(err) );
            throw 1;

        }

        double used_db = (double)total_byte - (double)free_byte ;

        printf("GPU memory usage at %s:%i: %f MB\n", file, line, used_db/(1024.0*1024.0));
        #endif
    }

    template<typename T>
    void allocate_gpu(T*& data_ptr, int length)
    {
        #ifndef CUDART_VERSION
        #error CUDART_VERSION Undefined!
        #elif (CUDART_VERSION >= 11020)  
        cuda_safe_call(cudaMallocAsync((void **) &data_ptr, length*sizeof(T), 0));
        #else
        cuda_safe_call(cudaMalloc((void **) &data_ptr, length*sizeof(T)));
        #endif
    }

    template<typename T>
    void free_gpu(T*& data_ptr)
    {
        #ifndef CUDART_VERSION
        #error CUDART_VERSION Undefined!
        #elif (CUDART_VERSION >= 11020)  
        cuda_safe_call(cudaFreeAsync(data_ptr, 0));
        #else
        cuda_safe_call(cudaFree(data_ptr));
        #endif
        data_ptr = nullptr;
    }
}
#endif
