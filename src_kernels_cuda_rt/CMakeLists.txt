FILE(GLOB sourcefiles_rt
    "../src_kernels_cuda_rt/gas_optics_kernel_launchers_rt.cu"
    "../src_kernels_cuda_rt/optical_props_kernel_launchers_rt.cu"
    "../src_kernels_cuda_rt/rte_solver_kernel_launchers_rt.cu"
    "../src_kernels_cuda_rt/subset_kernel_launchers_rt.cu"
    "../src_kernels_cuda_rt/gpt_combine_kernel_launchers_rt.cu"
    "../src_kernels_cuda_rt/raytracer_kernels.cu"
    "../src_kernels_cuda_rt/raytracer_kernels_bw.cu")

include_directories("../include" "../include_rt" SYSTEM ${INCLUDE_DIRS})

add_library(rte_rrtmgp_kernels_cuda_rt STATIC ${sourcefiles_rt})
