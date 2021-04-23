FILE(GLOB sourcefiles
    "../src_kernels_cuda/gas_optics_kernel_launchers.cu"
    "../src_kernels_cuda/rte_solver_kernel_launchers.cu"
    "../src_kernels_cuda/subset_kernel_launchers.cu")

include_directories("../include" SYSTEM ${INCLUDE_DIRS})

cuda_add_library(rte_rrtmgp_kernels_cuda STATIC ${sourcefiles})
