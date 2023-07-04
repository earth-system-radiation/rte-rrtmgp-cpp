FILE(GLOB sourcefiles
    "../src_kernels_cuda/gas_optics_rrtmgp_kernels_launchers.cu"
    "../src_kernels_cuda/optical_props_kernels_launchers.cu"
    "../src_kernels_cuda/rte_solver_kernels_launchers.cu"
    "../src_kernels_cuda/subset_kernels_launchers.cu"
    "../src_kernels_cuda/fluxes_kernels_launchers.cu")

include_directories("../include" "../include_kernels_cuda" SYSTEM ${INCLUDE_DIRS})

add_library(rte_rrtmgp_kernels_cuda STATIC ${sourcefiles})
