FILE(GLOB sourcefiles "../src_kernels_cuda/*_kernels.cu")
include_directories("../include" SYSTEM ${INCLUDE_DIRS})

cuda_add_library(rte_rrtmgp_kernels_cuda STATIC ${sourcefiles})
