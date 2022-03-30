#
# This file is part of a C++ interface to the Radiative Transfer for Energetics (RTE)
#
FILE(GLOB sourcefiles_cuda_rt "../src_cuda_rt/*.cu")
include_directories("../include" "../include_rt" SYSTEM ${INCLUDE_DIRS})

add_library(rte_rrtmgp_cuda_rt STATIC ${sourcefiles_cuda_rt})
target_link_libraries(rte_rrtmgp_cuda_rt rte_rrtmgp_kernels_cuda_rt)
