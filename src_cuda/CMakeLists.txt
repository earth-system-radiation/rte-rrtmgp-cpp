#
# This file is part of a C++ interface to the Radiative Transfer for Energetics (RTE)
#
FILE(GLOB sourcefiles_cuda "../src/*.cu")
include_directories("../include" SYSTEM ${INCLUDE_DIRS})

cuda_add_library(rte_rrtmgp STATIC ${sourcefiles_cuda})
target_link_libraries(rte_rrtmgp_cuda rte_rrtmgp_kernels_cuda)
