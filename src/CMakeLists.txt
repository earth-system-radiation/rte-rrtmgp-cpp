#
# This file is part of a C++ interface to the Radiative Transfer for Energetics (RTE)
#
FILE(GLOB sourcefiles "../src/*.cpp")
FILE(GLOB sourcefiles_gpu "../src/*.cu")
include_directories("../include" SYSTEM ${INCLUDE_DIRS})

if(USECUDA)
  cuda_add_library(rte_rrtmgp STATIC ${sourcefiles} ${sourcefiles_gpu})
  target_link_libraries(rte_rrtmgp rte_rrtmgp_kernels rte_rrtmgp_kernels_cuda)
else()
  add_library(rte_rrtmgp STATIC ${sourcefiles})
  target_link_libraries(rte_rrtmgp rte_rrtmgp_kernels)
endif()
