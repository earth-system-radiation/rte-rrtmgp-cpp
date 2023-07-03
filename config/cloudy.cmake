# Ubuntu 20.04
set(ENV{CC}  gcc) # C compiler for serial build
set(ENV{CXX} g++) # C++ compiler for serial build
set(ENV{FC}  gfortran) # Fortran compiler for serial build

if(USECUDA)
  set(USER_CXX_FLAGS "-std=c++17 -fopenmp")
else()
  set(USER_CXX_FLAGS "-std=c++17")
endif()

set(USER_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(USER_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wno-unknown-pragmas")
set(USER_FC_FLAGS "-fdefault-real-8 -fdefault-double-8 -fPIC -ffixed-line-length-none -fno-range-check")
set(USER_FC_FLAGS_RELEASE "-DNDEBUG -O3 -march=native")
set(USER_FC_FLAGS_DEBUG "-O0 -g -Wall -Wno-unknown-pragmas")

set(NETCDF_INCLUDE_DIR "/usr/include")
set(NETCDF_LIB_C       "/usr/lib/x86_64-linux-gnu/libnetcdf.so")
set(HDF5_LIB_1         "/usr/lib/x86_64-linux-gnu/libhdf5_serial.so")
set(HDF5_LIB_2         "/usr/lib/x86_64-linux-gnu/libhdf5_serial_hl.so")
set(SZIP_LIB           "")

if(USECUDA)
  set(CUDA_INCLUDE_DIRS "/opt/nvidia/hpc_sdk/Linux_x86_64/23.1/math_libs/include" "/opt/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/include")
  set(CURAND_LIBS       "/opt/nvidia/hpc_sdk/Linux_x86_64/23.1/math_libs/lib64/libcurand.so")
else()
  set(CUDA_INCLUDE_DIRS "")
  set(CURAND_LIBS       "")
endif()

set(LIBS ${NETCDF_LIB_C} ${HDF5_LIB_2} ${HDF5_LIB_1} ${SZIP_LIB} ${CURAND_LIBS} m z curl)
set(INCLUDE_DIRS ${FFTW_INCLUDE_DIR} ${NETCDF_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})

if(USECUDA)
  set(CMAKE_CUDA_ARCHITECTURES 86)
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
  set(USER_CUDA_NVCC_FLAGS "-std=c++17 -arch=sm_86 --expt-relaxed-constexpr")
  set(USER_CUDA_NVCC_FLAGS_RELEASE "-Xptxas -O3 -DNDEBUG")
  set(USER_CUDA_NVCC_FLAGS_DEBUG "-Xptxas -O0 -g -DCUDACHECKS")
  # set(LIBS ${LIBS} -rdynamic cufft)
  set(LIBS ${LIBS} -rdynamic /opt/nvidia/hpc_sdk/Linux_x86_64/23.1/math_libs/lib64/libcufft.so)
  add_definitions(-DRTE_RRTMGP_GPU_MEMPOOL_CUDA)
endif()

add_definitions(-DRESTRICTKEYWORD=__restrict__)
add_definitions(-DRTE_USE_CBOOL)
