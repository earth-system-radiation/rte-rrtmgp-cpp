set(ENV{CC}  gcc ) # C compiler for parallel build
set(ENV{CXX} g++) # C++ compiler for serial build

set(USER_CXX_FLAGS "-std=c++14 -fopenmp")
set(USER_CXX_FLAGS_RELEASE "-DNDEBUG -O3 -march=native")
add_definitions(-DRESTRICTKEYWORD=__restrict__)

set(USER_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wno-unknown-pragmas")

set(FFTW_LIB       "/opt/ohpc/pub/libs/gnu9/openmpi4/fftw/3.3.8/lib/libfftw3.so")
set(FFTWF_LIB      "/opt/ohpc/pub/libs/gnu9/openmpi4/fftw/3.3.8/lib/libfftw3f.so")
set(NETCDF_LIB_C   "/opt/ohpc/pub/libs/gnu9/openmpi4/netcdf/4.7.3/lib/libnetcdf.so")
set(NETCDF_INCLUDE "/opt/ohpc/pub/libs/gnu9/openmpi4/netcdf/4.7.3/include")
set(IRC_LIB        "irc")
set(IRC_LIB        "")
set(HDF5_LIB       "/opt/ohpc/pub/libs/gnu9/openmpi4/hdf5/1.10.6/lib/libhdf5.so")
set(SZIP_LIB       "")
set(BOOST_INCLUDE  "/opt/ohpc/pub/libs/gnu9/openmpi4/boost/1.73.0/include/")

#set(LIBS ${FFTW_LIB} ${FFTWF_LIB} ${NETCDF_LIB_C} ${HDF5_LIB} ${SZIP_LIB} ${IRC_LIB} m z curl)
set(LIBS ${FFTW_LIB} ${FFTWF_LIB} ${NETCDF_LIB_C} ${HDF5_LIB} ${SZIP_LIB} ${IRC_LIB}) 
set(INCLUDE_DIRS ${BOOST_INCLUDE} ${NETCDF_INCLUDE})

if(USECUDA)
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    set(LIBS ${LIBS} -rdynamic)
    set(USER_CUDA_NVCC_FLAGS "-arch=sm_80")
    list(APPEND CUDA_NVCC_FLAGS "-std=c++14")
    list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
endif()

add_definitions(-DRTE_USE_CBOOL)