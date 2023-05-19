# GCC compiler (if USECUDA is on, build on GPU):
# module purge
# module load eb #(Use the new software development and installation framework EasyBuild currently implemented by SURFsara)
# module load surfsara
# module load CMake/3.9.5-GCCcore-6.4.0 #(Loads GCCcore as well)
# module load cuDNN/7.0.5-CUDA-9.0.176 #(Loads CUDA as well,cuDNN needed for Tensorflow-gpu)
# module load netCDF/4.5.0-foss-2017b #(Loads as well HDF5,cURL,sZIP,openMPI,FFTW3,GCC)
# module load netCDF-C++4/4.3.0-foss-2017b
# module load Doxygen/1.8.13-GCCcore-6.4.0
# module unload ScaLAPACK/2.0.2-gompi-2017b-OpenBLAS-0.2.20 #(Prevent crash during compiling: "(..)/microhh/src/../src/tools.cu([number]): (..) identifier [name] is undefined")

set(ENV{CC}  gcc ) # C compiler for parallel build
set(ENV{CXX} g++) # C++ compiler for serial build

set(USER_CXX_FLAGS "-std=c++14 -fopenmp")
set(USER_CXX_FLAGS_RELEASE "-DNDEBUG -O3 -march=native")
add_definitions(-DRESTRICTKEYWORD=__restrict__)

set(USER_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wno-unknown-pragmas")

set(FFTW_LIB       "fftw3")
set(FFTWF_LIB      "fftw3f")
set(NETCDF_LIB_C   "/cm/shared/apps/netcdf/gcc/64/4.4.0/lib/libnetcdf.so")
set(NETCDF_INCLUDE "/cm/shared/apps/netcdf/gcc/64/4.4.0/include")
set(IRC_LIB        "irc")
set(IRC_LIB        "")
set(HDF5_LIB       "/cm/shared/apps/hdf5_18/1.8.18/lib/libhdf5.so")
set(SZIP_LIB       "sz")

set(LIBS ${FFTW_LIB} ${FFTWF_LIB} ${NETCDF_LIB_C} ${HDF5_LIB} ${SZIP_LIB} ${IRC_LIB} m z curl)
set(INCLUDE_DIRS ${NETCDF_INCLUDE})

if(USECUDA)
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    set(LIBS ${LIBS} -rdynamic)
    set(USER_CUDA_NVCC_FLAGS "-arch=sm_35")
    list(APPEND CUDA_NVCC_FLAGS "-std=c++14")
    list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
endif()

add_definitions(-DRTE_USE_CBOOL)
