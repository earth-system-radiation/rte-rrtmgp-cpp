if(FLOAT_TYPE STREQUAL "single")
    message(STATUS "Compiling RRTMGP kernels in single precision")
    FILE(GLOB sourcefiles "../src_fortran/*.F90")
    add_library(rrtmgp_kernels STATIC ${sourcefiles})
    set_target_properties(
        rrtmgp_kernels
        PROPERTIES
        COMPILE_DEFINITIONS REAL_TYPE=sp)
elseif(FLOAT_TYPE STREQUAL "double")
    message(STATUS "Compiling RRTMGP kernels in double precision")
    FILE(GLOB sourcefiles "../src_fortran/*.F90")
    add_library(rrtmgp_kernels STATIC ${sourcefiles})
    set_target_properties(
        rrtmgp_kernels
        PROPERTIES
        COMPILE_DEFINITIONS REAL_TYPE=dp)
else()
    message(FATAL_ERROR "Illegal FLOAT_TYPE specified.")
endif()
