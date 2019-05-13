if (FLOAT_SINGLE)
    message(STATUS "Compiling RRTMGP kernels in single precision")
    FILE(GLOB sourcefiles "../src/*.cpp")
    include_directories("../include" SYSTEM ${INCLUDE_DIRS})
    add_library(rrtmgp STATIC ${sourcefiles})
    set_target_properties(
        rrtmgp
        PROPERTIES
        COMPILE_DEFINITIONS REAL_TYPE=sp)
else()
    message(STATUS "Compiling RRTMGP kernels in double precision")
    FILE(GLOB sourcefiles "../src/*.cpp")
    include_directories("../include" SYSTEM ${INCLUDE_DIRS})
    add_library(rrtmgp STATIC ${sourcefiles})
    set_target_properties(
        rrtmgp
        PROPERTIES
        COMPILE_DEFINITIONS REAL_TYPE=dp)
endif()
