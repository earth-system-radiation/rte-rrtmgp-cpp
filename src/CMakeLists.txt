FILE(GLOB sourcefiles "../src/*.cpp")
include_directories("../include" SYSTEM ${INCLUDE_DIRS})

if(USECUDA)
  cuda_add_library(rrtmgp STATIC ${sourcefiles})
else()
  add_library(rrtmgp STATIC ${sourcefiles})
endif()
