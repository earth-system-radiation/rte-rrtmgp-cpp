#
# This file is part of a C++ interface to the Radiative Transfer for Energetics (RTE)
# and Rapid Radiative Transfer Model for GCM applications Parallel (RRTMGP).
#
# The original code is found at https://github.com/RobertPincus/rte-rrtmgp.
#
# Contacts: Robert Pincus and Eli Mlawer
# email: rrtmgp@aer.com
#
# Copyright 2015-2020,  Atmospheric and Environmental Research and
# Regents of the University of Colorado.  All right reserved.
#
# This C++ interface can be downloaded from https://github.com/microhh/rte-rrtmgp-cpp
#
# Contact: Chiel van Heerwaarden
# email: chiel.vanheerwaarden@wur.nl
#
# Copyright 2020, Wageningen University & Research.
#
# Use and duplication is permitted under the terms of the
# BSD 3-clause license, see http://opensource.org/licenses/BSD-3-Clause
#

include_directories(${INCLUDE_DIRS} "../include" "../include_test")

# retrieve the git hash from the current commit
find_package(Git)
if(GIT_EXECUTABLE)
  exec_program(git ${CMAKE_CURRENT_SOURCE_DIR} ARGS "describe --dirty" OUTPUT_VARIABLE GITHASH RETURN_VALUE GITRETURN)
  if(NOT GITRETURN MATCHES 0)
    set(GITHASH "not available")
  endif()
else()
  set(GITHASH "not available")
endif()

message(STATUS "Git hash " ${GITHASH})

# send a precompiler statement replacing the git hash
add_definitions(-DGITHASH="${GITHASH}")

if(USECUDA)
  cuda_add_executable(test_rte_rrtmgp Radiation_solver.cpp test_rte_rrtmgp.cpp)
  target_link_libraries(test_rte_rrtmgp rte_rrtmgp ${LIBS} m)
else()
  add_executable(test_rte_rrtmgp Radiation_solver.cpp test_rte_rrtmgp.cpp)
  target_link_libraries(test_rte_rrtmgp rte_rrtmgp ${LIBS} m)
endif()
