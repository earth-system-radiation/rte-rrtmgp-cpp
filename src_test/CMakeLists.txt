# 
#  MicroHH
#  Copyright (c) 2011-2019 Chiel van Heerwaarden
#  Copyright (c) 2011-2019 Thijs Heus
#  Copyright (c) 2014-2019 Bart van Stratum
# 
#  This file is part of MicroHH
# 
#  MicroHH is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
# 
#  MicroHH is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
# 
#  You should have received a copy of the GNU General Public License
#  along with MicroHH.  If not, see <http://www.gnu.org/licenses/>.
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
  cuda_add_executable(test_rrtmgp test_rrtmgp.cpp)
  target_link_libraries(test_rrtmgp rrtmgp ${LIBS} m)

  cuda_add_executable(test_rcemip test_rcemip.cpp)
  target_link_libraries(test_rcemip rrtmgp ${LIBS} m)
else()
  add_executable(test_rrtmgp test_rrtmgp.cpp)
  target_link_libraries(test_rrtmgp rrtmgp ${LIBS} m)

  add_executable(test_rcemip test_rcemip.cpp)
  target_link_libraries(test_rcemip rrtmgp ${LIBS} m)
endif()
