cmake_minimum_required(VERSION 3.5)
project(CollatzConjecture)

# set debug
set(CMAKE_BUILD_TYPE Debug)

# find threading libraries
find_package(Threads)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")

set(Collatz_SERVER_CLIENT "Collatz_SERVER" CACHE STRING "Collatz_SERVER_CLIENT chosen by the user at CMake configure time")
set_property(CACHE Collatz_SERVER_CLIENT PROPERTY STRINGS Collatz_SERVER Collatz_CLIENT)

# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
        "${PROJECT_SOURCE_DIR}/CollatzConfig.h.in"
        "${PROJECT_BINARY_DIR}/CollatzConfig.h"
)

# add the binary tree to the search path for include files
# so that we will find CollatzConfig.h
include_directories("${PROJECT_BINARY_DIR}")

set(SOURCE_FILES collatz.c collatz.h main.c server.c server.h client.c client.h types.h plugin_tcp.c plugin_tcp.h)
add_executable(CollatzConjecture ${SOURCE_FILES})

set(Collatz_CUDA ON CACHE STRING "To build or not to build the CUDA executable")
set_property(CACHE Collatz_CUDA PROPERTY TYPE BOOL)

if (Collatz_CUDA)
# find CUDA
find_package(CUDA)

# add cuda target
cuda_add_executable(CollatzConjectureCUDA cuda/collatz.cu)
endif(Collatz_CUDA)

target_link_libraries(CollatzConjecture ${CMAKE_THREAD_LIBS_INIT})

