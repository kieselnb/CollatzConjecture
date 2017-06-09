cmake_minimum_required(VERSION 3.6)
project(CollatzConjecture)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(Collatz_SERVER_CLIENT Collatz_SERVER)

# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
        "${PROJECT_SOURCE_DIR}/CollatzConfig.h.in"
        "${PROJECT_BINARY_DIR}/CollatzConfig.h"
)

# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
include_directories("${PROJECT_BINARY_DIR}")

set(SOURCE_FILES collatz.c collatz.h main.c server.c server.h client.c client.h types.h plugin_tcp.c plugin_tcp.h)
add_executable(CollatzConjecture ${SOURCE_FILES})