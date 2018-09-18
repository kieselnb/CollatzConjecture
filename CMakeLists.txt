cmake_minimum_required(VERSION 3.5)
project(CollatzConjecture C CXX)

option(ENABLE_CUDA "Enable CUDA GPU implementation of Collatz" OFF)

if(ENABLE_CUDA)
    enable_language(CUDA)
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(USE_LINTER "To lint or not to lint" OFF)

if(USE_LINTER)
    message("-- Looking for clang-tidy")
    find_program(CLANG_TIDY_EXECUTABLE clang-tidy)
    if(CLANG_TIDY_EXECUTABLE STREQUAL "CLANG_TIDY_EXECUTABLE-NOTFOUND")
        message("-- Looking for clang-tidy - not found")
    else()
        execute_process(COMMAND ${CLANG_TIDY_EXECUTABLE} -version
                        OUTPUT_VARIABLE VERSION_OUTPUT)
        separate_arguments(VERSION_OUTPUT_LIST UNIX_COMMAND "${VERSION_OUTPUT}")
        list(FIND VERSION_OUTPUT_LIST "version" VERSION_INDEX)
        math(EXPR VERSION_INDEX "${VERSION_INDEX} + 1")
        list(GET VERSION_OUTPUT_LIST ${VERSION_INDEX} CLANG_TIDY_VERSION)
        message("-- Looking for clang-tidy - found version: ${CLANG_TIDY_VERSION}")

        set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXECUTABLE};-format-style=google;-checks='-*,bugprone-*,clang-analyzer-*,modernize-*,performance-*,';-header-filter='.*,'")
    endif()
endif(USE_LINTER)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -Wall")

find_package(Boost REQUIRED program_options)

# find threading libraries
find_package(Threads)

# generate header file to pass in options
configure_file (
    "${PROJECT_SOURCE_DIR}/config.h.in"
    "${PROJECT_BINARY_DIR}/config.h"
)
include_directories("${PROJECT_BINARY_DIR}")

# add subdirs
add_subdirectory(c)

add_subdirectory(cpp)

add_subdirectory(cuda)

