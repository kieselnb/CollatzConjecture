cmake_minimum_required(VERSION 3.5)
project(CollatzConjecture)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -Wall")

find_package(Boost REQUIRED program_options)

# find threading libraries
find_package(Threads)

set(SOURCE_FILES src/collatz.c
                 src/main.c
                 src/server.c
                 src/client.c
                 src/plugin_tcp.c)
add_executable(CollatzConjecture ${SOURCE_FILES})

target_link_libraries(CollatzConjecture ${CMAKE_THREAD_LIBS_INIT})

# add subdirs
add_subdirectory(cuda)

add_executable(CollatzConjectureCPP src/main.cpp
                                    src/collatz_counter_local.cpp
                                    src/collatz_counter_client.cpp
                                    src/collatz_runner_cpu.cpp
                                    src/collatz_server.cpp)

target_link_libraries(CollatzConjectureCPP ${CMAKE_THREAD_LIBS_INIT}
                                           ${Boost_LIBRARIES})
