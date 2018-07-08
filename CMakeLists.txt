cmake_minimum_required(VERSION 3.5)
project(CollatzConjecture)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_CLANG_TIDY "clang-tidy;-format-style=google;-checks=-'*,bugprone-*,cppcoreguidelines-*,clang-analyzer-*,modernize-*,performance-*';-header-filter='.*'")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -Wall")

# find threading libraries
find_package(Threads)

set(SOURCE_FILES collatz.c main.c server.c client.c plugin_tcp.c)
add_executable(CollatzConjecture ${SOURCE_FILES})

# add subdirs
add_subdirectory(cuda)

target_link_libraries(CollatzConjecture ${CMAKE_THREAD_LIBS_INIT})

add_executable(CollatzConjectureCPP main.cpp
                                    collatz_counter.cpp
                                    collatz_runner_cpu.cpp)

target_link_libraries(CollatzConjectureCPP ${CMAKE_THREAD_LIBS_INIT})
