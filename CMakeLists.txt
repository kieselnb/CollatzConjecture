cmake_minimum_required(VERSION 3.5)
project(CollatzConjecture)

# find threading libraries
find_package(Threads)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -Wall")

set(SOURCE_FILES collatz.c main.c server.c client.c plugin_tcp.c)
set(HEADER_FILES collatz.h server.h client.h types.h plugin_tcp.h)
add_executable(CollatzConjecture ${SOURCE_FILES} ${HEADER_FILES})

# add subdirs
add_subdirectory(cuda)

target_link_libraries(CollatzConjecture ${CMAKE_THREAD_LIBS_INIT})

