set(SOURCES main.cpp
            collatz_counter_local.cpp
            collatz_counter_client.cpp
            collatz_runner_cpu.cpp
            collatz_server.cpp)

if(ENABLE_CUDA)
    set(SOURCES ${SOURCES} collatz_runner_gpu.cu)
endif()

if(ENABLE_OPENCL)
    set(SOURCES ${SOURCES} collatz_runner_boost.cpp)
endif()

add_executable(CollatzConjectureCPP ${SOURCES})

target_link_libraries(CollatzConjectureCPP ${CMAKE_THREAD_LIBS_INIT}
                                           ${OpenCL_LIBRARIES})

