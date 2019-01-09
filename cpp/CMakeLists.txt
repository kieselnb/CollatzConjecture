add_executable(CollatzConjectureCPP main.cpp
                                    collatz_counter_local.cpp
                                    collatz_counter_client.cpp
                                    collatz_runner_cpu.cpp
                                    collatz_runner_gpu.cu
                                    collatz_runner_boost.cpp
                                    collatz_server.cpp)

target_link_libraries(CollatzConjectureCPP ${CMAKE_THREAD_LIBS_INIT}
                                           ${Boost_LIBRARIES}
                                           ${OpenCL_LIBRARY})

