add_executable(CollatzConjecture collatz.c
                                 main.c
                                 server.c
                                 client.c
                                 plugin_tcp.c)

target_link_libraries(CollatzConjecture ${CMAKE_THREAD_LIBS_INIT})

