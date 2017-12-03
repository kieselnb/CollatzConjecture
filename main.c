//
// Created by Nick on 9/24/2016.
//

/* INCLUDES */

#include <stdio.h>
#include <unistd.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include "server.h"
#include "client.h"

/* DEFINES */

/* FUNCTION DEFINITIONS */

enum ExecMode {
    EXEC_SERVER,
    EXEC_CLIENT,
    NUM_EXEC_MODES
};

int threadStop = 0;
int collatzInitialized = 0;
uint64_t currentNum = 1;

void usage() {
    printf("Usage: CollatzConjecture <type> <[host:]port> [--num-threads <n>]\n");
    printf("   <type> can be either '--server' or '--client'\n");
    printf("      Server:\n");
    printf("         Specifying 'server' designates this thread as the server for this run\n");
    printf("         The server must be given a port number to listen on\n");
    printf("         This process will also spawn client threads for each processor on the system\n");
    printf("      Client:\n");
    printf("         The client must be given a host and a port at which a server process is listening\n");
    printf("\n");
    printf("   --num-threads <n>\n");
    printf("      Use n number of threads. Defaults to number of online cores on system\n");
}

int main(int argc, char *argv[]) {

    int exec_mode = NUM_EXEC_MODES;
    long server_port = -1;
    long num_threads = -1;

    printf("Parsing command line args\n");

    int argi;
    int skip = 1;
    for (argi = 0; argi < argc; ++argi) {
        if (skip > 0) {
            argi += (--skip);
            skip = 0;
            continue;
        }

        if (strcmp(argv[argi], "--server") == 0) {
            skip = 1;
            if (exec_mode == NUM_EXEC_MODES) {
                exec_mode = EXEC_SERVER;
            }
            else {
                printf("Error: multiple execution modes specified, only one allowed.\n");
                exit(1);
            }

            if (argi+1 == argc) {
                printf("Error: --server must be followed by a port number.\n\n");
                usage();
                exit(1);
            }

            char **end = &argv[argi+1];
            long temp_server_port = strtol(argv[argi+1], end, 10);
            if ((temp_server_port != LONG_MIN) && (temp_server_port != LONG_MAX) && (**end == '\0')) {
                server_port = temp_server_port;
            }
            else {
                printf("Error: Invalid port given.\n");
                exit(1);
            }
        }

        else if (strcmp(argv[argi], "--client") == 0) {
            if (exec_mode == NUM_EXEC_MODES) {
                exec_mode = EXEC_CLIENT;
            }
            else {
                printf("Error: multiple execution modes specified, only one allowed.\n");
                exit(1);
            }
        }

        else if (strcmp(argv[argi], "--num-threads") == 0) {
            skip = 1;
            if (argi+1 == argc) {
                printf("Error: --num-threads must be followed by a number.\n\n");
                usage();
                exit(1);
            }

            if (num_threads == -1) {
                char **end = &argv[argi+1];
                long temp_num_threads = strtol(argv[argi+1], end, 10);
                if ((temp_num_threads != LONG_MAX) && (temp_num_threads != LONG_MIN) && (**end == '\0')) {
                    num_threads = temp_num_threads;
                }
                else {
                    printf("Error: Invalid number of threads given.\n");
                    exit(1);
                }
            }
            else {
                printf("Error: num-threads specified more than once, only one allowed.\n");
                exit(1);
            }
        }

        else {
            printf("Error: Unknown command line arg '%s'.\n\n", argv[argi]);
            usage();
            exit(1);
        }
    }

    long num_logical_cores = sysconf(_SC_NPROCESSORS_ONLN);
    if (num_threads == -1) {
        num_threads = num_logical_cores;
    }
    else if (num_threads > num_logical_cores) {
        printf("Warning: Using more threads than there are logical processors.\n");
    }
    printf("Using %lu threads\n", num_threads);

    switch (exec_mode) {
        case EXEC_SERVER:
            startServer(&collatzInitialized, &threadStop, &currentNum, num_threads);
            break;
        case EXEC_CLIENT:
            startClient();
            break;
        default:
            printf("Error: Unknown exec mode.\n\n");
            exit(1);
    }

    while (!threadStop && collatzInitialized) {
        printf("Current value is: %14" PRIu64 "\n", currentNum);
        sleep(10);
    }

    return 0;
}
