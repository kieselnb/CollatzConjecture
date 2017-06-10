//
// Created by Nick on 9/24/2016.
//

/* INCLUDES */

#include "CollatzConfig.h"

#include <stdio.h>
#include <unistd.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

#include "server.h"
#include "client.h"

/* DEFINES */

/* FUNCTION DEFINITIONS */

int threadStop = 0;
int collatzInitialized = 0;
uint64_t currentNum = 1;

void usage() {
    printf("Usage: CollatzConjecture <type> <[host:]port>\n");
    printf("   <type> can be either '--server' or '--client'\n");
    printf("      Server:\n");
    printf("         Specifying 'server' designates this thread as the server for this run\n");
    printf("         The server must be given a port number to listen on\n");
    printf("         This process will also spawn client threads for each processor on the system\n");
    printf("      Client:\n");
    printf("         The client must be given a host and a port at which a server process is listening\n");
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Invalid number of arguments given\n");
        usage();
        exit(1);
    }

    printf("Parsing command line args\n");

    if (strcmp(argv[1], "--server") == 0) {
        startServer(&collatzInitialized, &threadStop, &currentNum);
    }
    else if (strcmp(argv[1], "--client") == 0) {
        startClient();
    }
    else {
        printf("Invalid arguments given\n");
        usage();
        exit(2);
    }

    // collatzInit(&collatzInitialized);

    // collatzStart(&threadStop, &currentNum);
//#if Collatz_SERVER_CLIENT == Collatz_SERVER
//    startServer(&collatzInitialized, &threadStop, &currentNum);
//#elif Collatz_SERVER_CLIENT == Collatz_CLIENT
//    startClient();
//#endif

    while (1)
    {
        if (threadStop || !collatzInitialized)
        {
            // One of the threads triggered a stop or the init failed. Exit

            return 1;
        } else
        {
            printf("Current value is: %" PRIu64 "\n", currentNum);
            sleep(10);
        }
    }
}
