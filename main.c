//
// Created by Nick on 9/24/2016.
//

/* INCLUDES */

#include <stdio.h>
#include <unistd.h>
#include <inttypes.h>

#include "collatz.h"
#include "server.h"
#include "client.h"

/* DEFINES */

#define HOST

/* FUNCTION DEFINITIONS */

int threadStop = 0;
int collatzInitialized = 0;
uint64_t currentNum = 1;

int main() {
    // collatzInit(&collatzInitialized);

    // collatzStart(&threadStop, &currentNum);
#ifdef HOST
    startServer(&collatzInitialized, &threadStop, &currentNum);
#else
    startClient();
#endif

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
