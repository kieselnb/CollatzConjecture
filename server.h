//
// Created by Nick on 9/24/2016.
//

#ifndef COLLATZCONJECTURE_SERVER_H
#define COLLATZCONJECTURE_SERVER_H

// Use port 8046 on server for communication
extern uint16_t portno;

void handleClientRequest();

void handleClientResponse();

static void * serverThread(void * arg);

int serverInit();

void startServer(int* initStatus, int* threadStop, uint64_t* num, long num_threads);


#endif //COLLATZCONJECTURE_SERVER_H
