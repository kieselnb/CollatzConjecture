//
// Created by Nick on 9/24/2016.
//

#ifndef COLLATZCONJECTURE_SERVER_H
#define COLLATZCONJECTURE_SERVER_H

void handleClientRequest();

void handleClientResponse();

static void * serverThread(void * arg);

int serverInit(long portno);

void startServer(int* initStatus, int* threadStop, uint64_t* num, long num_threads, long portno);


#endif //COLLATZCONJECTURE_SERVER_H
