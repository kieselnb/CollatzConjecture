//
// Created by Nick on 9/24/2016.
//

#include <sys/socket.h>
#include <stdio.h>
#include <string.h>
#include <netinet/in.h>

#include "server.h"
#include "types.h"
#include "plugin_tcp.h"
#include "collatz.h"

uint16_t portno = 8046;

void handleClientRequest()
{

}

void handleClientResponse()
{

}

static void * serverThread(void * arg)
{

    return 0;
}

void startServer(int* initStatus, int* threadStop, uint64_t* num, int num_threads) {
    collatzInit(initStatus, &takeNextNum);
    if (initStatus) {
        collatzStart(threadStop, num, num_threads);
    }
}

int serverInit() {
    int sockfd, newsockfd;
    socklen_t clilen;
    char buffer[256];
    struct sockaddr_in serv_addr, cli_addr;
    int n;

    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        perror("ERROR opening socket");
        return 1;
    }
    memset(&serv_addr, 0, sizeof(serv_addr));

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(portno);

    if (bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
        perror("ERROR on binding");
        return 2;
    }
    listen(sockfd,5);

    clilen = sizeof(cli_addr);
    newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
    if (newsockfd < 0) {
        perror("ERROR on accept");
        return 3;
    }

    return 0;
}
