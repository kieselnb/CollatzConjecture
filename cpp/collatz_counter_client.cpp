/**
 * @file collatz_counter_client.cpp
 *
 * This file contains the definition of the CollatzCounterClient class.
 */

#include <iostream>
#include <cstring>

#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

#include "collatz_counter_client.hpp"
#include "collatz_network_type.hpp"

using namespace std;

CollatzCounterClient::CollatzCounterClient(const std::string &serverIp,
        unsigned short serverPort) {
    struct sockaddr_in serverAddress;
    
    _fd = socket(AF_INET, SOCK_STREAM, 0);
    if (_fd < 0) {
        cerr << "Error creating socket" << endl;
        return;
    }

    memset(&serverAddress, 0, sizeof(serverAddress));

    serverAddress.sin_family = AF_INET;
    serverAddress.sin_port = htons(serverPort);

    if (inet_pton(AF_INET, serverIp.c_str(), &(serverAddress.sin_addr)) <= 0) {
        cerr << "Error: address " << serverIp << " not valid or not supported."
            << endl;
        return;
    }

    int result = connect(_fd, (const sockaddr*)&serverAddress,
            sizeof(serverAddress));
    if (result < 0) {
        cerr << "Error on connect" << endl;
        return;
    }

    _initialized = true;
}

CollatzCounterClient::~CollatzCounterClient() {
    close(_fd);
}

uint64_t CollatzCounterClient::take(unsigned int size) {
    CollatzNetworkType request;
    request.operation = CollatzOperation::TAKE;
    request.stride = size;

    uint64_t toReturn = 0;

    int result = send(_fd, &request, sizeof(request), 0);
    if (result < 0) {
        cout << "Error: failed to send" << endl;
    }
    else {
        // get result. we'll just use the same buffer since we already have it
        result = recv(_fd, &request, sizeof(request), 0);
        if (result < 0) {
            cout << "Error: failed to receive" << endl;
        }
        else {
            toReturn = request.collatzNumber;
        }
    }

    return toReturn;
}

uint64_t CollatzCounterClient::getCount() {
    CollatzNetworkType request;
    request.operation = CollatzOperation::GET_COUNT;

    uint64_t toReturn = 0;

    int result = send(_fd, &request, sizeof(request), 0);
    if (result < 0) {
        cout << "Error: failed to send request" << endl;
    }
    else {
        result = recv(_fd, &request, sizeof(request), 0);
        if (result < 0) {
            cout << "Error: failed to receive response" << endl;
        }
        else {
            toReturn = request.collatzNumber;
        }
    }

    return toReturn;
}

