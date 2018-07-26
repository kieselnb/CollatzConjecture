/**
 * @file collatz_counter_client.cpp
 *
 * This file contains the definition of the CollatzCounterClient class.
 */

#include <iostream>
#include <cstring>

#include <sys/socket.h>
#include <arpa/inet.h>

#include "collatz_counter_client.hpp"

using namespace std;

CollatzCounterClient::CollatzCounterClient(const std::string &serverIp,
        short serverPort) {
    struct sockaddr_in serverAddress;
    
    _fd = socket(AF_INET, SOCK_STREAM, 0);
    if (_fd < 0) {
        cerr << "Error creating socket" << endl;
        return;
    }

    bzero(&serverAddress, sizeof(serverAddress));

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
