/**
 * @file collatz_server.cpp
 *
 * This file contains the definition of the CollatzServer class.
 */

#include <iostream>
#include <thread>

#include <sys/socket.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <unistd.h>

#include "collatz_server.hpp"
#include "collatz_counter.hpp"
#include "collatz_network_type.hpp"

using namespace std;

CollatzServer::CollatzServer(CollatzCounter& counter, unsigned short port)
    : _counter(counter)
    , _port(port)
    , _initialized(false)
{
    int result;
    struct sockaddr_in address;

    // create file descriptor
    _fd = socket(AF_INET, SOCK_STREAM, 0);
    if (_fd == 0) {
        cerr << "Socket creation failed" << endl;
        return;
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(_port);

    result = bind(_fd, (struct sockaddr *)&address, sizeof(address));
    if (result < 0) {
        cerr << "Error on binding" << endl;
        return;
    }

    // magic number 5 for the backlog for now
    result = listen(_fd, 5);
    if (result != 0) {
        cerr << "Error on listening" << endl;
        return;
    }

    // start acceptor thread
    _acceptor = thread(acceptor, _fd, ref(_connections));

    // for now, set all sockets to non-blocking and loop through them
    // periodically in a dedicated thread
    _poller = thread(poller, this);
}

CollatzServer::~CollatzServer() {
    // close all connections
    close(_fd);
    for (auto & connection : _connections) {
        close(connection);
    }
}

void CollatzServer::poller(CollatzServer *server) {
    while (true) {
        int result;
        CollatzNetworkType request;
        for (auto &connection : server->_connections) {
            // call recv on each fd to see if they need something
            result = recv(connection, &request, sizeof(request), 0);
            if (result < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    // this is fine
                }
                else {
                    cout << "Unexpected error: " << errno << endl;
                }
            }
            else {
                // serving time
                switch (request.operation) {
                    case TAKE:
                        request.collatzNumber =
                            server->_counter.take(request.stride);
                        break;
                    case GET_COUNT:
                        request.collatzNumber = server->_counter.getCount();
                        break;
                    default:
                        cout << "Error: I don't know how to serve this request"
                            << endl;
                        break;
                }

                result = send(connection, &request, sizeof(request), 0);
                if (result < 0) {
                    cout << "Error: could not send response" << endl;
                }
            }
        }
        this_thread::sleep_until(chrono::system_clock::now()
                + chrono::milliseconds(100));
    }
}

void CollatzServer::acceptor(int sockfd, std::list<int> &connections) {
    while (true) {
        struct sockaddr_in clientAddr;
        socklen_t clientLen = sizeof(clientAddr);
        cout << "I'm listening..." << endl;
        int newsockfd = accept(sockfd, (struct sockaddr *)&clientAddr,
                &clientLen);
        if (newsockfd < 0) {
            cerr << "Error on accept" << endl;
            continue;
        }

        char ipAddress[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, (void*)&(clientAddr.sin_addr), ipAddress,
                INET_ADDRSTRLEN);
        unsigned short port = ntohs(clientAddr.sin_port);
        cout << "Received connection from " << ipAddress << ":" << port << endl;

        // set new socket to be non-blocking
        if (fcntl(newsockfd, F_SETFD,
                    fcntl(newsockfd, F_GETFD) | O_NONBLOCK) == -1) {
            cerr << "ERROR: could not set socket to be non-blocking" << endl;
        }

        connections.push_back(newsockfd);
        cout << "Added a new connection" << endl;
    }
}

