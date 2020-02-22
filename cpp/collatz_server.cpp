/**
 * @file collatz_server.cpp
 *
 * This file contains the definition of the CollatzServer class.
 */

#include "collatz_server.hpp"
#include "collatz_counter.hpp"
#include "collatz_network_type.hpp"

#include <iostream>
#include <thread>
#include <cerrno>
#include <cstring>

#ifdef WIN32
#include <WinSock2.h>
#include <WS2tcpip.h>
#define errno WSAGetLastError()
#else
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

#include <sys/types.h>
#include <fcntl.h>


using namespace std;

CollatzServer::CollatzServer(CollatzCounter& counter, unsigned short port)
    : _counter(counter)
    , _port(port)
    , _initialized(false)
{
    int result;
    struct sockaddr_in address;

#ifdef WIN32
	// init the winsock2 dll
	WSADATA wsaData;
	int iResult;
	iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
	if (iResult != 0) {
		std::cerr << "WSAStartup failed: " << iResult << std::endl;
	}
#endif

    // create file descriptor
    _fd = socket(AF_INET, SOCK_STREAM, 0);
#ifdef WIN32
	if (_fd == INVALID_SOCKET) {
#else
    if (_fd == 0) {
#endif
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

}

CollatzServer::~CollatzServer() {
    // close all connections
#ifdef WIN32
	closesocket(_fd);
#else
    close(_fd);
#endif
}

void CollatzServer::run() {
    // start acceptor thread
    _acceptor = thread(acceptor, _fd, this);
}

void CollatzServer::connectionHandler(CollatzServer *server, int fd) {
    while (true) {
        int result;
        CollatzNetworkType request;
        result = recv(fd, (char*)(&request), sizeof(request), 0);
        if (result <= 0) {
			if (result < 0) {
				// got some error, report it and quit
				cout << "recv failed for " << fd << ": " << strerror(errno)
					<< endl;
			}

			// socket either errored or closed, so clean it up
#ifdef WIN32
			closesocket(fd);
#else
            close(fd);
#endif
            return;
        }
        else {
            // serves up
            switch (request.operation) {
                case CollatzOperation::TAKE:
                    request.collatzNumber =
                        server->_counter.take(request.stride);
                    break;
                case CollatzOperation::GET_COUNT:
                    request.collatzNumber = server->_counter.getCount();
                    break;
                default:
                    cout << "Error: I don't know how to serve this request"
                        << endl;
                    break;
            }

            result = send(fd, (char*)(&request), sizeof(request), 0);
            if (result < 0) {
                cout << "Error: could not send response" << endl;
            }
        }
    }
}

void CollatzServer::acceptor(int sockfd, CollatzServer* server) {
    while (true) {
        struct sockaddr_in clientAddr;
        socklen_t clientLen = sizeof(clientAddr);
        cout << "I'm listening..." << endl;
        int newsockfd = accept(sockfd, (struct sockaddr *)&clientAddr,
                &clientLen);
#ifdef WIN32
		if (newsockfd == INVALID_SOCKET) {
#else
        if (newsockfd < 0) {
#endif
            cerr << "Error on accept" << endl;
            continue;
        }

        char ipAddress[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, (void*)&(clientAddr.sin_addr), ipAddress,
                INET_ADDRSTRLEN);
        unsigned short port = ntohs(clientAddr.sin_port);
        cout << "Received connection from " << ipAddress << ":" << port << endl;

        thread newConnThread(CollatzServer::connectionHandler, server,
                newsockfd);
        newConnThread.detach();
    }
}

