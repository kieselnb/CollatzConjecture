/**
 * @file collatz_server.hpp
 *
 * This file contains the declaration of the CollatzServer class.
 */

#ifndef COLLATZ_SERVER_HPP
#define COLLATZ_SERVER_HPP

#include <list>

#include "collatz_counter.hpp"

/**
 * This class instantiates a server on the local machine that can accept
 * clients that running the CollatzConjecture application. This provides an
 * interface of distributed computing and exposes the local CollatzCounter
 * to all subscribed clients.
 */
class CollatzServer {

    public:
        CollatzServer(CollatzCounter& counter, short port);
        void start();

    private:
        CollatzCounter& _counter;
        short _port;
        int _fd;
        bool _initialized;
        std::list<int> _connections;
        std::thread _acceptor;
        std::thread _poller;
        static void acceptor(int sockfd, std::list<int> &connections);
        static void poller(CollatzServer *server);
};

#endif  /* COLLATZ_SERVER_HPP */
