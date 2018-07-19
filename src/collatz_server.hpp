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
        /**
         * Constructor
         *
         * @param[in] counter Reference to The One Counter Object for clients
         *                    to pull from.
         * @param[in] port Port number to serve from.
         */
        CollatzServer(CollatzCounter& counter, short port);

    private:
        /**
         * The One Counter Object to pull from.
         */
        CollatzCounter& _counter;

        /**
         * Port number to serve from.
         */
        short _port;

        /**
         * File descriptor of the listening port.
         */
        int _fd;

        /**
         * Status on things in the constructor - make sure we have a valid
         * file descriptor before calling more things on it.
         */
        bool _initialized;

        /**
         * List of all the clients connected to me.
         */
        std::list<int> _connections;

        /**
         * Thread that handles accepting new clients.
         */
        std::thread _acceptor;

        /**
         * Thread that serves the connected clients.
         */
        std::thread _poller;

        /**
         * Accepts new clients for the server.
         */
        static void acceptor(int sockfd, std::list<int> &connections);

        /**
         * Loops through the connected clients and handles requests.
         */
        static void poller(CollatzServer *server);

};

#endif  /* COLLATZ_SERVER_HPP */
