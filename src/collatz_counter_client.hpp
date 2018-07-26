/**
 * @file collatz_counter_client.hpp
 *
 * This file contains the declaration of the CollatzCounterClient class.
 */

#ifndef COLLATZ_COUNTER_CLIENT_HPP
#define COLLATZ_COUNTER_CLIENT_HPP

#include "collatz_counter.hpp"

/**
 * This class implements the CollatzCounter interface in a way that pings the
 * server to get the results.
 */
class CollatzCounterClient : public CollatzCounter {
    public:

        /**
         * Constructor.
         *
         * @param[in] serverIp IP address of the server.
         * @param[in] serverPort Port of the server.
         */
        CollatzCounterClient(const std::string &serverIp, short serverPort);

    private:
        /**
         * File descriptor of the serving port.
         */
        int _fd;

        /**
         * Boolean to say whether the connect call was successful.
         */
        bool _initialized;

};

#endif  /* COLLATZ_COUNTER_CLIENT_HPP */
