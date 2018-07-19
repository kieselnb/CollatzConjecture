/**
 * @file collatz_network_type.hpp
 *
 * This file contains the declaration of the CollatzNetworkType object.
 */

#ifndef COLLATZ_NETWORK_TYPE_HPP
#define COLLATZ_NETWORK_TYPE_HPP

typedef struct CollatzNetworkType {
    uint32_t clientId;
    uint64_t collatzNumber;
    uint32_t stride;
} CollatzNetworkType;

#endif  /* COLLATZ_NETWORK_TYPE_HPP */
