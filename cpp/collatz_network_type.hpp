/**
 * @file collatz_network_type.hpp
 *
 * This file contains the declaration of the CollatzNetworkType object.
 */

#ifndef COLLATZ_NETWORK_TYPE_HPP
#define COLLATZ_NETWORK_TYPE_HPP

typedef enum class CollatzOperation {
    TAKE,
    GET_COUNT,
} CollatzOperation;

typedef struct CollatzNetworkType {
    CollatzOperation operation;
    uint64_t collatzNumber;
    uint32_t stride;
} CollatzNetworkType;

#endif  /* COLLATZ_NETWORK_TYPE_HPP */
