/**
 * @file collatz_counter_local.hpp
 *
 * This file contains the declaration of the CollatzCounterLocal class
 */

#ifndef COLLATZ_COUNTER_LOCAL_HPP
#define COLLATZ_COUNTER_LOCAL_HPP

#include "collatz_counter.hpp"

class CollatzCounterLocal : public CollatzCounter {
    public:
        /**
         * Constructor
         */
        CollatzCounterLocal();

        /**
         * Destructor
         */
        ~CollatzCounterLocal();

        uint64_t take(unsigned int size) override;

        uint64_t getCount() override;

};

#endif  /* COLLATZ_COUNTER_LOCAL_HPP */
