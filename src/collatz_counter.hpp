/**
 * @file collatz_counter.hpp
 *
 * This file contains the declaration of the CollatzCounter class.
 */

#ifndef COLLATZ_COUNTER_HPP
#define COLLATZ_COUNTER_HPP

#include <mutex>

/**
 * This class provides a wrapper around a counter and makes it thread-safe.
 */
class CollatzCounter {
    public:
        /**
         * Constructor
         */
        CollatzCounter();

        /**
         * Destructor
         */
        ~CollatzCounter();

        /**
         * Increments counter by the given amount.
         *
         * @param[in] size Number of elements to "take", aka how much to 
         * increment the counter by.
         *
         * @return Returns the value of counter before the increment.
         */
        uint64_t take(int size);

        /**
         * Gets the current value of the counter.
         *
         * @return Returns the current value of counter
         */
        uint64_t getCount();

    private:
        /**
         * Counter to keep track of the current Collatz number.
         */
        uint64_t counter {1};

        /**
         * Mutex to protect the counter.
         */
        std::mutex *counterProtector;
};

#endif  /* COLLATZ_COUNTER_HPP */