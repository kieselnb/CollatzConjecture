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
         * Destructor.
         */
        ~CollatzCounter();

        /**
         * Increments counter by the given amount.
         *
         * @param[in] size Number of elements to "take", aka how much to 
         * increment the counter by.
         */
        unsigned long long int take(int size);

        /**
         * Gets the current value of the counter.
         *
         * @return Returns the current value of counter
         */
        unsigned long long int getCount();

    private:
        /**
         * Counter to keep track of the current Collatz number.
         */
        unsigned long long int counter;

        /**
         * Mutex to protect the counter.
         */
        std::mutex *counterProtector;
};

#endif  /* COLLATZ_COUNTER_HPP */
