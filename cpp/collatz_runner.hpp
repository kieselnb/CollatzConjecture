/**
 * @file collatz_runner.hpp
 *
 * This file contains the declaration of the CollatzRunner class.
 */

#ifndef COLLATZ_RUNNER_HPP
#define COLLATZ_RUNNER_HPP

#include <thread>
#include <mutex>

#include "collatz_counter.hpp"

/**
 * Virtual class that starts a thread of the Collatz Conjecture.
 * 
 * This class starts a new thread of the Collatz Conjecture algorithm and
 * increments the given counter as it goes.
 */
class CollatzRunner {
    public:
        /**
         * Constructor.
         *
         * Mandate that at least one constructor must take in a reference
         * to the counter.
         *
         * @param[in] counter Reference to the 'global' counter
         */
        CollatzRunner(CollatzCounter &counter)
            : _counter(counter)
        {}

        /**
         * Launches the thread.
         */
        virtual void start() = 0;

        /**
         * Expose thread::join
         */
        virtual void join() = 0;

    protected:
        /**
         * Pointer to the counter object.
         */
        CollatzCounter& _counter;

        /**
         * Number of numbers to take at a time;
         */
        unsigned int _stride;
        
        /**
         * Pointer to the created thread.
         */
        std::thread* _collatzThread;

        /**
         * Pointer to the health monitor/performance thread.
         */
        std::thread* _collatzPerfHMThread;

};

#endif  /* COLLATZ_RUNNER_HPP */
