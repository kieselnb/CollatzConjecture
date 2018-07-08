/**
 * @file collatz_runner_cpu.hpp
 *
 * This file contains the declaration of the CollatzRunnerCPU class.
 */

#ifndef COLLATZ_RUNNER_CPU_HPP
#define COLLATZ_RUNNER_CPU_HPP

#include "collatz_counter.hpp"
#include "collatz_runner.hpp"

/**
 * Concrete implementation of the Collatz Runner class on CPU.
 *
 * Uses a CPU thread to run the Collatz Conjecture algorithm.
 */
class CollatzRunnerCPU : public CollatzRunner {
    public:
        /**
         * Constructor.
         *
         * @param[in] counter Reference to the 'global' counter to increment.
         */
        CollatzRunnerCPU(CollatzCounter &counter);

        /**
         * Implementation of CollatzRunner::start
         */
        void start();

        /**
         * Implementation of CollatzRunner::join
         */
        void join();

    private:
        /**
         * Runs the Collatz Conjecture on the given number.
         *
         * @param[in] number Number to check against the conjecture.
         *
         * @return Returns the ending value of the conjecture, should be 1.
         */
        static int collatz(unsigned long long int number);

        /**
         * Function to pass to thread creator.
         * Takes a group of numbers from the counter and checks them.
         */
        static void runner(CollatzCounter &counter);

        /**
         * Pointer to the created thread.
         */
        std::thread* collatzThread;

};

#endif  /* COLLATZ_RUNNER_CPU_HPP */
