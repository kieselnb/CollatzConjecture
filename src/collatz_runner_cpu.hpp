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
         * @param[in] counter Reference to the counter to increment.
         */
        CollatzRunnerCPU(CollatzCounter &counter);

        /**
         * Implementation of CollatzRunner::start
         */
        void start() override;

        /**
         * Implementation of CollatzRunner::join
         */
        void join() override;

    private:
        /**
         * Runs the Collatz Conjecture on the given number.
         *
         * @param[in] number Number to check against the conjecture.
         *
         * @return Returns the ending value of the conjecture, should be 1.
         */
        static int collatz(uint64_t number);

        /**
         * Function to pass to thread creator.
         * Takes a group of numbers from the counter and checks them.
         */
        static void runner(CollatzCounter &counter);

};

#endif  /* COLLATZ_RUNNER_CPU_HPP */
