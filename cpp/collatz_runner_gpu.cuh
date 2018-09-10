/**
 * @file collatz_runner_gpu.cuh
 *
 * This file contains the declaration of the CollatzRunnerGPU class.
 */

#ifndef COLLATZ_RUNNER_GPU_H
#define COLLATZ_RUNNER_GPU_H

#include "collatz_runner.hpp"
#include "collatz_counter.hpp"

/**
 * Concrete implementation of the CollatzRunner class on a GPU.
 *
 * Uses the CUDA programming API to run the Collatz algorithm on NVIDIA GPUs.
 */
class CollatzRunnerGPU : public CollatzRunner {
    public:
        /**
         * Constructor
         * 
         * @param[in] counter Reference to the counter to increment.
         */
        CollatzRunnerGPU(CollatzCounter &counter);

        /**
         * Implementation of CollatzRunner::start
         */
        void start() override;
        
        /**
         * Implementation of CollatzRunner::join
         */
        void join() override;

    private:
        static void runner(CollatzRunnerGPU& self);

};
        
#endif  /* COLLATZ_RUNNER_GPU_H */
