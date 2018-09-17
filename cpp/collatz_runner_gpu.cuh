/**
 * @file collatz_runner_gpu.cuh
 *
 * This file contains the declaration of the CollatzRunnerGPU class.
 */

#ifndef COLLATZ_RUNNER_GPU_CUH
#define COLLATZ_RUNNER_GPU_CUH

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
         * Destructor
         */
        ~CollatzRunnerGPU();

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
         * Thread to run the GPU collatz implementation.
         * Has an infinite loop - to be started in a new thread.
         * 
         * @param[in] self Running object to access things.
         */
        static void runner(CollatzRunnerGPU& self);

        /**
         * Pointer to GPU status variable, to be set to 0 if
         * something failed.
         *
         * Declared here in order to be RAII style.
         */
        int *_dStatus;

        /**
         * To see if we successfully allocated memory.
         */
        bool _initialized;

};

#endif  /* COLLATZ_RUNNER_GPU_CUH */
