/**
 * @file collatz_runner_gpu.cpp
 *
 * This file contains the definition of the CollatzRunnerGPU class
 */

#include "collatz_runner_gpu.cuh"
#include "collatz_counter.hpp"

CollatzRunnerGPU::CollatzRunnerGPU(CollatzCounter &counter)
    : CollatzRunner(counter)
{

}

void CollatzRunnerGPU::start() {

}

void CollatzRunnerGPU::join() {

}

void CollatzRunnerGPU::runner(CollatzRunnerGPU& self) {

}

__global__
void collatz(uint64_t start, int stride, int *status) {
    int k = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (k < stride) {
        uint64_t myNum = start + k;

        while (myNum > 1) {
            if (myNum & 2 == 0) {
                myNum = myNum >> 1;
            }
            else {
                myNum = ((myNum * 3) + 1) >> 1;
            }
        }

        if (myNum != 1) {
            *status = 0;
        }
    }
}
