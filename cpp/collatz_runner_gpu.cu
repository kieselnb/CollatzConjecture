/**
 * @file collatz_runner_gpu.cpp
 *
 * This file contains the definition of the CollatzRunnerGPU class
 */

#include <thread>
#include <iostream>

#include <cuda.h>

#include "collatz_runner_gpu.cuh"
#include "collatz_counter.hpp"

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

using namespace std;

CollatzRunnerGPU::CollatzRunnerGPU(CollatzCounter &counter)
    : CollatzRunner(counter)
{

}

void CollatzRunnerGPU::start() {
    collatzThread = new thread(runner, ref(*this));
}

void CollatzRunnerGPU::join() {
    collatzThread->join();
}

void CollatzRunnerGPU::runner(CollatzRunnerGPU& self) {
    self._stride = 1 << 21;

    int status, *d_status;
    cudaError_t err = cudaMalloc(&d_status, sizeof(int));
    if (err != cudaSuccess) {
        cout << "cudaMalloc failed, did you forget optirun?" << endl;
        return;
    }

    while (true) {
        status = 1;
        cudaMemcpy(d_status, &status, sizeof(int), cudaMemcpyHostToDevice);

        uint64_t start = self._counter.take(self._stride);
        collatz<<<(self._stride+255)/256, 256>>>(start, self._stride, d_status);

        cudaMemcpy(&status, d_status, sizeof(int), cudaMemcpyDeviceToHost);
        if (status == 0) {
            cout << "WE BROKE SOMETHING" << endl;
        }
    }
}
