/**
 * @file collatz_runner_gpu.cu
 *
 * This file contains the definition of the CollatzRunnerGPU class
 */

#include <thread>
#include <iostream>

#include <cuda.h>

#include "collatz_runner_gpu.cuh"
#include "collatz_counter.hpp"

using namespace std;

__global__
void collatz(uint64_t start, int stride, int *status) {
    int k = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (k < stride) {
        uint64_t myNum = start + k;

        while (myNum > 1) {
            if (myNum % 2 == 0) {
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

CollatzRunnerGPU::CollatzRunnerGPU(CollatzCounter &counter)
    : CollatzRunner(counter)
{
    _initialized = true;
    cudaError_t err = cudaMalloc(&_dStatus, sizeof(int));
    if (err != cudaSuccess) {
        _initialized = false;
        cout << "cudaMalloc failed, did you forget optirun?" << endl;
    }
}

CollatzRunnerGPU::~CollatzRunnerGPU() {
    if (_initialized) {
        cudaFree(_dStatus);
    }
}

void CollatzRunnerGPU::start() {
    _collatzThread = new thread(runner, ref(*this));
}

void CollatzRunnerGPU::join() {
    _collatzThread->join();
}

void CollatzRunnerGPU::runner(CollatzRunnerGPU& self) {
    self._stride = 1 << 21;
    int status;

    if (self._initialized) {
        while (true) {
            status = 1;
            cudaMemcpy(self._dStatus, &status, sizeof(int), cudaMemcpyHostToDevice);
    
            uint64_t start = self._counter.take(self._stride);
            collatz<<<(self._stride+255)/256, 256>>>(start, self._stride, self._dStatus);
    
            cudaMemcpy(&status, self._dStatus, sizeof(int), cudaMemcpyDeviceToHost);
            if (status == 0) {
                cout << "WE BROKE SOMETHING" << endl;
            }
        }
    }
    else {
        cout << "CollatzRunnerGPU: runner uninitialized, bailing out" << endl;
    }

    return;
}
