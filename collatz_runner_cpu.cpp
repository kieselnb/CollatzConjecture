/**
 * @file collatz_runner_cpu.cpp
 *
 * This file contains the definition of the CollatzRunnerCPU class
 */

#include <iostream>
#include <thread>

#include "collatz_counter.hpp"
#include "collatz_runner_cpu.hpp"

using namespace std;

CollatzRunnerCPU::CollatzRunnerCPU(CollatzCounter &counter)
    : CollatzRunner(counter)
{
}

void CollatzRunnerCPU::start()
{
    collatzThread = new thread(runner, ref(_counter));
}

void CollatzRunnerCPU::join()
{
    collatzThread->join();
}

int CollatzRunnerCPU::collatz(unsigned long long int number)
{
    while (number > 1) {
        if (number % 2 == 0) {
            // even, divide by two
            number /= 2;
        }
        else {
            // odd, multiply by 3, add 1, div 2 again
            number = (3 * number + 1) / 2;
        }
    }

    return number;
}

void CollatzRunnerCPU::runner(CollatzCounter &counter)
{
    // this will be the function called by thread creator
    int stride = 1000000;

    // perform collatz on said group of numbers
    while (1) {
        unsigned long long int start = counter.take(stride);
        int result = 0;
        for (int i = 0; i < stride; i++) {
            result = collatz(start + i);
            if (result != 1) {
                cout << "WE BROKE SOMETHING" << endl;
            }
        }
    }
}
