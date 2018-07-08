/**
 * @file main.cpp
 *
 * This file contains the main definition for the Collatz Conjecture
 * project
 */

#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <chrono>

#include "collatz_counter.hpp"
#include "collatz_runner.hpp"
#include "collatz_runner_cpu.hpp"

using namespace std;

int main(int argc, char* argv[]) {
    // shared counter and counter protector
    CollatzCounter collatzCounter;

    // get number of cores
    int numProcs = thread::hardware_concurrency();
    cout << "I have " << numProcs << " threads available." << endl;

    // kick off runners for each core
    vector<CollatzRunner*> runners(numProcs);
    for (int i = 0; i < numProcs; i++) {
        runners[i] = new CollatzRunnerCPU(collatzCounter);
    }

    // start all threads
    for (auto & runner : runners) {
        runner->start();
    }

    while (true) {
        cout << "Current value of the counter is: " << collatzCounter.getCount()
            << endl;
        this_thread::sleep_for(chrono::seconds(10));
    }

    return 0;
}
