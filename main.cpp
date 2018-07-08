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
    vector<CollatzRunner*> runners;
    for (int i = 0; i < numProcs; i++) {
        runners.push_back(new CollatzRunnerCPU(collatzCounter));
    }

    // start all threads
    for (unsigned int i = 0; i < runners.size(); i++) {
        runners[i]->start();
    }

    while (1) {
        cout << "Current value of the counter is: " << collatzCounter.getCount()
            << endl;
        this_thread::sleep_for(chrono::seconds(10));
    }

    return 0;
}
