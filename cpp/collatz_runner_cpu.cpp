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
    heartBeatMtx = new mutex();
}

void CollatzRunnerCPU::start()
{
    _collatzThread = new thread(runner, ref(*this));
    //monitorThread = new thread(monitor, ref(*this));
}

void CollatzRunnerCPU::join()
{
    _collatzThread->join();
}

int CollatzRunnerCPU::collatz(uint64_t number)
{
    while (number > 1) {
        if (number % 2 == 0) {
            // even, divide by two
            number >>= 1;
        }
        else {
            // odd, multiply by 3, add 1, div 2 again
            number = (3 * number + 1) >> 1;
        }
    }

    return number;
}

void CollatzRunnerCPU::runner(CollatzRunnerCPU& self)
{
    // this will be the function called by thread creator
    self._stride = 1<<20;

    // perform collatz on said group of numbers
    while (true) {
        //self.beat();
        uint64_t start = self._counter.take(self._stride);
        int result = 0;
        for (unsigned int i = 0; i < self._stride; i++) {
            result = collatz(start + i);
            if (result != 1) {
                cout << "WE BROKE SOMETHING" << endl;
            }
        }
    }
}

void CollatzRunnerCPU::monitor(CollatzRunnerCPU& self)
{
    // not sure what's gonna happen here yet
    // make sure we get "heartbeats" from the runner periodically
    // adjust the stride based off some sort of logic
    int maxMissedBeats = 6;
    int missedBeats = 0;
    int lastChangeAmount;
    while (true) {
        // check if there was a heartbeat
        if (self.isAlive()) {
            // logic to adjust stride
        }
        else if (++missedBeats >= maxMissedBeats) {
            cout << "WARNING: missing thread heartbeat" << endl;
        }

        this_thread::sleep_until(chrono::system_clock::now() +
                chrono::milliseconds(500));
    }
}

bool CollatzRunnerCPU::isAlive() {
    lock_guard<mutex> l(*heartBeatMtx);
    bool result = heartBeat;
    heartBeat = false;
    return result;
}

void CollatzRunnerCPU::beat() {
    lock_guard<mutex> l(*heartBeatMtx);
    heartBeat = true;
}
