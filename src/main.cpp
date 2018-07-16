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
#include <boost/program_options.hpp>

#include "collatz_counter.hpp"
#include "collatz_runner.hpp"
#include "collatz_runner_cpu.hpp"

using namespace std;
namespace po = boost::program_options;

int main(int argc, char* argv[]) {
    // parse options
    po::options_description desc("Options");
    desc.add_options()
        ("help,h", "Display this message")
        ("cpu", po::value<int>(), "Number of cpu threads to use")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    // check if the user needs help. help the best way we know how
    if (vm.count("help")) {
        cout << desc << endl;
        return 1;
    }

    int numProcs = thread::hardware_concurrency();
    if (vm.count("cpu")) {
        numProcs = vm["cpu"].as<int>();
    }
    cout << "Using " << numProcs << " local CPU compute threads." << endl;

    // shared counter and counter protector
    CollatzCounter collatzCounter;

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
        // have some logic to check performance of the threads
        cout << "Current value of the counter is: " << collatzCounter.getCount()
            << endl;
        this_thread::sleep_for(chrono::seconds(10));
    }

    return 0;
}
