/**
 * @file main.cpp
 *
 * This file contains the main definition for the Collatz Conjecture
 * project
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <boost/program_options.hpp>

#include "collatz_counter.hpp"
#include "collatz_runner.hpp"
#include "collatz_runner_cpu.hpp"
#include "collatz_server.hpp"

using namespace std;
namespace po = boost::program_options;

int main(int argc, char* argv[]) {
    // parse options
    po::options_description desc("Options");
    desc.add_options()
        ("help,h", "Display this message")
        ("cpu,c", po::value<int>(), "Number of cpu threads to use")
        ("server,s", po::value<short>(),
            "Start a CollatzServer on this machine")
    ;

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
    } catch (exception& e) {
        cerr << "Error parsing options: " << e.what() << endl << endl;
        cerr << desc << endl;
        return 1;
    }
    po::notify(vm);

    // check if the user needs help. help the best way we know how
    if (vm.count("help")) {
        cout << desc << endl;
        return 1;
    }

    // get number of available threads - use either that or the
    // user-specified number of threads
    int numProcs = thread::hardware_concurrency();
    if (vm.count("cpu")) {
        int desiredNumProcs = vm["cpu"].as<int>();
        if (desiredNumProcs > numProcs) {
            cout << "WARNING: using more threads than available on system."
                << endl << "    Requested " << desiredNumProcs << " threads, "
                << "system reports " << numProcs << " threads available."
                << endl;
        }
        numProcs = desiredNumProcs;
    }
    cout << "Using " << numProcs << " local CPU compute thread(s)." << endl;

    // TODO: make a network counter class implements the same functions as
    // CollatzCounter (i.e. make CollatzCounter abstract), but ping the
    // server when those are called

    // TODO: client - have each thread take a new port, so that the only
    // choke point is at the actual counter object

    // shared counter and counter protector
    CollatzCounter collatzCounter;

    CollatzServer* server;
    // see if the user wants the server running on this machine
    if (vm.count("server")) {
        cout << "Starting server..." << endl;
        short port = vm["server"].as<short>();
        server = new CollatzServer(collatzCounter, port);
        cout << "Starting server... done" << endl;
    }

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
