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
#include "collatz_counter_client.hpp"
#include "collatz_counter_local.hpp"
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
        ("numproc,n", po::value<int>(), "Number of cpu threads to use")
        ("server,s", po::value<short>(),
            "Start a CollatzServer on this machine")
        ("client,c", po::value<string>(),
            "Point this machine as a client to the given server")
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
    if (vm.count("numproc")) {
        int desiredNumProcs = vm["numproc"].as<int>();
        if (desiredNumProcs > numProcs) {
            cout << "WARNING: using more threads than available on system."
                << endl << "    Requested " << desiredNumProcs << " threads, "
                << "system reports " << numProcs << " threads available."
                << endl;
        }
        numProcs = desiredNumProcs;
    }
    cout << "Using " << numProcs << " local CPU compute thread(s)." << endl;

    // shared counter and counter protector
    // if in client config, we'll just ignore this
    CollatzCounterLocal collatzCounter;

    // make an array of CollatzCounter pointers
    // if we are in a client configuration, each will be a new instance
    // of the CollatzCounterClient class
    // otherwise, all will point to the same CollatzCointer object
    vector<CollatzCounter*> counters(numProcs);

    if (vm.count("client")) {
        // parse arg (of form 'x.x.x.x:y' into string ip, short port
        string ipPort = vm["client"].as<string>();
        string serverAddress = ipPort.substr(0, ipPort.find(':'));
        short serverPort = stoi(ipPort.substr(ipPort.find(':') + 1));

        for (auto & counter : counters) {
            counter = new CollatzCounterClient(serverAddress, serverPort);
        }
    }
    else {
        for (auto & counter : counters) {
            counter = &collatzCounter;
        }
    }

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
        runners[i] = new CollatzRunnerCPU(*counters[i]);
    }

    // get value before threads start for perf checking
    uint64_t lastCount = collatzCounter.getCount();

    // start all threads
    for (auto & runner : runners) {
        runner->start();
    }

    while (true) {
        uint64_t thisCount = collatzCounter.getCount();
        float perf = float(thisCount - lastCount) / 10.0;
        cout << "Current: " << thisCount << ". Perf: " << perf
            << " numbers/sec." << endl;
        lastCount = thisCount;
        this_thread::sleep_for(chrono::seconds(10));
    }

    return 0;
}
