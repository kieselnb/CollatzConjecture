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

#include "config.h"
#include "collatz_counter.hpp"
#include "collatz_counter_client.hpp"
#include "collatz_counter_local.hpp"
#include "collatz_runner.hpp"
#include "collatz_runner_cpu.hpp"
#include "collatz_server.hpp"

#ifdef ENABLE_CUDA
#include "collatz_runner_gpu.cuh"
#endif

#ifdef ENABLE_OPENCL
#include "collatz_runner_boost.h"
#endif


using namespace std;
namespace po = boost::program_options;

void usage() {
    std::cout
        << "Options:\n"
        << "  -h [ --help ]         Display this message\n"
        << "  -n [ --numproc ] arg  Number of CPU threads to use\n"
#ifdef ENABLE_CUDA
        << "  -g [ --gpu ]          Run the CUDA implementation on the GPU\n"
#endif
#ifdef ENABLE_OPENCL
        << "  -o [ --opencl ]       Run the OpenCL implementation on the GPU\n"
#endif
        << "  -s [ --server ] arg   Start a CollatzServer on this machine\n"
        << "  -c [ --client ] arg   Point this machine as a client to the given server\n"
        << std::endl;
}

int main(int argc, char* argv[]) {
    // parse options
    po::options_description desc("Options");
    desc.add_options()
        ("help,h", "Display this message")
        ("numproc,n", po::value<unsigned int>(), "Number of cpu threads to use")
#ifdef ENABLE_CUDA
        ("gpu,g", "Activate the gpu thread")
        ("opencl,o", "Use OpenCL to run the gpu thread")
#endif
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

    // get number of available threads - use either that or the
    // user-specified number of threads
    unsigned int numProcs = thread::hardware_concurrency();

    int skip = 0;
    for (int i = 0; i < argc; i += 1 + skip) {
        skip = 0;

        std::string arg(argv[i]);
        if (arg == "-h" || arg == "--help") {
            usage();
            exit(0);
        }
        else if (arg == "-n" || arg == "--numproc") {
            skip = 1;
            bool threw = false;

            std::string numProcArg = argv[i+1];
            std::size_t pos;
            unsigned int desiredNumProcs;
            try {
                desiredNumProcs = std::stoul(numProcArg, &pos);
            }
            catch (std::invalid_argument const &e) {
                std::cerr << "Invalid number: " << numProcArg << std::endl;
                threw = true;
            }
            catch (std::out_of_range const &e) {
                std::cerr << "Number out of range: " << numProcArg << std::endl;
                threw = true;
            }

            if (threw || std::to_string(desiredNumProcs).size() != pos) {
                std::cerr << "Conversion failed" << std::endl;
                exit(1);
            }

            if (numProcs < desiredNumProcs) {
                std::cout << "WARNING: using more threads than available on system."
                    << std::endl << "    Requested " << desiredNumProcs << " threads, "
                    << "system reports " << numProcs << " threads available."
                    << std::endl;
            }
            numProcs = desiredNumProcs;
        }
    }

    cout << "Using " << numProcs << " local CPU compute thread(s)." << endl;

    // expose CUDA and OpenCL separately, but only use one

    bool useGPU = false;
#ifdef ENABLE_CUDA
    bool useCUDA = false;
    if (vm.count("gpu")) {
        cout << "Using local GPU" << endl;
        useGPU = true;
        useCUDA = true;
    }

    if (vm.count("opencl")) {
        if (useCUDA) {
            std::cout << "Cannot use both CUDA and OpenCL simultaneously - pick one"
                      << std::endl;
            return 1;
        }
        useGPU = true;
    }
#endif

    unsigned int numRunners = useGPU ? (numProcs + 1) : numProcs;

    // shared counter and counter protector
    // if in client config, we'll just ignore this
    CollatzCounterLocal collatzCounter;

    // make an array of CollatzCounter pointers
    // if we are in a client configuration, each will be a new instance
    // of the CollatzCounterClient class
    // otherwise, all will point to the same CollatzCointer object
    vector<CollatzCounter*> counters(numRunners);

    if (vm.count("client")) {
        // parse arg (of form 'x.x.x.x:y' into string ip, short port
        string ipPort = vm["client"].as<string>();
        string serverAddress = ipPort.substr(0, ipPort.find(':'));
        int given = stoi(ipPort.substr(ipPort.find(':') + 1));
        if (given > numeric_limits<unsigned short>::min() &&
                given < numeric_limits<unsigned short>::max()) {
            auto serverPort = static_cast<unsigned short>(given);
            for (auto & counter : counters) {
                counter = new CollatzCounterClient(serverAddress, serverPort);
            }
        }
        else {
            cerr << "Invalid port" << endl;
            exit(1);
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
        server->run();
        cout << "Starting server... done" << endl;
    }

    // kick off runners for each core
    vector<CollatzRunner*> runners(numRunners);
    for (unsigned int i = 0; i < numProcs; i++) {
        runners[i] = new CollatzRunnerCPU(*counters[i]);
    }

    // if using GPU, last runner is GPU
#if defined(ENABLE_CUDA) || defined(ENABLE_OPENCL)
    if (useGPU) {
        // do the selection if both were enabled
#if defined(ENABLE_CUDA) && defined(ENABLE_OPENCL)
        runners[numRunners-1] = useCUDA ? (CollatzRunner*)new CollatzRunnerGPU(*counters[numRunners-1])
                                        : (CollatzRunner*)new CollatzRunnerBoost(*counters[numRunners-1]);
#else
        // just assign the one enabled if not both were enabled
        runners[numRunners-1] =
#ifdef ENABLE_CUDA
            (CollatzRunner*)new CollatzRunnerGPU(*counters[numRunners-1]);
#else
            (CollatzRunner*)new CollatzRunnerBoost(*counters[numRunners-1]);
#endif /* ENABLE_CUDA */

#endif /* defined(ENABLE_CUDA) && defined(ENABLE_OPENCL) */
    }
#endif /* defined(ENABLE_CUDA) || defined(ENABLE_OPENCL) */

    // get value before threads start for perf checking
    uint64_t lastCount = collatzCounter.getCount();

    // start all threads
    for (auto & runner : runners) {
        runner->start();
    }

    // idle loop to calculate perf and watch progress
    while (true) {
        uint64_t thisCount = collatzCounter.getCount();
        float perf = float(thisCount - lastCount) / 10.0f;
        cout << "Current: " << thisCount << ". Perf: " << perf
            << " numbers/sec." << endl;
        lastCount = thisCount;
        this_thread::sleep_for(chrono::seconds(10));
    }

    return 0;
}
