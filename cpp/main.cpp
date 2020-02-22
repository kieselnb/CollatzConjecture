/**
 * @file main.cpp
 *
 * This file contains the main definition for the Collatz Conjecture
 * project
 */

#include "config.h"
#include "collatz_counter.hpp"
#include "collatz_counter_client.hpp"
#include "collatz_counter_local.hpp"
#include "collatz_runner.hpp"
#include "collatz_runner_cpu.hpp"
#include "collatz_server.hpp"

#include <iostream>
#include <vector>
#include <chrono>
#include <string>

#ifdef ENABLE_CUDA
#include "collatz_runner_gpu.cuh"
#endif

#ifdef ENABLE_OPENCL
#include "collatz_runner_boost.h"
#endif


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
    // get number of available threads - use either that or the
    // user-specified number of threads
    unsigned int numProcs{std::thread::hardware_concurrency()};

    // keep track of GPU requests
    bool useCUDA{false};
    bool useOpenCL{false};

    // only port will signify server, both populated will be client
    int port{-1};
    std::string serverAddress;

    // parse options
    int skip = 0;
    for (int i = 1; i < argc; i += 1 + skip) {
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
#ifdef ENABLE_CUDA
        else if (arg == "-g" || arg == "--gpu") {
            // see if opencl was enabled already
            if (useOpenCL) {
                std::cerr << "Error: both OpenCL and CUDA implementations "
                    << "requested. Pick one." << std::endl;
                std::exit(1);
            }
            std::cout << "Using local NVIDIA GPU" << std::endl;
            useCUDA = true;
        }
#endif
#ifdef ENABLE_OPENCL
        else if (arg == "-o" || arg == "--opencl") {
            if (useCUDA) {
                std::cerr << "Error: both OpenCL and CUDA implementations "
                    << "requested. Pick one." << std::endl;
                std::exit(1);
            }
            useOpenCL = true;
        }
#endif
        else if (arg == "-c" || arg == "--client") {
            skip = 1;

            // ensure another argument was given
            if (i+skip >= argc) {
                std::cerr << "Error: --client requires one argument" << std::endl;
                usage();
                std::exit(1);
            }

            // pull out server hostname/ip and port
            std::string ipPort = argv[i+1];
            serverAddress = ipPort.substr(0, ipPort.find(':'));
            int given = std::stoi(ipPort.substr(ipPort.find(':') + 1));
            if (given > std::numeric_limits<unsigned short>::max() ||
                    given < std::numeric_limits<unsigned short>::min()) {
                std::cerr << "Error: invalid port: " << given << std::endl;
                std::exit(1);
            }
            port = given;
        }
        else if (arg == "-s" || arg == "--server") {
            skip = 1;

            // ensure port was given
            if (i+skip >= argc) {
                std::cerr << "Error: enabling server requires one argument" << std::endl;
                usage();
                std::exit(1);
            }

            // pull out port - ensure it is sane
            int given = std::stoi(argv[i+1]);
            if (given > std::numeric_limits<unsigned short>::max() ||
                    given < std::numeric_limits<unsigned short>::min()) {
                std::cerr << "Error: invalid port: " << given << std::endl;
                std::exit(1);
            }
            port = given;
        }
        else {
            std::cerr << "Unrecognized option\n" << std::endl;
            usage();
            std::exit(1);
        }
    }

    std::cout << "Using " << numProcs << " local CPU compute thread(s)." << std::endl;

    unsigned int numRunners = (useCUDA || useOpenCL) ? (numProcs + 1) : numProcs;

    // shared counter and counter protector
    // if in client config, we'll just ignore this
    CollatzCounterLocal collatzCounter;

    // make an array of CollatzCounter pointers
    // if we are in a client configuration, each will be a new instance
    // of the CollatzCounterClient class
    // otherwise, all will point to the same CollatzCointer object
    std::vector<CollatzCounter*> counters(numRunners, &collatzCounter);

    CollatzServer* server;

    // if we have a server address, in client mode
    if (!serverAddress.empty()) {
        // populate the counters with new guys
        unsigned short serverPort = static_cast<unsigned short>(port);
        for (auto &counter : counters) {
            counter = new CollatzCounterClient(serverAddress, serverPort);
        }
    }
    else if (port != -1) {
        // if serverAddress was empty but port was given, kick off the server
        std::cout << "Starting server..." << std::endl;
        server = new CollatzServer(collatzCounter, port);
        server->run();
        std::cout << "Starting server... done" << std::endl;
    }

    // kick off runners for each core
    std::vector<CollatzRunner*> runners(numRunners);
    for (unsigned int i = 0; i < numProcs; i++) {
        runners[i] = new CollatzRunnerCPU(*counters[i]);
    }

    // if using GPU, last runner is GPU
#if defined(ENABLE_CUDA) || defined(ENABLE_OPENCL)
    if (useCUDA || useOpenCL) {
        runners[numRunners-1] =
#ifdef ENABLE_CUDA
#ifdef ENABLE_OPENCL
            useCUDA ?
#endif
            (CollatzRunner*)new CollatzRunnerGPU(*counters[numRunners-1])
#endif
#ifdef ENABLE_OPENCL
#ifdef ENABLE_CUDA
            :
#endif
            (CollatzRunner*)new CollatzRunnerBoost(*counters[numRunners-1])
#endif
            ;
    }
#endif

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
        std::cout << "Current: " << thisCount << ". Perf: " << perf
            << " numbers/sec." << std::endl;
        lastCount = thisCount;
        std::this_thread::sleep_for(std::chrono::seconds(10));
    }

    return 0;
}
