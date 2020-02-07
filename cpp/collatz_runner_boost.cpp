/* Just disable the entire file if GPU was disabled */
#include "config.h"

#ifdef ENABLE_CUDA

#include "collatz_runner_boost.h"
#include "collatz_counter.hpp"

#include <vector>
#include <algorithm>
#include <iostream>
#include <thread>

#include <boost/compute.hpp>

// class generator:
static struct c_unique {
  uint64_t current;
  c_unique() {current=0;}
  void set_val(uint64_t val) {current = val;}
  uint64_t operator()() {return ++current;}
} UniqueNumber;

// boost.compute version of collatz - lets transform be used
BOOST_COMPUTE_FUNCTION(uint64_t, collatz, (uint64_t x),
{
    while (x > 1)
    {
        if (x % 2 == 0)
        {
            x >>= 1;
        }
        else
        {
            x = ((3 * x) + 1) >> 1;
        }
    }

    return x;
});

CollatzRunnerBoost::CollatzRunnerBoost(CollatzCounter &counter)
    : CollatzRunner(counter)
{
}

void CollatzRunnerBoost::start()
{
    _collatzThread = new std::thread(runner, std::ref(*this));
}

void CollatzRunnerBoost::join()
{
    _collatzThread->join();
}

void CollatzRunnerBoost::runner(CollatzRunnerBoost &self)
{
    auto devices = boost::compute::system::devices();
    std::cout << "Avaialble devices:" << std::endl;
    for (auto &device : devices) {
        std::cout << "\t" << device.name() << std::endl;
    }

    boost::compute::device gpu = devices[0];
    boost::compute::context ctx(gpu);
    boost::compute::command_queue queue(ctx, gpu);

    std::cout << "Using OpenCL device " << gpu.name() << std::endl;

    self._stride = 1 << 21;
    std::vector<uint64_t> numbers(self._stride);
    boost::compute::vector<uint64_t> d_numbers(numbers.size(), ctx);

    while (true)
    {
        uint64_t start = self._counter.take(self._stride);
        UniqueNumber.set_val(start);
        std::generate(numbers.begin(), numbers.end(), UniqueNumber);

        boost::compute::copy(
                    numbers.begin(),
                    numbers.end(),
                    d_numbers.begin(),
                    queue);

        boost::compute::transform(
                        d_numbers.begin(),
                        d_numbers.end(),
                        d_numbers.begin(),
                        collatz,
                        queue);

        std::vector<uint64_t> result(1);
        boost::compute::reduce(
                    d_numbers.begin(),
                    d_numbers.end(),
                    result.begin(),
                    queue);

        if (result[0] != self._stride)
        {
            std::cout << "WE BROKE SOMETHING" << std::endl;
        }
    }
}

#endif
