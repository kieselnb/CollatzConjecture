#include <vector>
#include <algorithm>
#include <iostream>
#include <thread>

#include <boost/compute.hpp>

#include "collatz_runner_boost.h"
#include "collatz_counter.hpp"


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
    boost::compute::device gpu = boost::compute::system::default_device();
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

        boost::compute::copy(
                    d_numbers.begin(),
                    d_numbers.end(),
                    numbers.begin(),
                    queue);

        for (auto number : numbers)
        {
            if (number != 1) {
                std::cout << "WE BROKE SOMETHING" << std::endl;
                break;
            }
        }
    }
}
