#ifndef COLLATZ_RUNNER_BOOST_H
#define COLLATZ_RUNNER_BOOST_H

#include <boost/compute.hpp>

#include "collatz_counter.hpp"
#include "collatz_runner.hpp"


class CollatzRunnerBoost : public CollatzRunner
{
public:
    CollatzRunnerBoost(CollatzCounter &counter);

    void start() override;

    void join() override;

private:
    static void runner(CollatzRunnerBoost& self);
};

#endif // COLLATZ_RUNNER_BOOST_H
