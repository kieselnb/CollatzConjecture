/**
 * @file collatz_counter_local.cpp
 *
 * This file contains the definition of the CollatzCounterLocal class.
 */

#include <mutex>

#include "collatz_counter.hpp"
#include "collatz_counter_local.hpp"

using namespace std;

CollatzCounterLocal::CollatzCounterLocal()
{
    counterProtector = new mutex();
}

CollatzCounterLocal::~CollatzCounterLocal()
{
    delete counterProtector;
}

uint64_t CollatzCounterLocal::take(unsigned int size)
{
    lock_guard<mutex> l(*counterProtector);
    uint64_t current = counter;
    counter += size;
    return current;
}

uint64_t CollatzCounterLocal::getCount()
{
    lock_guard<mutex> l(*counterProtector);
    return counter;
}
