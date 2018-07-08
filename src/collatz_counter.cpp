/**
 * @file collatz_counter.cpp
 *
 * This file contains the definition of the CollatzCounter class.
 */

#include <mutex>

#include "collatz_counter.hpp"

using namespace std;

CollatzCounter::CollatzCounter()
    : counter(1)
{
    counterProtector = new mutex();
}

CollatzCounter::~CollatzCounter()
{
    delete counterProtector;
}

uint64_t CollatzCounter::take(int size)
{
    lock_guard<mutex> l(*counterProtector);
    uint64_t current = counter;
    counter += size;
    return current;
}

uint64_t CollatzCounter::getCount()
{
    lock_guard<mutex> l(*counterProtector);
    return counter;
}
