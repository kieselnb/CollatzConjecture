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

unsigned long long int CollatzCounter::take(int size)
{
    lock_guard<mutex> l(*counterProtector);
    unsigned long long int current = counter;
    counter += size;
    return current;
}

unsigned long long int CollatzCounter::getCount()
{
    lock_guard<mutex> l(*counterProtector);
    return counter;
}
