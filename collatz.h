//
// Created by Nick on 9/23/2016.
//

#ifndef COLLATZCONJECTURE_COLLATZ_H
#define COLLATZCONJECTURE_COLLATZ_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <pthread.h>

int collatzRecursive(uint64_t num);

void collatzStart(int* tStop, uint64_t* num, int num_threads);

void collatzInit(int* initStatus, void* takeNextNumPointer);

void takeNextNum(uint64_t* numBuf);

#ifdef __cplusplus
}
#endif

#endif //COLLATZCONJECTURE_COLLATZ_H
