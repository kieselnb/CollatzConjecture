//
// Created by Nick on 9/23/2016.
//

/// C INCLUDES
#include <stdio.h>

/// LOCAL INCLUDES
#include "collatz.h"

pthread_mutex_t collatzMutex;
int* initialized;
int* stop;
uint64_t* thisNum;

const uint64_t STEP = 1<<15;

void (*foo)(uint64_t*);

/*
 * TODO: make this take more than one number at a time
 * Probably make it configurable via command line
 *
 * Hard-coded step for now
 */
void takeNextNum(uint64_t* numBuf)
{
    pthread_mutex_lock(&collatzMutex);
    if (*stop)
    {
        // Someone got a bad value. Trigger an exit
        *numBuf = 0;
    } else {
        *numBuf = *thisNum;
        *thisNum = *thisNum + STEP;
    }
    pthread_mutex_unlock(&collatzMutex);
}

/*
 * TODO: modify this to check against the threshold of this group
 * of numbers, not each individual number
 */
int collatzRecursive(uint64_t num) {
    uint64_t myNum = num;
    while (myNum > 1 && myNum >= num) {
        if (myNum % 2 == 0) {
            myNum = myNum / 2;
        }
        else {
            myNum = ((3 * myNum) + 1) / 2;
        }
    }
    if (myNum > 1 && myNum < num) {
        return 1;
    }
    return myNum;


    /* Old recursive strategy. Sucks.
    if (myNum < 1)
    {
        return 0;
    }
    else if (myNum == 1)
    {
        return 1;
    }
    else if (myNum % 2 == 0)
    {
        myNum = myNum / 2;
        return collatzRecursive(myNum);
    }
    else
    {
        myNum = ((3 * myNum) + 1) / 2;
        return collatzRecursive(myNum);
    }
    */
}

static void * collatzThread(void * arg)
{
    uint64_t threadNum;

    while (1) {
        (*foo)(&threadNum);

        if (threadNum == 0) {
            // Somebody got a bad value. Exit
            break;
        }

        for (unsigned int i = 0; i < STEP; ++i) {
            if (collatzRecursive(threadNum + i) == 0) {
                // Something failed. Trigger an exit
                pthread_mutex_lock(&collatzMutex);
                *stop = 1;
                pthread_mutex_unlock(&collatzMutex);
            }
        }
    }
    return 0;
}

void collatzStart(int* tStop, uint64_t* num, long num_threads)
{
    pthread_attr_t pthreadAttr;
    pthread_attr_init(&pthreadAttr);
    printf("entered collatzStart\n");
    stop = tStop;
    thisNum = num;
    if (*initialized)
    {
        printf("Starting threads\n");
        pthread_t threadIds[num_threads];
        void * arg = NULL;
        int i = 0;
        for (i = 0; i < num_threads; i++) {
            pthread_create(&threadIds[i], &pthreadAttr, collatzThread, arg);
        }
    } else
    {
        printf("Collatz initialization failed or was not called\n");
    }
}

void collatzInit(int* initStatus, void* takeNextNumPointer)
{
    int status1, status2;
    initialized = initStatus;
    pthread_mutexattr_t mutexattr;

    status1 = pthread_mutexattr_init(&mutexattr);
    status2 = pthread_mutex_init(&collatzMutex, &mutexattr);

    foo = takeNextNumPointer;

    if ((status1 && status2) == 0)
    {
        *initialized = 1;
    }
}
