//
// Created by Nick on 9/23/2016.
//

#include <stdio.h>
#include "collatz.h"

pthread_mutex_t collatzMutex;
int* initialized;
int* stop;
uint64_t* thisNum;

void (*foo)(uint64_t*);

void takeNextNum(uint64_t* numBuf)
{
    pthread_mutex_lock(&collatzMutex);
    if (*stop)
    {
        // Someone got a bad value. Trigger an exit
        *numBuf = 0;
    } else {
        *numBuf = *thisNum;
        *thisNum = *thisNum + 1;
    }
    pthread_mutex_unlock(&collatzMutex);
}

int collatzRecursive(uint64_t num)
{
    uint64_t myNum = num;
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
        myNum = (3 * myNum) + 1;
        return collatzRecursive(myNum);
    }
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

        if (collatzRecursive(threadNum) == 0) {
            // Something failed. Trigger an exit
            pthread_mutex_lock(&collatzMutex);
            *stop = 1;
            pthread_mutex_unlock(&collatzMutex);
        }
    }
    return 0;
}

void collatzStart(int* tStop, uint64_t* num)
{
    pthread_attr_t pthreadAttr;
    pthread_attr_init(&pthreadAttr);
    printf("entered collatzStart\n");
    stop = tStop;
    thisNum = num;
    if (*initialized)
    {
        printf("Starting threads\n");
        pthread_t threadIds[4];
        void * arg = NULL;
        int i = 0;
        for (i = 0; i < 4; i++) {
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
