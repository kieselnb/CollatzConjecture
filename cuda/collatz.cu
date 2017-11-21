/*
 * cuda implementation of collatz conjecture
 */

#include <iostream>
#include <sstream>

#include <boost/format.hpp>

#include <time.h>
#include <signal.h>

__global__
void collatz(int n, long long unsigned int threshold, long long unsigned int *numbers) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < n) {
    // we've already checked numbers <= threshold, so don't do those again
    while (numbers[i] > 1 && numbers[i] > threshold) {
      if (numbers[i] % 2 == 0)
        numbers[i] = numbers[i] / 2;
      else
        numbers[i] = ((numbers[i] * 3) + 1) / 2;
    }
    if ((numbers[i] > 1) && (numbers[i] <= threshold))
      numbers[i] = 1;
  }
}

bool keep_going;

void signalHandler(int sig) {
  std::cout << "Received SIGINT. Exiting..." << std::endl;
  keep_going = false;
}

int main(int argc, char* argv[]) {
  // first thing's first, start signal handler to catch SIGINT and exit gracefully
  keep_going = true;
  signal(SIGINT, signalHandler);
  
  long long unsigned int currentNumber = 1;
  // let's have the user specify a start number if they want
  std::cout << "Enter a number greater than or equal to 1 to start from [default: 1]: ";
  std::string buffer;
  getline(std::cin, buffer);
  if (!buffer.empty()) {
    std::stringstream ss(buffer);
    ss >> currentNumber;
  }
  if (currentNumber < 1)
    currentNumber = 1;
  
  int N = 1<<21;
  struct timespec tp_start, tp_loopstart, tp_end;
  
  long long unsigned int *numbers, *d_numbers;
  // Initialize main memory array
  numbers = (long long unsigned int *) malloc(N * sizeof(long long unsigned int));
  
  // Initialize GPU memory array
  cudaMalloc(&d_numbers, N * sizeof(long long unsigned int));
  
  // start main loop
  int result = clock_gettime(CLOCK_REALTIME, &tp_start);
  result = clock_gettime(CLOCK_REALTIME, &tp_loopstart);
  while (keep_going) {
    result = clock_gettime(CLOCK_REALTIME, &tp_end);
    if (tp_end.tv_sec - tp_loopstart.tv_sec > 4) {
      std::stringstream timestamp;
      timestamp << "Uptime: ";
      timestamp << boost::str(boost::format("%02d:%02d")
                              % ((tp_end.tv_sec - tp_start.tv_sec) / 60)
                              % ((tp_end.tv_sec - tp_start.tv_sec) % 60));
      timestamp << ". ";
      timestamp << "Current number: ";
      timestamp << boost::str(boost::format("%12d") % currentNumber);
      timestamp << std::endl;
      std::cout << timestamp.str();
      result = clock_gettime(CLOCK_REALTIME, &tp_loopstart);
    }

    // populate the array
    for (int i = 0; i < N; i++) {
      numbers[i] = currentNumber + i;
    }
    
    // copy to device
    cudaMemcpy(d_numbers, numbers, N*sizeof(long long unsigned int), cudaMemcpyHostToDevice);
    
    // perform collatz on entire array
    collatz<<<(N+255)/256, 256>>>(N, (currentNumber-1), d_numbers);
    
    // copy back to main mem
    cudaMemcpy(numbers, d_numbers, N*sizeof(long long unsigned int), cudaMemcpyDeviceToHost);
    
    // check for errors (non-one values)
    for (int i = 0; i < N; i++) {
      if (numbers[i] != 1) {
        std::cout << "Collatz failed on number: " << (currentNumber + i) << std::endl;
        keep_going = false;
        break;
      }
    }
    currentNumber += N;
  }
  
  std::cout << "While loop broken. Cleaning up..." << std::endl;
  cudaFree(d_numbers);
  free(numbers);
  
  return 0;
}
