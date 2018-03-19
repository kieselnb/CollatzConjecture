/*
 * cuda implementation of collatz conjecture
 */

#include <iostream>
#include <sstream>

#include <boost/format.hpp>

#include <time.h>
#include <signal.h>

__global__
void collatz(int n, long long unsigned int threshold, long long unsigned int *numbers, int* status) {
  int k = 2 * ((blockIdx.x * blockDim.x) + threadIdx.x);

  // each thread gets two numbers - an odd and an even - to mitigate work imbalance
  if (k < n) {
    for (int j = 0; j < 2; j++) {
      int i = k + j;
      // we've already checked numbers <= threshold, so don't do those again
      while (numbers[i] > 1 && numbers[i] > threshold) {
        if (numbers[i] % 2 == 0)
          numbers[i] = numbers[i] >> 1;
        else
          numbers[i] = ((numbers[i] * 3) + 1) >> 1;
      }
      if ((numbers[i] > 1) && (numbers[i] <= threshold) || numbers[i]==1)
        numbers[i] = 1;
      else
        *status = 0;
    }
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
  
  // Create status variable space
  int status, *d_status;
  cudaError_t err = cudaMalloc(&d_status, sizeof(int));
  if (err != cudaSuccess) {
    std::cout << "cudaMalloc failed, did you forget optirun?" << std::endl;
    return 1;
  }

  // Initialize GPU memory array
  cudaMalloc(&d_numbers, N * sizeof(long long unsigned int));
  
  // start main loop
  int result = clock_gettime(CLOCK_REALTIME, &tp_start);
  result = clock_gettime(CLOCK_REALTIME, &tp_loopstart);
  while (keep_going) {
    // log timestamp and stats to console
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

    // initialize status to success
    status = 1;
    cudaMemcpy(d_status, &status, sizeof(int), cudaMemcpyHostToDevice);
    
    // perform collatz on entire array
    collatz<<<((N/2)+255)/256, 256>>>(N, (currentNumber-1), d_numbers, d_status);
    
    // copy back to main mem
    // cudaMemcpy(numbers, d_numbers, N*sizeof(long long unsigned int), cudaMemcpyDeviceToHost);

    // bring status back
    cudaMemcpy(&status, d_status, sizeof(int), cudaMemcpyDeviceToHost);
    
    // just check status variable
    if (status == 0) {
      std::cout << "Collatz failed" << std::endl;
      // copy back to main mem and check where we failed
      cudaMemcpy(numbers, d_numbers, N*sizeof(long long unsigned int), cudaMemcpyDeviceToHost);
      for (int i = 0; i < N; i++) {
        if (numbers[i] != 1) {
          std::cout << "Collatz failed on: " << (currentNumber + i) << std::endl;
          keep_going = false;
          break;
        }
      }
      break;
    }

    // check for errors (non-one values)
    // for (int i = 0; i < N; i++) {
    //   if (numbers[i] != 1) {
    //     std::cout << "Collatz failed on number: " << (currentNumber + i) << std::endl;
    //     keep_going = false;
    //     break;
    //   }
    // }
    currentNumber += N;
  }
  
  std::cout << "While loop broken. Cleaning up..." << std::endl;
  cudaFree(d_numbers);
  cudaFree(d_status);
  free(numbers);
  
  return 0;
}
