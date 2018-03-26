/*
 * cuda implementation of collatz conjecture
 */

#include <iostream>
#include <sstream>

#include <boost/format.hpp>

#include <time.h>
#include <signal.h>

__global__
void collatz(int n, long long unsigned int threshold, int* status) {
  int k = 2 * ((blockIdx.x * blockDim.x) + threadIdx.x);

  if (k < n) {
    // Use tid to get offset
    long long unsigned int number = threshold + k + 1; // +1 since we took off one from currentnum to get threshold

    for (int j = 0; j < 2; j++) {
      while (number > 1 && number > threshold) {
        if (number % 2 == 0)
          number = number >> 1;
        else
          number = ((number * 3) + 1) >> 1;
      }
      if ((number > 1 && number <= threshold) || (number == 1))
        number = 1;
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
  
  // Create status variable space
  int status, *d_status;
  cudaError_t err = cudaMalloc(&d_status, sizeof(int));
  if (err != cudaSuccess) {
    std::cout << "cudaMalloc failed, did you forget optirun?" << std::endl;
    return 1;
  }

  // start main loop
  clock_gettime(CLOCK_REALTIME, &tp_start);
  clock_gettime(CLOCK_REALTIME, &tp_loopstart);
  while (keep_going) {
    // log timestamp and stats to console
    clock_gettime(CLOCK_REALTIME, &tp_end);
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
      clock_gettime(CLOCK_REALTIME, &tp_loopstart);
    }

    // initialize status to success
    status = 1;
    cudaMemcpy(d_status, &status, sizeof(int), cudaMemcpyHostToDevice);
    
    // perform collatz on entire array
    collatz<<<((N/2)+255)/256, 256>>>(N, (currentNumber-1), d_status);
    
    // bring status back
    cudaMemcpy(&status, d_status, sizeof(int), cudaMemcpyDeviceToHost);
    
    // just check status variable
    if (status == 0) {
      std::cout << "Collatz failed" << std::endl;
      break;
    }

    currentNumber += N;
  }
  
  std::cout << "While loop broken. Cleaning up..." << std::endl;
  cudaFree(d_status);
  
  return 0;
}
