
set(Collatz_CUDA ON CACHE STRING "To build or not to build the CUDA executable")
set_property(CACHE Collatz_CUDA PROPERTY TYPE BOOL)

if (Collatz_CUDA)
# find CUDA
find_package(CUDA)

# cuda executable also requires boost
find_package(Boost)

# add cuda target
cuda_add_executable(CollatzConjectureCUDA collatz.cu)
endif(Collatz_CUDA)

