
if (Collatz_CUDA)
# find CUDA
find_package(CUDA)

# cuda executable also requires boost
find_package(Boost)

# add cuda target
cuda_add_executable(CollatzConjectureCUDA collatz.cu)
endif(Collatz_CUDA)

