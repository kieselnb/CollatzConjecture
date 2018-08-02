# CollatzConjecture
C implementation of an exhaustive method of checking the Collatz Conjecture

This project is an exhaustive check of numbers against the Collatz Conjecture (explained [here](https://youtu.be/5mFpVDpKX70)).

Although the usefulness of this project is very questionable, it's become a great source of coding practice for myself.

### Requirements
* Boost
* CUDA (for the CUDA executable)

### Building
This project uses CMake, so build using this format after cloning:
```
mkdir build
cd build
cmake ..
make
```

By default, this will build the C and C++ versions of the code. To enable build of the CUDA executable, turn on the
Collatz_CUDA flag using either `ccmake .` or `cmake .. -DCollatz_CUDA=ON`.

### Documentation
The C++ code has adopted a Doxygen-compatible documentation standard, so documentation can be generated by just running Doxygen
in the checkout directory.
