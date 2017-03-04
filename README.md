# K2-CPM
Causal Pixel Model for K2 data

# How to use
```
$ python run_cpm.py

positional arguments:
  epic          int k2 epic number
  campaign      int campaign number, 91 for phase a, 92 for phase b
  n_predictors  int number of the predictors pixels
  l2            float strength of l2 regularization
  n_pca         int number of the PCA components to use, if 0, no PCA 
  distance      int distance between target pixel and predictor pixels
  exclusion     int how many rows and columns that are excluded around the target pixel
  input_dir     str directory to the target pixel files
  output_dir    str path to the output file

optional arguments:
  -p [pixel_list], --pixel [pixel_list]
                str path to the pixel list file that specify which pixels to be modelled. If not provided, the whole target pixel file will be modeled
```

# Example
```
$ python run_cpm.py 200069974 92 800 1e3 0 16 5 ./tpf ./output/200069974 -p ./test_pixel.dat
```

# Using the C++ version of the code

## Introduction

The C++ version replace the calculations performed when running
`cpm_part2.py`,  but not the ones done by `cpm_part1.py`. The C++
files must be compiled by hand prior using it for
the first time, as explained below.

## Files

The C++ version consists in the following files.

* `table.cpp` and the corresponding header file `table.h`: these files
define a  new class of one, two or three dimensional tables.

* `matrix.cpp` and the corresponding header file `matrix.h`: these
files define a new class  of square matrix that includes some very
commun operations such as Cholesky's  decomposition and a linear
system solver.

* `libcpm.cpp` and the corresponding header file `libcpm.h`: these
files corresponds to the  main code that should be executed.

* `Makefile`: this file allow compilation of the C++ version on most
of the OS without any editing.

## Compilation

In most UNIX OS, the C++ version can be compiled by the following commands:
```
$ cd path/K2-CPM/code/
$ make
```

where `path` is the full path to the directory `K2-CPM` on your machine.

If the command `make` returns an error, it might be because your C++ compiler is not found 
or because you have not a C++ compiler.

If the first case, the problem can be solve by editing the following line
in the file `Makefile`:
```
#CC=g++
CC=clang++
CFLAGS=-Wall
```
one can uncomment the first line and comment the second one to use `g++` as a compiler
rather than `clang++`, or just choose another compiler.

In the second case, it is necessary to install a C++ compiler. A free option is to 
install GCC 5 or later (see <https://gcc.gnu.org>).

## Usage

For now, the C++ version is tested and can only be run for the unit tests.
Once compiled, the C++ version can be run directly as follow,
```
$ ./libcpm
```
or with optional arguments,
```
$ ./libcpm 1
```
where 1 is the number of the test. Soon, a working version that includes
microlensing models will come.

## Future developments

* A version that can be used directly from a modeling code.
* C++ version that can be used as a shared library (from C, C++ and fortran).
