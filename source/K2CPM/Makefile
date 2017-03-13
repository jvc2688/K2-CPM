#!/bin/bash
# ===================================================================#
# Makefile that compiles libcpm. It should work in most (all) UNIX
# systems. If you are not running this code in UNIX (Linux or MacOS), or
# if you would like to choose your own C++ compiler, please edit the
# lines 27 and 28.
# ===================================================================#

OSNAME = $(shell uname -s)
ifeq ($(OSNAME),Darwin)
	CC = clang++
else
	CC = g++
endif

#CFLAGS = -Wall
CFLAGS =
ifeq ($(OSNAME),Linux)
	CFLAGS += -std=gnu++11 -O2
else
	CFLAGS += -std=c++11 -stdlib=libc++ -O2
endif

# CC = your-own-compiler
# CFLAGS = -Wall -YourFlags

EXECUTABLE = libcpm
SOURCES = libcpm.cpp table.cpp matrix.cpp
OBJECTS = $(SOURCES:.cpp=.o)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) -c $<

clean:
	rm $(OBJECTS)
