#!/bin/bash
# ===================================================================#
# Makefile that compile libcpm. It is written for UNIX OS.The
# compilation commands can differ from one OS to another. Please choose
# between line 10 or 11 your C++ compiler: comment one of these two line
# by using the character '#'. #
# ===================================================================#

#CC=g++
CC=clang++
CFLAGS=-Wall

EXECUTABLE = libcpm
SOURCES = libcpm.cpp table.cpp matrix.cpp
OBJECTS = $(SOURCES:.cpp=.o)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) -c $<

clean:
	rm $(OBJECTS)
