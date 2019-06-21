CXX=g++
COPTS=-std=c++11 -Wall -Wextra -Werror
BUILD_DIR=../build
VPATH=$(BUILD_DIR)
HEADERS=Node.h lattice.h observables.h wolff_algorithm.h

all: main test

main:  makefile main.cpp $(HEADERS)
	$(CXX) $(COPTS) main.cpp -o $(BUILD_DIR)/main -lboost_program_options

test: makefile test.cpp $(HEADERS) poisson_distribution.h
	$(CXX) $(COPTS)  test.cpp -o $(BUILD_DIR)/test -lgtest -lgtest_main 
