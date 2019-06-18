CXX=g++
COPTS=-std=c++11 -Wall -Wextra -Werror
BUILD_DIR=../build
VPATH=$(BUILD_DIR)
main:  makefile main.cpp Node.h lattice.h observables.h
	$(CXX) $(COPTS) main.cpp -o $(BUILD_DIR)/main -lboost_program_options

test: makefile test.cpp Node.h lattice.h observables.h
	$(CXX) $(COPTS)  test.cpp -o $(BUILD_DIR)/test -lgtest -lgtest_main 
