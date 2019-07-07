CXX=clang++
COPTS=-std=c++11 -Wall -Wextra -Werror 
BUILD_DIR=../build
VPATH=$(BUILD_DIR)
HEADERS=ud_spin.h udh_spin.h udh_metropolis_algorithm.h tensor.h observables.h wolff_algorithm.h

all: test

main:  makefile main.cpp $(HEADERS)
	$(CXX) $(COPTS) main.cpp -o $(BUILD_DIR)/main -lboost_program_options

test: makefile test.cpp $(HEADERS) udh_simulation.pb.o
	$(CXX) $(COPTS)  test.cpp udh_simulation.pb.o -o $(BUILD_DIR)/test \
        -lgtest -lgtest_main -lprotobuf

udh_simulation.pb.o: makefile udh_simulation.proto
	protoc --cpp_out=./ udh_simulation.proto
	$(CXX) $(COPTS) -c udh_simulation.pb.cc
