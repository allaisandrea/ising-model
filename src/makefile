CXX=clang++
COPTS=-std=c++11 -Wall -Wextra -Werror 
BUILD_DIR=../build
VPATH=$(BUILD_DIR)
HEADERS=ud_spin.h udh_spin.h udh_metropolis_algorithm.h tensor.h observables.h wolff_algorithm.h \
        udh_measure.h timer.h progress.h throttle.h udh_arguments.h

all: run-simulation

run-simulation: makefile udh_run_simulation.cpp $(HEADERS) udh_simulation.pb.o
	$(CXX) $(COPTS) udh_run_simulation.cpp udh_simulation.pb.o -o $(BUILD_DIR)/run-simulation \
        -lboost_program_options -lprotobuf

main: makefile main.cpp $(HEADERS)
	$(CXX) $(COPTS) main.cpp -o $(BUILD_DIR)/main -lboost_program_options

test: makefile test.cpp $(HEADERS) udh_simulation.pb.o
	$(CXX) $(COPTS)  test.cpp udh_simulation.pb.o -o $(BUILD_DIR)/test \
        -lgtest -lgtest_main -lprotobuf -lboost_program_options

udh_simulation.pb.o: makefile udh_simulation.proto
	protoc --cpp_out=./ udh_simulation.proto
	$(CXX) $(COPTS) -c udh_simulation.pb.cc
