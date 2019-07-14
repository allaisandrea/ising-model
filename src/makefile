CXX=clang++
COPTS=-std=c++11 -Wall -Wextra -Werror 
BUILD_DIR=../build
VPATH=$(BUILD_DIR)
HEADERS=ud_spin.h udh_spin.h udh_metropolis_algorithm.h tensor.h observables.h wolff_algorithm.h \
        udh_measure.h timer.h progress.h throttle.h udh_arguments.h

all: compute-autocorrelation print-parameters-table print-observables run-simulation

run-simulation: makefile udh_run_simulation.cpp $(HEADERS) udh_simulation.pb.o
	$(CXX) $(COPTS) udh_run_simulation.cpp udh_simulation.pb.o -o $(BUILD_DIR)/run-simulation \
        -lboost_program_options -lprotobuf

main: makefile main.cpp $(HEADERS)
	$(CXX) $(COPTS) main.cpp -o $(BUILD_DIR)/main -lboost_program_options

test: makefile test.cpp $(HEADERS) udh_simulation.pb.o
	$(CXX) $(COPTS)  test.cpp udh_simulation.pb.o -o $(BUILD_DIR)/test \
        -lgtest -lgtest_main -lprotobuf -lboost_program_options

compute-autocorrelation: makefile compute_autocorrelation.cpp udh_simulation.pb.o \
                         udh_parameters_set.h udh_file_group.h udh_io.h
	$(CXX) $(COPTS)  compute_autocorrelation.cpp udh_simulation.pb.o \
        -o $(BUILD_DIR)/compute-autocorrelation  -lprotobuf 

print-parameters-table: makefile print_parameters_table.cpp udh_simulation.pb.o \
                        udh_parameters_set.h udh_io.h
	$(CXX) $(COPTS)  print_parameters_table.cpp udh_simulation.pb.o \
        -o $(BUILD_DIR)/print-parameters-table -lprotobuf 

print-observables: makefile print_observables.cpp udh_simulation.pb.o udh_io.h
	$(CXX) $(COPTS)  print_observables.cpp udh_simulation.pb.o \
        -o $(BUILD_DIR)/print-observables  -lprotobuf 

udh_simulation.pb.o: makefile udh_simulation.proto
	protoc --cpp_out=./ udh_simulation.proto
	$(CXX) $(COPTS) -c udh_simulation.pb.cc
