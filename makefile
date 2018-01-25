all: run test

run: makefile main.cpp lattice.h observables.h Node.h proto
	g++ main.cpp simulation.pb.o -o run -std=c++11 -Wall -Wextra -Werror -Wpedantic \
	-lboost_program_options -lprotobuf

test: makefile test.cpp lattice.h observables.h Node.h 
	g++ test.cpp -o test -std=c++11  -Wall -Wextra -Werror -Wpedantic 

proto: simulation.proto
	protoc simulation.proto --cpp_out ./ --python_out ./data
	g++ simulation.pb.cc -c -o simulation.pb.o