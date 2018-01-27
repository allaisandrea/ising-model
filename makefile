all: run test
# 	clang-format -i lattice.h main.cpp Node.h observables.h test.cpp

run: makefile main.cpp lattice.h observables.h Node.h proto
	g++ main.cpp simulation.pb.o -g -O3 -o run -std=c++11 -Wall -Wextra  \
	-Werror -Wpedantic -lboost_program_options -lprotobuf

test: makefile test.cpp lattice.h observables.h Node.h 
	g++ test.cpp -o test -std=c++11  -Wall -Wextra -Werror -Wpedantic 

proto: simulation.proto
	protoc simulation.proto --cpp_out ./ --python_out ./data
	g++ simulation.pb.cc -c -g -o simulation.pb.o