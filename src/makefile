BIN_DIR=../bin/
all: run test
# 	clang-format -i lattice.h main.cpp Node.h observables.h test.cpp

run: makefile main.cpp lattice.h observables.h Node.h
	g++ main.cpp -g -O3 -o $(BIN_DIR)/run -std=c++11 -Wall -Wextra  \
	-Werror -Wpedantic -lboost_program_options

test: makefile test.cpp lattice.h observables.h Node.h 
	g++ test.cpp -o $(BIN_DIR)/test -std=c++11  -Wall -Wextra -Werror \
	-Wpedantic 