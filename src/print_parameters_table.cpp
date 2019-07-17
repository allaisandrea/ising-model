#include "udh_parameters_set.h"

int main(int argc, char **argv) {
    const auto pairs = ReadUdhParametersFromFiles(argv + 1, argv + argc);
    PrintParametersAsCsv(pairs.begin(), pairs.end(), &std::cout);
}
