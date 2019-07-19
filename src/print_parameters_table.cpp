#include "udh_parameters_set.h"

int main(int argc, char **argv) {
    auto pairs = ReadUdhParametersFromFiles(argv + 1, argv + argc);
    SortParametersArray(&pairs);
    PrintParametersAsCsv(pairs.begin(), pairs.end(), &std::cout);
}
