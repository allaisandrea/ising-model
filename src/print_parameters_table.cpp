#include "udh_parameters_set.h"

int main(int argc, char **argv) {
    const ParametersSet parameters_set =
        ParametersSetFromFileNames(argv + 1, argv + argc);
    PrintParametersAsCsv(parameters_set.begin(), parameters_set.end(),
                         &std::cout);
}
