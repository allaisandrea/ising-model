#include "udh_io.h"
#include <fstream>

int main(int argc, char **argv) {
    if (argc < 2) {
        throw std::runtime_error("Missing file name argument");
    }
    const std::string file_name = argv[1];
    std::ifstream in_file(file_name);
    if (!in_file.good()) {
        throw std::runtime_error("Unable to open file \"" +
                                 std::string(file_name) + "\"");
    }

    udh::Parameters parameters;
    if (!Read(&parameters, &in_file)) {
        throw std::runtime_error("Unable to read parameters from file \"" +
                                 std::string(file_name) + "\"");
    }

    PrintCsvHeader<udh::Observables>(&std::cout);
    std::cout << std::endl;
    udh::Observables observables;
    while (Read(&observables, &in_file)) {
        PrintAsCsv(observables, &std::cout);
        std::cout << std::endl;
    }
}
