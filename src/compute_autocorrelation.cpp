#include "udh_file_group.h"

int main(int argc, char **argv) {
    std::vector<UdhFileGroup> file_groups =
        GroupFiles(argv + 1, argv + argc, 3);
    for (auto &group : file_groups) {
        std::cout << "Group ";
        PrintAsCsv(group.parameters(), &std::cout);
        std::cout << std::endl;
        PrintCsvHeader<udh::Observables>(&std::cout);
        std::cout << std::endl;
        udh::Observables observables;
        int i = 0;
        while (group.NextObservables(&observables)) {
            std::cout << i << ",";
            PrintAsCsv(observables, &std::cout);
            std::cout << std::endl;
            ++i;
        }
        std::cout << "Counting observables\n";
        const uint64_t n_observables = group.CountObservables();
        std::cout << "nObservables: " << n_observables << std::endl;
    }
}
