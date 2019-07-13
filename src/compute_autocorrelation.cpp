#include "udh_file_group.h"

int main(int argc, char **argv) {
    std::vector<UdhFileGroup> file_groups = GroupFiles(argv + 1, argv + argc);
    for (auto &group : file_groups) {
        std::cout << "Group ";
        PrintAsCsv(group.parameters(), &std::cout);
        std::cout << std::endl;
        PrintCsvHeader<udh::Observables>(&std::cout);
        std::cout << std::endl;
        udh::Observables observables;
        while (group.NextObservables(&observables)) {
            PrintAsCsv(observables, &std::cout);
            std::cout << std::endl;
        }
    }
}
