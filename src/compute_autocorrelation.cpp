#include "udh_file_group.h"

int main(int argc, char **argv) {
    const std::vector<UdhFileGroup> file_groups =
        GroupFiles(argv + 1, argv + argc);
    for (const auto &group : file_groups) {
        PrintAsCsv(group.parameters(), &std::cout);
        std::cout << std::endl;
    }
}
