#include "compute_autocorrelation.h"
#include "cross_validate.h"
#include "error_format.h"
#include "udh_file_group.h"
#include <cmath>

int main(int argc, char **argv) {
    auto pairs = ReadUdhParametersFromFiles(argv + 1, argv + argc);
    std::vector<UdhFileGroup> file_groups = GroupFiles(pairs, 3);
    std::cout << std::fixed << std::setprecision(3);
    // clang-format off
    std::cout << std::setw(3) << "grp" << ","
              << std::setw(20) << "file_name" << ","
              << std::setw(7) << "J" << ","
              << std::setw(7) << "mu" << ","
              << std::setw(3) << "L0" << ","
              << std::setw(5) << "wolff" << ","
              << std::setw(5) << "metrp" << ","
              << std::setw(5) << "every" << ","
              << std::setw(7) << "full" << ","
              << std::setw(7) << "fullstd" << ","
              << std::setw(7) << "hole" << ","
              << std::setw(7) << "holestd" << "\n";
    // clang-format on
    for (uint64_t i = 0; i < file_groups.size(); ++i) {
        auto &group = file_groups.at(i);
        const CrossValidationStats stats = CrossValidate(
            /*n_batches=*/16,
            [](uint64_t n_read, UdhFileGroup *file_group) {
                const Eigen::Matrix3d R =
                    ComputeAutocorrelation(n_read, file_group);
                Eigen::ArrayXd result(9, 1);
                for (int i = 0; i < R.size(); ++i) {
                    result(i) = R(i);
                }
                return result;
            },
            &group);
        for (const auto &entry : group.entries()) {
            // clang-format off
            std::cout << std::setw(3) << i << ","
                      << std::setw(20) << entry.file_name <<","
                      << std::setw(7) << group.parameters().j() <<","
                      << std::setw(7) << group.parameters().mu() <<","
                      << std::setw(3) << group.parameters().shape(0) << ","
                      << std::setw(5) << group.parameters().n_wolff() << ","
                      << std::setw(5) << group.parameters().n_metropolis() << ","
                      << std::setw(5) << group.parameters().measure_every() << ","
                      << std::setw(7) << 0.5 * (stats.mean(0) + stats.mean(8)) << ","
                      << std::setw(7) << std::sqrt(0.5 * (std::pow(stats.std_dev(0), 2) + std::pow(stats.std_dev(8), 2))) << ","
                      << std::setw(7) << stats.mean(4) << ","
                      << std::setw(7) << stats.std_dev(4) << "\n";
            // clang-format on
        }
    }
}
