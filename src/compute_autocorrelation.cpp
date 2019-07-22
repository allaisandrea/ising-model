#include "compute_autocorrelation.h"
#include "cross_validate.h"
#include "udh_file_group.h"

int main(int argc, char **argv) {
    auto pairs = ReadUdhParametersFromFiles(argv + 1, argv + argc);
    std::vector<UdhFileGroup> file_groups = GroupFiles(pairs, 3);
    for (auto &group : file_groups) {
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
        std::cout << stats.mean << "\n\n" << stats.std_dev << "\n";
    }
}
