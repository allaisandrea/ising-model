#include "compute_autocorrelation.h"
#include "cross_validate.h"
#include "error_format.h"
#include "udh_file_group.h"

int main(int argc, char **argv) {
    auto pairs = ReadUdhParametersFromFiles(argv + 1, argv + argc);
    std::vector<UdhFileGroup> file_groups = GroupFiles(pairs, 3);
    std::vector<const char *> row_headers = {"n_d0", "n_h0", "n_u0"};
    std::vector<const char *> col_headers = {"n_d1", "n_h1", "n_u1"};
    for (auto &group : file_groups) {
        std::cout << "J: " << group.parameters().j() << "\n";
        std::cout << "mu: " << group.parameters().mu() << "\n";
        std::cout << "shape: [ ";
        for (const auto &x : group.parameters().shape()) {
            std::cout << x << " ";
        }
        std::cout << "]\n";
        std::cout << "n_wolff: " << group.parameters().n_wolff() << "\n";
        std::cout << "n_metropolis: " << group.parameters().n_metropolis()
                  << "\n";
        std::cout << "measure_every: " << group.parameters().measure_every()
                  << "\n";

        std::cout << "files:\n";
        for (const auto &entry : group.entries()) {
            std::cout << entry.file_name << " " << entry.read_every << "\n";
        }

        std::cout << "n_observables:" << group.CountObservables() << "\n";

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
        std::cout << "autocorrelation:\n";
        std::cout << std::setw(5) << "";
        for (int j = 0; j < 3; ++j) {
            std::cout << std::setw(15) << col_headers[j];
        }
        std::cout << std::endl;
        for (int i = 0; i < 3; ++i) {
            std::cout << std::setw(5) << row_headers[i];
            for (int j = 0; j < 3; ++j) {
                const int k = 3 * j + i;
                std::cout << std::setw(15)
                          << ErrorFormat(stats.mean(k), stats.std_dev(i));
            }
            std::cout << std::endl;
        }
    }
}
