#include "compute_autocorrelation.h"
#include "compute_timing.h"
#include "cross_validate.h"
#include "error_format.h"
#include "udh_file_group.h"
#include <cmath>

int main(int argc, char **argv) {
    auto pairs = ReadUdhParametersFromFiles(argv + 1, argv + argc);
    std::vector<UdhFileGroup> file_groups = GroupFiles(pairs, 3);
    std::cout << "group,file_name,J,mu,L0,n_wolff,n_metropolis,measure_every,"
                 "ud_ac,ud_ac_std,hole_ac,hole_ac_std,t_wolff,t_metropolis,t_"
                 "measure,t_serialize,t_residual\n";
    for (uint64_t i = 0; i < file_groups.size(); ++i) {
        auto &group = file_groups.at(i);
        const CrossValidationStats autocorrelation = CrossValidate(
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
        const CrossValidationStats timing = CrossValidate(
            /*n_batches=*/16, &ComputeTiming, &group);
        const double ud_ac =
            0.5 * (autocorrelation.mean(0) + autocorrelation.mean(8));
        const double ud_ac_std =
            std::sqrt(0.5 * (std::pow(autocorrelation.std_dev(0), 2) +
                             std::pow(autocorrelation.std_dev(8), 2)));
        for (const auto &entry : group.entries()) {
            std::cout << i << "," << entry.file_name << ","
                      << group.parameters().j() << ","
                      << group.parameters().mu() << ","
                      << group.parameters().shape(0) << ","
                      << group.parameters().n_wolff() << ","
                      << group.parameters().n_metropolis() << ","
                      << group.parameters().measure_every() << "," << ud_ac
                      << "," << ud_ac_std << "," << autocorrelation.mean(4)
                      << "," << autocorrelation.std_dev(4) << ","
                      << timing.mean(0) << "," << timing.mean(1) << ","
                      << timing.mean(2) << "," << timing.mean(3) << ","
                      << timing.mean(4) << "\n";
        }
    }
}
