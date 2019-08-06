#include "compute_autocorrelation.h"
#include "compute_timing.h"
#include "cross_validate.h"
#include "error_format.h"
#include "udh_file_group.h"
#include <cmath>

int main(int argc, char **argv) {
    auto pairs = ReadUdhParametersFromFiles(argv + 1, argv + argc);
    std::vector<UdhFileGroup> file_groups = GroupFiles(pairs, 3);
    // std::vector<UdhFileGroup> file_groups;
    // for (int i = 1; i < argc; ++i) {
    //    file_groups.emplace_back(UdhFileGroup({{argv[i], 1}}));
    //}
    std::cout << "J,mu,L0,n_wolff,n_metropolis,measure_every,n_measure,"
                 "tau,ud_ac,ud_ac_std,h_ac,h_ac_std,t_wolff,t_"
                 "metropolis,t_measure,t_serialize,t_residual\n";
    for (uint64_t i = 0; i < file_groups.size(); ++i) {
        auto &group = file_groups.at(i);
        const CrossValidationStats autocorrelation = CrossValidate(
            /*n_batches=*/16, &ComputeAutocorrelation, &group);
        const CrossValidationStats timing = CrossValidate(
            /*n_batches=*/16, &ComputeTiming, &group);
        const uint64_t n_measure = group.CountObservables();
        const uint64_t n_tau = autocorrelation.mean.size() / 2;
        for (uint64_t tau = 1; tau < n_tau; ++tau) {
            // clang-format off
            std::cout 
                << group.parameters().j()
                << "," << group.parameters().mu()
                << "," << group.parameters().shape(0) << ","
                << group.parameters().n_wolff() << ","
                << group.parameters().n_metropolis() << ","
                << group.parameters().measure_every() << ","
                << n_measure << ","
                << tau << ","
                << autocorrelation.mean(2 * tau) << ","
                << autocorrelation.std_dev(2 * tau) << ","
                << autocorrelation.mean(2 * tau + 1) << ","
                << autocorrelation.std_dev(2 * tau + 1) << ","
                << timing.mean(0) << ","
                << timing.mean(1) << ","
                << timing.mean(2) << ","
                << timing.mean(3) << ","
                << timing.mean(4) << "\n";
            // clang-format on
        }
    }
}
