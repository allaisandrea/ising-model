#include "compute_autocorrelation.h"
#include "compute_timing.h"
#include "cross_validate.h"
#include "error_format.h"
#include "udh_file_group.h"
#include <cmath>

int main(int argc, char **argv) {
    auto pairs = ReadUdhParametersFromFiles(argv + 1, argv + argc);
    std::vector<UdhFileGroup> file_groups = GroupFiles(pairs, 3);
    std::cout << "J,mu,L0,wolf,metr,every,"
                 "ud_ac,ud_ac_std,h_ac,h_ac_std,ud_log_ac,ud_log_ac_std,"
                 "h_log_ac,h_log_ac_std,t_wolf,t_metr,t_meas,t_ser,t_res\n";
    for (uint64_t i = 0; i < file_groups.size(); ++i) {
        auto &group = file_groups.at(i);
        const CrossValidationStats autocorrelation = CrossValidate(
            /*n_batches=*/16, &ComputeAutocorrelation, &group);
        const CrossValidationStats timing = CrossValidate(
            /*n_batches=*/16, &ComputeTiming, &group);
        std::cout << group.parameters().j() << "," << group.parameters().mu()
                  << "," << group.parameters().shape(0) << ","
                  << group.parameters().n_wolff() << ","
                  << group.parameters().n_metropolis() << ","
                  << group.parameters().measure_every() << ","
                  << autocorrelation.mean(0) << ","
                  << autocorrelation.std_dev(0) << ","
                  << autocorrelation.mean(1) << ","
                  << autocorrelation.std_dev(1) << ","
                  << autocorrelation.mean(2) << ","
                  << autocorrelation.std_dev(2) << ","
                  << autocorrelation.mean(3) << ","
                  << autocorrelation.std_dev(3) << "," << timing.mean(0) << ","
                  << timing.mean(1) << "," << timing.mean(2) << ","
                  << timing.mean(3) << "," << timing.mean(4) << "\n";
    }
}
