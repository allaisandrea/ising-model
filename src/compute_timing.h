#pragma once

#include "udh_file_group.h"
#include <Eigen/Core>

inline Eigen::ArrayXd ComputeTiming(uint64_t n_read, UdhFileGroup *file_group) {
    if (n_read == 0) {
        throw std::logic_error("n_read must be greater than zero");
    }
    Eigen::Array<double, 5, 1> mean;
    mean.setZero();

    uint64_t previous_stamp = 0;
    UdhObservables observables;
    for (uint64_t i = 0; i < n_read; ++i) {
        if (!file_group->NextObservables(&observables)) {
            throw std::runtime_error(
                "Reached end of data at i = " + std::to_string(i) + " / " +
                std::to_string(n_read));
        }
        mean(0) += observables.flip_cluster_duration() +
                   observables.clear_flag_duration();
        mean(1) += observables.metropolis_sweep_duration();
        mean(2) += observables.measure_duration();
        if (i > 0) {
            mean(3) += observables.serialize_duration();
            mean(4) += observables.stamp() - previous_stamp -
                       observables.flip_cluster_duration() -
                       observables.clear_flag_duration() -
                       observables.metropolis_sweep_duration() -
                       observables.measure_duration() -
                       observables.serialize_duration();
            previous_stamp = observables.stamp();
        }
    }
    mean.topRows<3>() /= n_read;
    mean.bottomRows<2>() /= n_read - 1;
    mean *= 1.0e-6;

    return mean;
}
