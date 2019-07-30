#pragma once

#include "udh_file_group.h"
#include <Eigen/Core>

inline Eigen::ArrayXd ComputeAutocorrelation(uint64_t n_read,
                                             UdhFileGroup *file_group) {
    if (n_read == 0) {
        throw std::logic_error("n_read must be greater than zero");
    }
    const UdhFileGroup::Position position = file_group->GetPosition();
    Eigen::Array3d mean;
    mean.setZero();

    UdhObservables observables;
    for (uint64_t i = 0; i < n_read; ++i) {
        if (!file_group->NextObservables(&observables)) {
            throw std::runtime_error(
                "Reached end of data at i = " + std::to_string(i) + " / " +
                std::to_string(n_read));
        }
        mean(0) += observables.n_down();
        mean(1) += observables.n_holes();
        mean(2) += observables.n_up();
    }
    mean /= n_read;

    Eigen::Array3d var;
    Eigen::Array3d cov;
    var.setZero();
    cov.setZero();

    file_group->SetPosition(position);

    bool same_file;
    Eigen::Array3d prev_delta;
    uint64_t cov_count = 0;
    uint64_t prev_stamp = 0;
    double mean_time = 0.0;
    for (uint64_t i = 0; i < n_read; ++i) {
        if (!file_group->NextObservables(&observables, &same_file)) {
            throw std::runtime_error(
                "Reached end of data at i = " + std::to_string(i) + " / " +
                std::to_string(n_read));
        }
        Eigen::Array3d delta;
        delta << observables.n_down(), observables.n_holes(),
            observables.n_up();
        delta -= mean;
        var += delta.square();
        if (i > 0 && same_file) {
            cov += prev_delta * delta;
            ++cov_count;
            mean_time += (observables.stamp() - prev_stamp) * 1.0e-6;
        }
        prev_delta = delta;
        prev_stamp = observables.stamp();
    }
    if (cov_count == 0) {
        throw std::logic_error("cov_count must be greater than zero");
    }
    var /= n_read;
    cov /= cov_count;
    mean_time /= cov_count;

    Eigen::ArrayXd result(4, 1);
    result(0) = (cov(0) + cov(2)) / (var(0) + var(2));
    result(1) = cov(1) / var(1);
    result(2) = -std::log(std::abs(result(0))) / mean_time;
    result(3) = -std::log(std::abs(result(1))) / mean_time;
    return result;
}
