#pragma once

#include "udh_file_group.h"
#include <Eigen/Core>

Eigen::Matrix3d ComputeAutocorrelation(uint64_t n_read,
                                       UdhFileGroup *file_group) {
    const UdhFileGroup::Position position = file_group->GetPosition();
    Eigen::Array3d mean;
    mean.setZero();

    UdhObservables observables;
    for (uint64_t i = 0; i < n_read; ++i) {
        if (!file_group->NextObservables(&observables)) {
            throw std::runtime_error("Reached end of data");
        }
        mean(0) += observables.n_down();
        mean(1) += observables.n_holes();
        mean(2) += observables.n_up();
    }
    mean /= n_read;

    Eigen::Array3d var;
    Eigen::Matrix3d cov;
    var.setZero();
    cov.setZero();
    uint64_t cov_count = 0;

    file_group->SetPosition(position);

    Eigen::Array3d prev_delta;
    bool same_file;
    for (uint64_t i = 0; i < n_read; ++i) {
        if (!file_group->NextObservables(&observables, &same_file)) {
            throw std::runtime_error("Reached end of data");
        }
        Eigen::Array3d delta;
        delta << observables.n_down(), observables.n_holes(),
            observables.n_up();
        delta -= mean;
        var += delta.square();
        if (i > 0 && same_file) {
            cov += prev_delta.matrix() * delta.matrix().transpose();
            ++cov_count;
        }
        prev_delta = delta;
    }
    var /= n_read;
    cov /= cov_count;

    for (int j = 0; j < cov.cols(); ++j) {
        for (int i = 0; i < cov.rows(); ++i) {
            cov(i, j) /= std::sqrt(var(i) * var(j));
        }
    }
    return cov;
}
