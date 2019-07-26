#pragma once
#include "Eigen/Core"
#include "udh_file_group.h"

struct CrossValidationStats {
    Eigen::ArrayXd mean;
    Eigen::ArrayXd std_dev;
};

using CrossValidationFunction =
    std::function<Eigen::ArrayXd(uint64_t, UdhFileGroup *)>;

inline CrossValidationStats CrossValidate(uint64_t n_batches,
                                          CrossValidationFunction f,
                                          UdhFileGroup *file_group) {
    const uint64_t n_observables = file_group->CountObservables();
    const uint64_t n_batch = n_observables / n_batches;
    CrossValidationStats result;
    const UdhFileGroup::Position position = file_group->GetPosition();
    result.mean = f(n_observables, file_group);
    UdhObservables observables;
    Eigen::ArrayXXd means(result.mean.rows(), n_batches);
    file_group->SetPosition(position);
    for (uint64_t i = 0; i < n_batches; ++i) {
        means.col(i) =
            f(std::min(n_batch, n_observables - i * n_batch), file_group);
    }
    means = means.colwise() - result.mean;
    result.std_dev =
        means.square().rowwise().sum() / (n_batches - 1) / n_batches;
    result.std_dev = result.std_dev.sqrt();
    return result;
}
