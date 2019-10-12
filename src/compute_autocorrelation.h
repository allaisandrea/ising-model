#pragma once

#include "udh_file_group.h"
#include <Eigen/Core>
#include <array>

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
    auto circular_next = [](uint64_t i, uint64_t n) {
        ++i;
        if (i >= n) {
            i = 0;
        }
        return i;
    };

    constexpr int hist_length = 33;
    Eigen::Array<double, 3, hist_length> delta;
    std::array<uint64_t, hist_length> file_index;
    Eigen::Array<double, 3, hist_length> autocovariance =
        Eigen::Array<double, 3, hist_length>::Zero();
    std::array<uint64_t, hist_length> count{};
    uint64_t hist_begin = 0;
    uint64_t hist_end = 0;
    for (uint64_t i = 0; i < n_read; ++i) {
        if (!file_group->NextObservables(&observables,
                                         &file_index.at(hist_end))) {
            throw std::runtime_error(
                "Reached end of data at i = " + std::to_string(i) + " / " +
                std::to_string(n_read));
        }

        delta(0, hist_end) = observables.n_down() - mean(0);
        delta(1, hist_end) = observables.n_holes() - mean(1);
        delta(2, hist_end) = observables.n_up() - mean(2);
        const uint64_t hist_last = hist_end;
        hist_end = circular_next(hist_end, hist_length);
        if (hist_end == hist_begin) {
            hist_begin = circular_next(hist_begin, hist_length);
        }
        for (uint64_t j = hist_begin; j != hist_end;
             j = circular_next(j, hist_length)) {
            const uint64_t k = (hist_last + hist_length - j) % hist_length;
            if (file_index.at(k) == file_index.at(hist_last)) {
                autocovariance.col(k) += delta.col(j) * delta.col(hist_last);
                ++count.at(k);
            }
        }
    }
    const int n_result = autocovariance.cols() - 1;
    Eigen::ArrayXd result(2 * n_result, 1);
    Eigen::Map<Eigen::ArrayXXd> autocorrelation(result.data(), 2, n_result);
    for (int i = 0; i < n_result; ++i) {
        autocorrelation(0, i) =
            0.5 * (autocovariance(0, i) / autocovariance(0, 0) +
                   autocovariance(2, i) / autocovariance(2, 0));

        autocorrelation(1, i) = autocovariance(1, i) / autocovariance(1, 0);
    }
    return result;
}
