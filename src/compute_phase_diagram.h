#pragma once
#include "udh_file_group.h"
#include <Eigen/Core>
#include <array>

UdhObservables GetObservables(UdhFileGroup *file_group) {
    UdhObservables observables;
    if (!file_group->NextObservables(&observables)) {
        throw std::runtime_error("Reached end of data");
    }
    return observables;
}

struct ParameterRange {
    int64_t begin;
    int64_t end;
    double increment;
};

Eigen::ArrayXd ComputePhaseDiagram(const ParameterRange &J_range,
                                   const ParameterRange &mu_range,
                                   uint64_t n_read, UdhFileGroup *file_group) {
    const UdhFileGroup::Position position = file_group->GetPosition();

    const double J0 = file_group->parameters().j();
    const double mu0 = file_group->parameters().mu();
    const int64_t n_J = J_range.end - J_range.begin;
    const int64_t n_mu = mu_range.end - mu_range.begin;

    Eigen::ArrayXXd max_log_weight =
        Eigen::ArrayXXd::Constant(n_J, n_mu, -INFINITY);
    for (uint64_t i = 0; i < n_read; ++i) {
        const UdhObservables observables = GetObservables(file_group);
        const int64_t sum_si_sj = observables.sum_si_sj();
        const int64_t sum_si_si = observables.n_down() + observables.n_up();
        for (int64_t i_J = 0; i_J < n_J; ++i_J) {
            const double J = (J_range.begin + i_J) * J_range.increment;
            for (int64_t i_mu = 0; i_mu < n_mu; ++i_mu) {
                const double mu = (mu_range.begin + i_mu) * mu_range.increment;
                const double log_weight =
                    (J - J0) * sum_si_sj - (mu - mu0) * sum_si_si;
                max_log_weight(i_J, i_mu) =
                    std::max(max_log_weight(i_J, i_mu), log_weight);
            }
        }
    }

    std::array<Eigen::ArrayXXd, 3> moments;
    for (auto &slice : moments) {
        slice.setZero(n_J, n_mu);
    }

    file_group->SetPosition(position);
    for (uint64_t i = 0; i < n_read; ++i) {
        const UdhObservables observables = GetObservables(file_group);
        const int64_t sum_si_sj = observables.sum_si_sj();
        const int64_t n_down = observables.n_down();
        const int64_t n_holes = observables.n_holes();
        const int64_t n_up = observables.n_up();
        const int64_t sum_si_si = n_down + n_up;
        const double phi = double(n_up - n_down) / (n_down + n_holes + n_up);
        const double phi2 = std::pow(phi, 2);
        const double phi4 = std::pow(phi2, 2);
        for (int64_t i_J = 0; i_J < n_J; ++i_J) {
            const double J = (J_range.begin + i_J) * J_range.increment;
            for (int64_t i_mu = 0; i_mu < n_mu; ++i_mu) {
                const double mu = (mu_range.begin + i_mu) * mu_range.increment;
                const double log_weight =
                    (J - J0) * sum_si_sj - (mu - mu0) * sum_si_si;
                const double weight =
                    std::exp(log_weight - max_log_weight(i_J, i_mu));
                moments[0](i_J, i_mu) += weight;
                moments[1](i_J, i_mu) += weight * phi2;
                moments[2](i_J, i_mu) += weight * phi4;
            }
        }
    }
    moments[1] /= moments[0];
    moments[2] /= moments[0];

    Eigen::ArrayXd result(2 * n_J * n_mu);
    Eigen::Map<Eigen::ArrayXXd> chi_view(result.data(), n_J, n_mu);
    Eigen::Map<Eigen::ArrayXXd> u_view(result.data() + n_J * n_mu, n_J, n_mu);
    chi_view = moments[1];
    u_view = 1.0 - moments[2] / (3.0 * moments[1].square());
    return result;
}
