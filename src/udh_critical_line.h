#pragma once
#include "chebyshev.h"

Eigen::VectorXd FitCriticalLine(uint64_t n_dim) {
    if (n_dim == 3) {
        Eigen::ArrayXd mu(3, 1), J(3, 1);
        // clang-format off
        mu <<    0.0,    1.0,    2.0;
        J  << 0.3128, 0.4455, 0.7033;
        // clang-format on
        return FitChebyshevPolynomial(mu, J, 3);
    } else if (n_dim == 4) {
        Eigen::ArrayXd mu(7, 1), J(7, 1);
        // clang-format off
        mu <<  -1.0,     0.0,    1.0,    1.5, 1.625,   1.75,   2.0;
        J  << 0.175, 0.21575, 0.3167, 0.4073, 0.435, 0.4655, 0.525;
        // clang-format on
        return FitChebyshevPolynomial(mu, J, 7);
    } else {
        throw std::runtime_error("Only d = 3 or d = 4");
    }
}

Eigen::ArrayXd GetCriticalJ(int n_dim, const Eigen::ArrayXd &mu) {
    const Eigen::VectorXd c = FitCriticalLine(n_dim);
    return EvaluateChebyshevPolynomial(mu, c.rows()) * c;
}

double GetCriticalJ(int n_dim, double mu) {
    Eigen::ArrayXd mu_array(1, 1);
    mu_array << mu;
    return GetCriticalJ(n_dim, mu_array)(0);
}
