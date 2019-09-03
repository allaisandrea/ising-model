#pragma once
#include <Eigen/Dense>

template <typename Derived>
Eigen::MatrixXd EvaluateChebyshevPolynomial(const Eigen::ArrayBase<Derived> &x,
                                            int n) {
    Eigen::MatrixXd P(x.rows(), n);
    if (n > 0) {
        P.col(0).setOnes();
    }
    if (n > 1) {
        P.col(1) = x;
    }
    for (int i = 2; i < n; ++i) {
        P.col(i) = 2.0 * x * P.col(i - 1).array() - P.col(i - 2).array();
    }
    return P;
}

template <typename Derived>
Eigen::VectorXd FitChebyshevPolynomial(const Eigen::ArrayBase<Derived> &x,
                                       const Eigen::ArrayBase<Derived> &y,
                                       int n) {
    const Eigen::MatrixXd P = EvaluateChebyshevPolynomial(x, n);
    return P.fullPivHouseholderQr().solve(y.matrix());
}
