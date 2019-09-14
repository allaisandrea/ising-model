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

Eigen::MatrixXd GetRegulatorMatrix(int n) {
    Eigen::MatrixXd result(n - 2, n);
    result.setZero();
    for (int i = 0; i < n - 2; ++i) {
        result(i, i + 2) = (i + 1) * (i + 2);
    }
    return result;
}

template <typename Derived>
std::tuple<Eigen::VectorXd, Eigen::MatrixXd>
FitChebyshevPolynomial(const Eigen::ArrayBase<Derived> &x,
                       const Eigen::ArrayBase<Derived> &y,
                       const Eigen::ArrayBase<Derived> &sy, double reg, int n) {
    const Eigen::MatrixXd P = EvaluateChebyshevPolynomial(x, n);
    const Eigen::MatrixXd Q = P.array().colwise() / sy;
    const Eigen::MatrixXd R = reg * GetRegulatorMatrix(n);
    Eigen::MatrixXd A(Q.rows() + R.rows(), n);
    A.topRows(Q.rows()) = Q;
    A.bottomRows(R.rows()) = R;
    Eigen::VectorXd b(Q.rows() + R.rows());
    b.topRows(Q.rows()) = y / sy;
    b.bottomRows(R.rows()).setZero();
    const Eigen::MatrixXd Sinv = A.transpose() * A;
    return {A.fullPivHouseholderQr().solve(b),
            Sinv.fullPivHouseholderQr().inverse()};
}
