#pragma once
#include "chebyshev.h"
#include <iostream>

Eigen::ArrayX3d GetCriticalPointsTable3D() {
    Eigen::ArrayX3d result(6, 3);
    // clang-format off
    result << -1.0e128, 0.22165, 0.0001,
                  0.00, 0.31288, 0.0001,
                  1.00, 0.44566, 0.0001,
                  1.50, 0.55755, 0.0001,
                  1.75, 0.62635, 0.0001,
                  2.00, 0.70325, 0.0001;
    // clang-format on
    return result;
}

Eigen::ArrayX3d GetCriticalPointsTable4D() {
    Eigen::ArrayX3d result(8, 3);
    // clang-format off
    result << -1.0e128, 0.149700, 0.000100,
                0.0000, 0.215750, 0.002000,
                1.0000, 0.316700, 0.001000,
                1.5000, 0.407480, 0.000010,
                1.6250, 0.435480, 0.000020,
                1.6875, 0.450300, 0.000020,
                1.7500, 0.465715, 0.000010,
                2.0000, 0.530000, 0.005000;
    // clang-format on
    return result;
}

Eigen::ArrayX3d GetCriticalPointsTable(uint64_t n_dim) {
    switch (n_dim) {
    case 3:
        return GetCriticalPointsTable3D();
    case 4:
        return GetCriticalPointsTable4D();
    default:
        throw std::logic_error("Only 3 and 4 dimensions supported");
    };
}

double ChemicalPotentialToFitAbscissa(double mu) {
    const double y = mu - 1.0;
    if (std::abs(y) < 1.0e-7) {
        return y;
    } else {
        return (std::sqrt(1 + 4.0 * y * y) - 1.0) / (2.0 * y);
    }
}

template <typename Derived>
Eigen::ArrayXd
ChemicalPotentialToFitAbscissa(const Eigen::ArrayBase<Derived> &mu) {
    Eigen::ArrayXd result(mu.rows(), mu.cols());
    for (int i = 0; i < mu.size(); ++i) {
        result(i) = ChemicalPotentialToFitAbscissa(mu(i));
    }
    return result;
}

std::tuple<Eigen::VectorXd, Eigen::MatrixXd> FitCriticalLine(uint64_t n_dim) {
    const Eigen::ArrayX3d table = GetCriticalPointsTable(n_dim);
    const Eigen::ArrayXd x_fit = ChemicalPotentialToFitAbscissa(table.col(0));
    const Eigen::ArrayXd y_fit = (1.0 - x_fit) * table.col(1);
    const Eigen::ArrayXd sy_fit = (1.0 - x_fit) * table.col(2);
    return FitChebyshevPolynomial(x_fit, y_fit, sy_fit, 50.0, 32);
}

Eigen::ArrayX2d GetCriticalJ(int n_dim, const Eigen::ArrayXd &mu) {
    const auto pair = FitCriticalLine(n_dim);
    const Eigen::VectorXd &c = std::get<0>(pair);
    const Eigen::MatrixXd &sc = std::get<1>(pair);

    const Eigen::ArrayXd x_fit = ChemicalPotentialToFitAbscissa(mu);
    const Eigen::MatrixXd P = EvaluateChebyshevPolynomial(x_fit, c.rows());
    const Eigen::ArrayXd y_fit = (P * c).array();
    const Eigen::ArrayXd sy_fit = (P * sc * P.transpose()).diagonal();

    Eigen::ArrayX2d result(mu.rows(), 2);
    result.col(0) = y_fit / (1.0 - x_fit),
    result.col(1) = sy_fit.sqrt() / (1.0 - x_fit);
    return result;
}

Eigen::Array2d GetCriticalJ(int n_dim, double mu) {
    Eigen::ArrayXd mu_array(1, 1);
    mu_array << mu;
    return GetCriticalJ(n_dim, mu_array).transpose();
}
