#include "chebyshev.h"
#include "eigen_expect_near.h"

#include <gtest/gtest.h>
#include <random>

TEST(Chebyshev, Chebyshev) {
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniform_distribution(-1.0, 1.0);
    std::normal_distribution<double> normal_distribution;
    for (int n = 1; n < 4; ++n) {
        Eigen::ArrayXd x(n, 1);
        for (int i = 0; i < x.rows(); ++i) {
            x(i) = uniform_distribution(rng);
        }
        Eigen::ArrayXd c(n, 1);
        for (int i = 0; i < c.rows(); ++i) {
            c(i) = normal_distribution(rng);
        }

        Eigen::ArrayXd y(x.rows(), 1);
        y.setZero();
        for (int i = 0; i < c.rows(); ++i) {
            y += c(i) * x.pow(i);
        }

        const Eigen::ArrayXd sy = Eigen::ArrayXd::Ones(x.rows(), 1);

        const auto pair = FitChebyshevPolynomial(x, y, sy, n);
        const auto &d = std::get<0>(pair);

        Eigen::ArrayXd x_test;
        x_test.setLinSpaced(32, -1.0, 1.0);

        Eigen::ArrayXd y_expected(x_test.rows(), 1);
        y_expected.setZero();
        for (int i = 0; i < c.rows(); ++i) {
            y_expected += c(i) * x_test.pow(i);
        }

        Eigen::ArrayXd y_actual = EvaluateChebyshevPolynomial(x_test, n) * d;

        EIGEN_EXPECT_NEAR(y_expected, y_actual, 1.0e-14);
    }
}
