#include "eigen_expect_near.h"
#include "udh_critical_line.h"
#include <gtest/gtest.h>

TEST(UdhCriticalLine, UdhCriticalLine) {
    Eigen::ArrayX3d table(7, 3);
    // clang-format off
    table << -1.0e128, 0.14970, 0.000100,
                0.000, 0.21575, 0.002000,
                1.000, 0.31670, 0.001000,
                1.500, 0.40745, 0.000025,
                1.625, 0.43500, 0.001000,
                1.750, 0.46569, 0.000050,
                2.000, 0.52500, 0.002000;
    // clang_format on

    const auto critical_J = GetCriticalJ(4, table.col(0));

    ASSERT_EQ(critical_J.rows(), table.rows());
    ASSERT_EQ(critical_J.cols(), 2);
    EIGEN_EXPECT_NEAR(critical_J, table.rightCols<2>(), 0.01);
}
