#include "error_format.h"

#include <gtest/gtest.h>
namespace {

TEST(ErrorFormat, ErrorFormat) {
    EXPECT_EQ(ErrorFormat(1.2275e-4, 0.023e-4), "1.23(2)E-4");
    EXPECT_EQ(ErrorFormat(1.2275e-3, 0.023e-3), "1.23(2)E-3");
    EXPECT_EQ(ErrorFormat(1.2275e-2, 0.023e-2), "1.23(2)E-2");
    EXPECT_EQ(ErrorFormat(1.2275e-1, 0.023e-1), "1.23(2)E-1");
    EXPECT_EQ(ErrorFormat(1.2275e0, 0.023e0), "1.23(2)E0");
    EXPECT_EQ(ErrorFormat(1.2275e1, 0.023e1), "1.23(2)E1");
    EXPECT_EQ(ErrorFormat(1.2275e2, 0.023e2), "1.23(2)E2");
    EXPECT_EQ(ErrorFormat(1.2275e3, 0.023e3), "1.23(2)E3");
    EXPECT_EQ(ErrorFormat(1.2275e4, 0.023e4), "1.23(2)E4");

    EXPECT_EQ(ErrorFormat(9.3712e-4, 0.0089e-4), "9.371(9)E-4");
    EXPECT_EQ(ErrorFormat(9.3712e-3, 0.0089e-3), "9.371(9)E-3");
    EXPECT_EQ(ErrorFormat(9.3712e-2, 0.0089e-2), "9.371(9)E-2");
    EXPECT_EQ(ErrorFormat(9.3712e-1, 0.0089e-1), "9.371(9)E-1");
    EXPECT_EQ(ErrorFormat(9.3712e0, 0.0089e0), "9.371(9)E0");
    EXPECT_EQ(ErrorFormat(9.3712e1, 0.0089e1), "9.371(9)E1");
    EXPECT_EQ(ErrorFormat(9.3712e2, 0.0089e2), "9.371(9)E2");
    EXPECT_EQ(ErrorFormat(9.3712e3, 0.0089e3), "9.371(9)E3");
    EXPECT_EQ(ErrorFormat(9.3712e4, 0.0089e4), "9.371(9)E4");

    EXPECT_EQ(ErrorFormat(0.000479427, 0.0194104), "0(2)E-2");
}
} // namespace
