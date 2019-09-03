#include "udh_critical_line.h"
#include <gtest/gtest.h>

TEST(UdhCriticalLine, UdhCriticalLine) {
    EXPECT_NEAR(GetCriticalJ(3, 0.5), 0.363, 0.01);
    EXPECT_NEAR(GetCriticalJ(4, 0.5), 0.270270, 0.01);
}
