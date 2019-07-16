#include "throttle.h"
#include "mock_clock.h"

#include <gtest/gtest.h>

TEST(Throttle, Throttle) {
    Throttle<MockClock> throttle(5);
    uint64_t counter = 0;
    MockClock::time = 31;
    throttle([&counter]() { ++counter; });
    EXPECT_EQ(counter, 1ul);
    throttle([&counter]() { ++counter; });
    EXPECT_EQ(counter, 1ul);
    MockClock::time = 35;
    throttle([&counter]() { ++counter; });
    EXPECT_EQ(counter, 1ul);
    MockClock::time = 36;
    throttle([&counter]() { ++counter; });
    EXPECT_EQ(counter, 2ul);
}
