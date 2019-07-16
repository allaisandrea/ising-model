#include "mock_clock.h"
#include "timer.h"

#include <gtest/gtest.h>

namespace {

TEST(Timer, Timer) {
    Timer<MockClock> timer;
    MockClock::time = 5;
    EXPECT_EQ(timer.elapsed(), 0l);
    timer.start();
    MockClock::time = 7;
    EXPECT_EQ(timer.elapsed(), 2l);
    MockClock::time = 10;
    EXPECT_EQ(timer.elapsed(), 5l);
    timer.stop();
    EXPECT_EQ(timer.elapsed(), 5l);
    MockClock::time = 12;
    EXPECT_EQ(timer.elapsed(), 5l);
    timer.start();
    MockClock::time = 15;
    EXPECT_EQ(timer.elapsed(), 8l);
    timer.stop();
    MockClock::time = 19;
    EXPECT_EQ(timer.elapsed(), 8l);
}

} // namespace
