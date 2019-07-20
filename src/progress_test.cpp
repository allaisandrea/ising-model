#include "progress.h"

#include <gtest/gtest.h>
namespace {
TEST(Progress, ProgressString) {
    {
        ProgressIndicator progress(0, 100);
        EXPECT_EQ(progress.string(60, 0), "  0.00%");
        EXPECT_EQ(progress.string(60, 50),
                  " 50.00% ETA Thu Jan  1 00:02:00 1970");
        EXPECT_EQ(progress.string(60, 100),
                  "100.00% ETA Thu Jan  1 00:01:00 1970");
    }
    {
        ProgressIndicator progress(30, 100);
        EXPECT_EQ(progress.string(60, 0), "  0.00%");
        EXPECT_EQ(progress.string(90, 50),
                  " 50.00% ETA Thu Jan  1 00:02:30 1970");
        EXPECT_EQ(progress.string(90, 100),
                  "100.00% ETA Thu Jan  1 00:01:30 1970");
    }
    {
        ProgressIndicator progress(0, 1ul << 32);
        EXPECT_EQ(progress.string(3600, 1),
                  "  0.00% ETA Sun Jul 18 16:00:00     ");
    }
}
} // namespace
