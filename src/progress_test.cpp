#include "progress.h"

#include <gtest/gtest.h>
namespace {
TEST(Progress, ProgressString) {
    {
        ProgressIndicator progress(0, 100);
        EXPECT_EQ(progress.string(60, 0), "  0.00%");
        EXPECT_EQ(progress.string(60, 50),
                  " 50.00% ETA Wed Dec 31 16:02:00 1969");
        EXPECT_EQ(progress.string(60, 100),
                  "100.00% ETA Wed Dec 31 16:01:00 1969");
    }
    {
        ProgressIndicator progress(30, 100);
        EXPECT_EQ(progress.string(60, 0), "  0.00%");
        EXPECT_EQ(progress.string(90, 50),
                  " 50.00% ETA Wed Dec 31 16:02:30 1969");
        EXPECT_EQ(progress.string(90, 100),
                  "100.00% ETA Wed Dec 31 16:01:30 1969");
    }
    {
        ProgressIndicator progress(0, 1ul << 32);
        EXPECT_EQ(progress.string(3600, 1),
                  "  0.00% ETA Sun Jul 18 09:00:00     ");
    }
}
} // namespace
