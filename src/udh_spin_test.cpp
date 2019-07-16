#include "udh_spin.h"

#include <gtest/gtest.h>

namespace {

TEST(UdhSpin, VisitedFlag) {
    for (UdhSpin s0{0}; s0.value < 3; ++s0.value) {
        UdhSpin s1 = s0;
        EXPECT_FALSE(Visited(s1)) << "s1: " << std::bitset<8>(s1.value)
                                  << " s0: " << std::bitset<8>(s0.value);
        MarkVisited(&s1);
        EXPECT_TRUE(Visited(s1)) << "s1: " << std::bitset<8>(s1.value)
                                 << " s0: " << std::bitset<8>(s0.value);
        ClearVisitedFlag(&s1);
        EXPECT_FALSE(Visited(s1)) << "s1: " << std::bitset<8>(s1.value)
                                  << " s0: " << std::bitset<8>(s0.value);
        EXPECT_EQ(s0, s1) << "s1: " << std::bitset<8>(s1.value)
                          << " s0: " << std::bitset<8>(s0.value);
    }
}

void TestMaskedEqual(UdhSpin sd, UdhSpin sh, UdhSpin su) {
    EXPECT_TRUE(MaskedEqual(sd, sd));
    EXPECT_TRUE(MaskedEqual(sh, sh));
    EXPECT_TRUE(MaskedEqual(su, su));
    EXPECT_FALSE(MaskedEqual(sd, sh));
    EXPECT_FALSE(MaskedEqual(sd, su));
    EXPECT_FALSE(MaskedEqual(sh, sd));
    EXPECT_FALSE(MaskedEqual(sh, su));
    EXPECT_FALSE(MaskedEqual(su, sd));
    EXPECT_FALSE(MaskedEqual(su, sh));
}

TEST(UdhSpin, MaskedEqual) {
    UdhSpin sd = UdhSpinDown();
    UdhSpin sh = UdhSpinHole();
    UdhSpin su = UdhSpinUp();
    TestMaskedEqual(sd, sh, su);
    MarkVisited(&sd);
    TestMaskedEqual(sd, sh, su);
    MarkVisited(&sd);
    MarkVisited(&sh);
    TestMaskedEqual(sd, sh, su);
    MarkVisited(&sh);
    MarkVisited(&su);
    TestMaskedEqual(sd, sh, su);
}

TEST(UdhSpin, Flip) {
    UdhSpin s = UdhSpinDown();
    Flip(&s);
    EXPECT_EQ(s, UdhSpinUp());
    Flip(&s);
    EXPECT_EQ(s, UdhSpinDown());
}

TEST(UdhSpin, Increment) {
    UdhSpin s = UdhSpinDown();
    ++s.value;
    EXPECT_EQ(s, UdhSpinHole());
    ++s.value;
    EXPECT_EQ(s, UdhSpinUp());
    --s.value;
    EXPECT_EQ(s, UdhSpinHole());
    --s.value;
    EXPECT_EQ(s, UdhSpinDown());
}
}
