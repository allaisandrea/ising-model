#pragma once
#define EIGEN_EXPECT_NEAR(A, B, tolerance)                                     \
    EXPECT_LT(((A) - (B)).array().abs().maxCoeff(), tolerance)                 \
        << "\n" #A ":\n"                                                       \
        << (A) << "\n" #B ":\n"                                                \
        << (B) << "\n" #A " - " #B ":\n"                                       \
        << (A) - (B)
