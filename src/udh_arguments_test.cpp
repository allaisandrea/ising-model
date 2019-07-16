
#include "udh_arguments.h"

#include <gtest/gtest.h>
namespace {
TEST(UdhArguments, UdhArguments) {
    UdhParameters parameters;
    const char *args1[] = {"run-simulation", "--help"};
    EXPECT_FALSE(ParseArgs(2, args1, &parameters));

    // clang-format off
    const char *args2[] = { "run-simulation",
        "--J", "0.5",
        "--mu", "0.25",
        "--shape", "8", "16",
        "--n-wolff", "5",
        "--n-metropolis", "1",
        "--measure-every", "100",
        "--n-measure", "200"};
    // clang-format on

    EXPECT_TRUE(ParseArgs(16, args2, &parameters));

    EXPECT_EQ(parameters.j(), 0.5);
    EXPECT_EQ(parameters.mu(), 0.25);
    EXPECT_EQ(parameters.shape().size(), 2);
    EXPECT_EQ(parameters.shape(0), 8ul);
    EXPECT_EQ(parameters.shape(1), 16ul);
    EXPECT_EQ(parameters.n_wolff(), 5ul);
    EXPECT_EQ(parameters.n_metropolis(), 1ul);
    EXPECT_EQ(parameters.measure_every(), 100ul);
    EXPECT_EQ(parameters.n_measure(), 200ul);
    EXPECT_GT(parameters.seed(), 0ul);
    EXPECT_NE(parameters.id(), "");
    EXPECT_EQ(parameters.tag(), "");

    // clang-format off
    const char *args3[] = { "run-simulation",
        "--J", "0.5",
        "--mu", "0.25",
        "--shape", "8", "16",
        "--n-wolff", "1",
        "--n-metropolis", "6",
        "--measure-every", "100",
        "--n-measure", "200",
        "--seed", "7",
        "--id", "5ab4fd9aa",
        "--tag", "foo"};
    // clang-format on

    EXPECT_TRUE(ParseArgs(22, args3, &parameters));

    EXPECT_EQ(parameters.j(), 0.5);
    EXPECT_EQ(parameters.mu(), 0.25);
    EXPECT_EQ(parameters.shape().size(), 2);
    EXPECT_EQ(parameters.shape(0), 8l);
    EXPECT_EQ(parameters.shape(1), 16ul);
    EXPECT_EQ(parameters.n_wolff(), 1ul);
    EXPECT_EQ(parameters.n_metropolis(), 6ul);
    EXPECT_EQ(parameters.measure_every(), 100ul);
    EXPECT_EQ(parameters.n_measure(), 200ul);
    EXPECT_EQ(parameters.seed(), 7ul);
    EXPECT_EQ(parameters.id(), "5ab4fd9aa");
    EXPECT_EQ(parameters.tag(), "foo");
}
}
