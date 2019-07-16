#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <queue>
#include <random>
#include <thread>
#include <unordered_map>

#include <gtest/gtest.h>

#include "observables.h"
#include "progress.h"
#include "tensor.h"
#include "next_configuration.h"
#include "throttle.h"
#include "timer.h"
#include "udh_arguments.h"
#include "udh_io.h"
#include "udh_measure.h"
#include "udh_metropolis_algorithm.h"
#include "compute_distribution_p_value.h"
#include "udh_spin.h"
#include "wolff_algorithm.h"
#include "distributions.h"




template <size_t nDim>
void TestUdhMeasureRandomConfiguration(const Index<nDim> &shape) {
    std::mt19937 rng;
    std::uniform_int_distribution<uint8_t> random_spin(0, 2);
    Tensor<nDim, UdhSpin> lattice(shape, UdhSpinDown());
    for (uint64_t i = 0; i < 8; ++i) {
        for (UdhSpin &node : lattice) {
            node.value = random_spin(rng);
        }
        UdhObservables observables;
        Measure(lattice, &observables);
        const Index<2> energies = ComputeEnergies(lattice);
        EXPECT_EQ(nDim * lattice.size() + observables.sum_si_sj(), energies[0]);
        EXPECT_EQ(observables.n_up() + observables.n_down(), energies[1]);
        EXPECT_GT(observables.n_holes(), 0ul);
        EXPECT_LT(observables.n_holes(), lattice.size());
    }
}

TEST(UdhMeasure, RandomConfiguration) {
    TestUdhMeasureRandomConfiguration<2>({5, 4});
    TestUdhMeasureRandomConfiguration<3>({3, 4, 5});
}

template <size_t nDim>
void TestUdhMeasure(const Index<nDim> &shape,
                    const std::vector<uint64_t> &values, uint64_t n_down,
                    uint64_t n_holes, uint64_t n_up, int64_t sum_si_sj) {
    Tensor<nDim, UdhSpin> lattice(shape, UdhSpinDown());
    ASSERT_EQ(lattice.size(), values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        ASSERT_LT(values[i], 3ul);
        lattice[i].value = values[i];
    }
    UdhObservables observables;
    Measure(lattice, &observables);
    EXPECT_EQ(observables.n_down(), n_down);
    EXPECT_EQ(observables.n_holes(), n_holes);
    EXPECT_EQ(observables.n_up(), n_up);
    EXPECT_EQ(observables.sum_si_sj(), sum_si_sj);
}

TEST(UdhMeasure, ChosenConfigurations) {
    // clang-format off
    TestUdhMeasure<2>({3, 2},
           {1, 1, 1,
            1, 1, 1}, 
            /*n_down=*/0, /*n_holes=*/6, /*n_up=*/0, /*sum_si_sj=*/0);

    TestUdhMeasure<2>({3, 2},
           {1, 0, 1,
            1, 1, 1}, 
            /*n_down=*/1, /*n_holes=*/5, /*n_up=*/0, /*sum_si_sj=*/0);

    TestUdhMeasure<2>({3, 2},
           {1, 0, 0,
            1, 1, 1}, 
            /*n_down=*/2, /*n_holes=*/4, /*n_up=*/0, /*sum_si_sj=*/1);

    TestUdhMeasure<2>({3, 2},
           {1, 0, 2,
            1, 1, 1}, 
            /*n_down=*/1, /*n_holes=*/4, /*n_up=*/1, /*sum_si_sj=*/-1);
    // clang-format on
}

struct MockClock {
    using time_point = int64_t;
    using duration = int64_t;
    static int64_t time;
    static int64_t now() { return time; }
};
int64_t MockClock::time{};

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
