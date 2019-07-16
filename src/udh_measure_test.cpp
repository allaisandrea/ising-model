#include "udh_measure.h"
#include "udh_metropolis_algorithm.h"

#include <gtest/gtest.h>

namespace {

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

} // namespace
