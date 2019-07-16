#include "udh_metropolis_algorithm.h"
#include "wolff_algorithm.h"
#include "compute_distribution_p_value.h"
#include "next_configuration.h"

#include <gtest/gtest.h>

uint64_t get_p_hu(const UdhTransitionProbs &tp) {
    return tp.p_hd_plus_p_hu - tp.p_hd;
}

TEST(UdhMetropolis, TransitionProbabilities) {
    std::mt19937 rng;
    std::uniform_real_distribution<double> random_uniform(0.0, 1.0);
    for (uint64_t i = 0; i < 8; ++i) {
        const double p_d = random_uniform(rng);
        const double p_h = random_uniform(rng);
        const double p_u = random_uniform(rng);
        const UdhTransitionProbs tp = ComputeUdhTransitionProbs(p_d, p_h, p_u);
        EXPECT_GT(tp.p_dh_or_uh, 0ul);
        EXPECT_GT(tp.p_hd, 0ul);
        EXPECT_LT(tp.p_hd, 1ul << 32);
        EXPECT_LE(tp.p_hd, tp.p_hd_plus_p_hu);
        EXPECT_GT(tp.p_hd_plus_p_hu, 0ul);
        EXPECT_GT(tp.p_hd_plus_p_hu, tp.p_hd);
        EXPECT_NEAR(p_d * tp.p_dh_or_uh, p_h * tp.p_hd, 2.0);
        EXPECT_NEAR(p_u * tp.p_dh_or_uh, p_h * get_p_hu(tp), 2.0);
    }
}


template <size_t nDim>
Index<2> GetHistogramShape(const Index<nDim> &lattice_shape) {
    return {
        typename Index<nDim>::value_type(2 * nDim * GetSize(lattice_shape) + 1),
        typename Index<nDim>::value_type(GetSize(lattice_shape) + 1)};
}

template <size_t nDim>
Tensor<2, uint64_t> ComputeEntropyHistogramUdh(const Index<nDim> &shape) {
    Tensor<nDim, UdhSpin> lattice(shape, UdhSpinDown());
    Tensor<2, uint64_t> histogram(GetHistogramShape(shape), 0);
    do {
        const Index<2> energies = ComputeEnergies(lattice);
        EXPECT_TRUE(IndexIsValid(energies, histogram.shape()));
        ++histogram[energies];
    } while (NextConfiguration(lattice.begin(), lattice.end(), 0, 2));
    return histogram;
}

template <size_t nDim>
Tensor<2, uint64_t> ComputeVisitHistogramUdh(const Index<nDim> &shape, double J,
                                             double mu, uint64_t nMeasure,
                                             uint64_t measureEvery) {

    std::mt19937 rng;
    Tensor<nDim, UdhSpin> lattice(shape, UdhSpinDown());
    Tensor<2, uint64_t> histogram(GetHistogramShape(shape), 0);
    const UdhTransitionProbsArray<nDim> transition_probs_array =
        ComputeUdhTransitionProbs<nDim>(J, mu);
    const uint32_t p_no_add = GetNoAddProbabilityFromJ(J);

    std::queue<Index<nDim>> queue;
    for (size_t iStep0 = 0; iStep0 < nMeasure; ++iStep0) {
        for (size_t iStep1 = 0; iStep1 < measureEvery; ++iStep1) {
            UdhMetropolisSweep(transition_probs_array, &lattice, &rng);
            const Index<nDim> i0 = GetRandomIndex(shape, &rng);
            if (lattice[i0] != UdhSpinHole()) {
                FlipCluster(p_no_add, i0, &lattice, &rng, &queue);
                ClearVisitedFlag(i0, &lattice, &queue);
            }
        }
        const Index<2> energies = ComputeEnergies(lattice);
        EXPECT_TRUE(IndexIsValid(energies, histogram.shape()));
        ++histogram[energies];
    }
    return histogram;
}

Tensor<2, double> ComputeExpectedVisitingProbability(
    double J, double mu, const Tensor<2, uint64_t> &entropy_histogram) {

    Tensor<2, double> expected_probability(entropy_histogram.shape(), 0.0);
    Index<2> i{};
    do {
        expected_probability[i] =
            entropy_histogram[i] * std::exp(J * i[0] - mu * i[1]);

    } while (NextIndex(&i, expected_probability.shape()));

    double total_probability = std::accumulate(expected_probability.begin(),
                                               expected_probability.end(), 0.0);
    for (double &p : expected_probability) {
        p /= total_probability;
    }
    return expected_probability;
}

std::string MakeDebugString(const Tensor<2, uint64_t> &entropy_hist,
                            const Tensor<2, uint64_t> &visit_hist,
                            const Tensor<2, double> &expected_probability) {
    std::ostringstream strm;
    strm << std::setw(12) << "si sj" << std::setw(12) << "si si"
         << std::setw(12) << "entropy" << std::setw(12) << "exp. prob."
         << std::setw(12) << "visit prob." << std::setw(12) << "z" << std::endl;
    const uint64_t n_measure =
        std::accumulate(visit_hist.begin(), visit_hist.end(), 0);
    Index<2> i{};
    do {
        if (visit_hist[i] == 0 && entropy_hist[i] == 0) {
            continue;
        }
        const double lambda = entropy_hist[i] * expected_probability[i];
        const double pValue = PoissonExactTest(visit_hist[i], lambda);
        const double z =
            -2.0 *
            std::log(std::max(pValue, std::numeric_limits<double>::min()));
        strm << std::setw(12) << i[0] << std::setw(12) << i[1] << std::setw(12)
             << entropy_hist[i] << std::setw(12) << expected_probability[i]
             << std::setw(12) << double(visit_hist[i]) / n_measure
             << std::setw(12) << z << std::endl;
    } while (NextIndex(&i, visit_hist.shape()));

    return strm.str();
}

template <size_t nDim>
void TestUdhMetropolisAlgorithmCorrectDistribution(
    const Index<nDim> &shape, std::array<double, 2> true_params,
    const std::vector<std::array<double, 2>> &counterfactual_params,
    uint64_t n_measure, uint64_t measure_every) {

    const Tensor<2, uint64_t> entropy_hist = ComputeEntropyHistogramUdh(shape);
    const Tensor<2, uint64_t> visit_hist = ComputeVisitHistogramUdh(
        shape, true_params[0], true_params[1], n_measure, measure_every);
    const Tensor<2, double> expected_probability =
        ComputeExpectedVisitingProbability(true_params[0], true_params[1],
                                           entropy_hist);

    EXPECT_GT(ComputeDistributionPValue(expected_probability, visit_hist), 0.05)
        << std::endl
        << "True params: J " << true_params[0] << " mu " << true_params[1]
        << std::endl
        << MakeDebugString(entropy_hist, visit_hist, expected_probability);
    for (const std::array<double, 2> &params : counterfactual_params) {
        const Tensor<2, double> expected_probability =
            ComputeExpectedVisitingProbability(params[0], params[1],
                                               entropy_hist);
        EXPECT_LT(ComputeDistributionPValue(expected_probability, visit_hist),
                  0.01)
            << std::endl
            << "Counterfactual params: J " << params[0] << " mu " << params[1]
            << std::endl
            << MakeDebugString(entropy_hist, visit_hist, expected_probability);
    }
}

TEST(UdhMetropolisAlgorithm, CorrectDistribution3D) {
    const double J = 0.3;
    const double mu = -0.5;
    TestUdhMetropolisAlgorithmCorrectDistribution<3>(
        {2, 2, 2}, {J, mu},
        {{J + 0.01, mu}, {J - 0.01, mu}, {J, mu + 0.10}, {J, mu - 0.10}},
        1 << 17, 2);
}

TEST(UdhMetropolisAlgorithm, CorrectDistribution2D) {
    const double J = 0.3;
    const double mu = -0.5;
    TestUdhMetropolisAlgorithmCorrectDistribution<2>(
        {2, 3}, {J, mu},
        {{J + 0.01, mu}, {J - 0.01, mu}, {J, mu + 0.10}, {J, mu - 0.10}},
        1 << 17, 2);
}
