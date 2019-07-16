#include "wolff_algorithm.h"
#include "ud_spin.h"
#include "distributions.h"
#include "next_configuration.h"
#include "compute_distribution_p_value.h"

#include <gtest/gtest.h>

namespace {
template <size_t nDim>
uint64_t ComputeParallelCount(const Tensor<nDim, UdSpin> &lattice) {
    uint64_t parallelCount = 0;
    Index<nDim> i{};
    do {
        UdSpin spin = lattice[i];
        for (size_t d = 0; d < nDim; ++d) {
            const typename Index<nDim>::value_type i_d = i[d];
            i[d] = (i_d + 1) % lattice.shape(d);
            UdSpin spin1 = lattice[i];
            if (MaskedEqual(spin, spin1)) {
                ++parallelCount;
            }
            i[d] = i_d;
        }
    } while (NextIndex(&i, lattice.shape()));
    return parallelCount;
}

template <size_t nDim>
uint64_t GetMaxParallelCount(const Tensor<nDim, UdSpin> &lattice) {
    return nDim * lattice.size();
}

template <size_t nDim>
std::vector<uint64_t> ComputeEntropyHistogram(const Index<nDim> &shape) {
    Tensor<nDim, UdSpin> lattice(shape, UdSpinDown());
    std::vector<uint64_t> histogram(GetMaxParallelCount(lattice) + 1, 0);
    do {
        const uint64_t n_parallel = ComputeParallelCount(lattice);
        ++histogram.at(n_parallel);
    } while (NextConfiguration(lattice.begin(), lattice.end(), 0, 1));
    return histogram;
}

template <size_t nDim>
std::vector<uint64_t> ComputeVisitHistogram(Index<nDim> shape, double prob,
                                            uint64_t nMeasure,
                                            uint64_t measureEvery) {
    const uint64_t p_no_add = std::round((1ul << 32) * prob);
    std::mt19937 rng;
    Tensor<nDim, UdSpin> lattice(shape, UdSpinDown());
    std::queue<Index<nDim>> queue;
    const uint64_t max_n_parallel = GetMaxParallelCount(lattice);
    std::vector<uint64_t> histogram(max_n_parallel + 1, 0);

    for (size_t iStep0 = 0; iStep0 < nMeasure; ++iStep0) {
        for (size_t iStep1 = 0; iStep1 < measureEvery; ++iStep1) {
            const Index<nDim> i0 = GetRandomIndex(shape, &rng);
            FlipCluster(p_no_add, i0, &lattice, &rng, &queue);
            ClearVisitedFlag(i0, &lattice, &queue);
        }
        const uint64_t n_parallel = ComputeParallelCount(lattice);
        ++histogram.at(n_parallel);
    }
    return histogram;
}

std::vector<double> ComputeExpectedVisitingProbability(
    double prob, const std::vector<uint64_t> &entropy_histogram) {
    const uint64_t max_n_parallel = entropy_histogram.size() - 1;
    std::vector<double> expected_probability(entropy_histogram.size());
    for (size_t n_parallel = 0; n_parallel <= max_n_parallel; ++n_parallel) {
        expected_probability[n_parallel] =
            entropy_histogram[n_parallel] *
            std::pow(prob, double(max_n_parallel - n_parallel));
    }

    double total_probability = std::accumulate(expected_probability.begin(),
                                               expected_probability.end(), 0.0);
    for (double &p : expected_probability) {
        p /= total_probability;
    }
    return expected_probability;
}

template <size_t nDim>
void TestWolffAlgorithmCorrectDistribution(
    const Index<nDim> &shape, double true_prob,
    const std::vector<double> &counterfactual_probs, uint64_t n_measure,
    uint64_t measure_every) {

    const std::vector<uint64_t> entropy_hist = ComputeEntropyHistogram(shape);
    const std::vector<uint64_t> visit_hist =
        ComputeVisitHistogram(shape, true_prob, n_measure, measure_every);
    const std::vector<double> expected_probability =
        ComputeExpectedVisitingProbability(true_prob, entropy_hist);
    EXPECT_GT(ComputeDistributionPValue(expected_probability, visit_hist),
              0.05);
    for (const double prob : counterfactual_probs) {
        const std::vector<double> expected_probability =
            ComputeExpectedVisitingProbability(prob, entropy_hist);
        EXPECT_LT(ComputeDistributionPValue(expected_probability, visit_hist),
                  0.01);
    }
}

TEST(WolffAlgorithm, CorrectDistribution3D) {
    TestWolffAlgorithmCorrectDistribution<3>({2, 2, 2}, 0.642, {0.63, 0.65},
                                             1 << 14, 8);
}

TEST(WolffAlgorithm, CorrectDistribution2D) {
    TestWolffAlgorithmCorrectDistribution<2>({2, 3}, 0.413, {0.40, 0.42},
                                             1 << 14, 8);
}
}
