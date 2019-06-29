#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <queue>
#include <random>
#include <thread>
#include <unordered_map>

#include <boost/math/special_functions/gamma.hpp>
#include <gtest/gtest.h>

#include "UpDownHoleSpin.h"
#include "lattice.h"
#include "observables.h"
#include "wolff_algorithm.h"

double ChiSquaredCdf(double x, double dof) {
    return boost::math::gamma_p(0.5 * dof, x);
}

double PoissonLogPmf(uint64_t k, double lambda) {
    return k * std::log(lambda) - std::lgamma(k + 1) - lambda;
}

double PoissonExactTest(uint64_t n, double lambda) {
    const double log_pn = PoissonLogPmf(n, lambda);
    double result = 0.0;
    double log_pk = 0.0;
    for (uint64_t k = 0; k <= lambda + 3 || log_pk > log_pn; ++k) {
        log_pk = PoissonLogPmf(k, lambda);
        if (log_pk > log_pn) {
            result += std::exp(log_pk);
        }
    }
    return 1.0 - result;
}

template <size_t nDim>
Index<nDim> GetRandomShape(typename Index<nDim>::value_type min,
                           typename Index<nDim>::value_type max,
                           std::mt19937 *rng) {
    std::uniform_int_distribution<size_t> rand_int(min, max);
    Index<nDim> shape;
    for (size_t d = 0; d < nDim; ++d) {
        shape[d] = rand_int(*rng);
    }
    return shape;
}

template <size_t nDim> void TestIndexConversion() {
    std::mt19937 rng;
    for (size_t it = 0; it < 10; ++it) {
        const auto shape = GetRandomShape<nDim>(2, 6, &rng);
        const size_t size = GetSize(shape);
        std::uniform_int_distribution<size_t> rand_int(0, size - 1);
        for (size_t it2 = 0; it2 < 10; ++it2) {
            const size_t i = rand_int(rng);
            const Index<nDim> j = GetVectorIndex(i, shape);
            const size_t k = GetScalarIndex(j, shape);
            EXPECT_EQ(i, k);
        }
    }
}

TEST(Lattice, TestIndexConversion) {
    TestIndexConversion<2>();
    TestIndexConversion<3>();
    TestIndexConversion<4>();
}

template <size_t nDim> void TestGetFirstNeighbors() {
    std::mt19937 rng;
    for (size_t it1 = 0; it1 < 10; ++it1) {
        const auto shape = GetRandomShape<nDim>(2, 6, &rng);
        const size_t size = GetSize(shape);
        for (size_t si = 0; si < size; ++si) {
            const Index<nDim> i = GetVectorIndex(si, shape);
            const auto n1 = GetFirstNeighbors(i, shape);
            for (const auto &j : n1) {
                const auto n2 = GetFirstNeighbors(j, shape);
                EXPECT_NE(std::find(n2.begin(), n2.end(), i), n2.end());
            }
        }
    }
}

TEST(Lattice, TestGetFirstNeighbors) {
    TestGetFirstNeighbors<2>();
    TestGetFirstNeighbors<3>();
    TestGetFirstNeighbors<4>();
}

void MakeLaplacianMatrix(const size_t period, Eigen::MatrixXf *K) {
    K->resize(period, period);
    K->setZero();
    for (size_t i = 0; i < period; ++i) {
        (*K)(i, i) = -2.0f;
        (*K)(i, (i + 1) % period) = 1.0f;
        (*K)((i + 1) % period, i) = 1.0f;
    }
}

TEST(Observables, MakeFourierTable) {
    for (size_t period = 3; period < 10; ++period) {
        std::vector<uint32_t> waveNumbers;
        for (size_t i = 1; i < period; ++i) {
            waveNumbers.emplace_back(i);
        }

        Eigen::MatrixXf t;
        MakeFourierTable(period, waveNumbers, &t);
        Eigen::MatrixXf K;
        MakeLaplacianMatrix(period, &K);

        Eigen::MatrixXf s = t.transpose() * t;
        s -= Eigen::MatrixXf::Identity(s.rows(), s.cols());

        Eigen::MatrixXf w = t.transpose() * K * t;
        w.diagonal().setZero();

        EXPECT_LT(s.norm(), 1.0e-5);
        EXPECT_LT(w.norm(), 1.0e-5);
    }
}

TEST(Observables, Measure) {
    std::mt19937 rng;
    Observables obs;
    constexpr size_t nDim = 3;
    for (size_t i = 0; i < 5; ++i) {
        Lattice<nDim, UpDownSpin> lattice(GetRandomShape<nDim>(3, 8, &rng),
                                          UpDownSpin{0});

        std::map<Index<2>, UpDownSpin, IndexLess<2>> indexMap;

        size_t nFlipped = 0;
        for (size_t it = 0; it < lattice.size() / 2; ++it) {
            const Index<nDim> i = GetRandomIndex<nDim>(lattice.shape(), &rng);
            const size_t si = lattice.getScalarIndex(i);
            if (lattice[si].value == 0) {
                lattice[si].value = 1;
                Index<2> i2 = {i[0], i[1]};
                auto pair = indexMap.emplace(i2, 0);
                ++pair.first->second.value;
                ++nFlipped;
            }
        }

        std::vector<uint32_t> waveNumbers(10);
        std::iota(waveNumbers.begin(), waveNumbers.end(), 1);
        std::array<Eigen::MatrixXf, 2> ftTables;
        for (size_t i = 0; i < 2; ++i) {
            MakeFourierTable(lattice.shape(i), waveNumbers, &ftTables[i]);
        }

        Measure(lattice, ftTables, &obs);

        EXPECT_LT(std::abs(int64_t(nFlipped) - int64_t(obs.upCount)), 1.0e-5);
        EXPECT_LT(std::abs(2.0 * nFlipped / lattice.size() - 1.0 -
                           obs.fourierTransform2d(0, 0)),
                  1.0e-5);
    }
}

TEST(Poission, Pmf) {
    for (double lambda : {0.5, 1.0, 1.5, 2.0, 2.5}) {
        double sum = 0.0;
        for (uint64_t k = 0; k < 30 * lambda; ++k) {
            sum += std::exp(PoissonLogPmf(k, lambda));
        }
        EXPECT_LT(1.0 - sum, 1.0e-7) << "lambda: " << lambda;
    }
}

template <typename Iterator, typename ValueType>
bool NextConfiguration(Iterator begin, Iterator end, ValueType min_value,
                       ValueType max_value) {
    Iterator it = begin;
    while (it != end) {
        ++(it->value);
        if (it->value > max_value) {
            it->value = min_value;
            ++it;
        } else {
            return true;
        }
    }
    return false;
}

TEST(NextConfiguration, NextConfiguration) {
    using Config = std::array<UpDownSpin, 3>;
    // clang-format off
    std::array<Config, 27> expected_configs = {{
        {2, 2, 2}, {3, 2, 2}, {4, 2, 2},
        {2, 3, 2}, {3, 3, 2}, {4, 3, 2},
        {2, 4, 2}, {3, 4, 2}, {4, 4, 2},
        {2, 2, 3}, {3, 2, 3}, {4, 2, 3},
        {2, 3, 3}, {3, 3, 3}, {4, 3, 3},
        {2, 4, 3}, {3, 4, 3}, {4, 4, 3},
        {2, 2, 4}, {3, 2, 4}, {4, 2, 4},
        {2, 3, 4}, {3, 3, 4}, {4, 3, 4},
        {2, 4, 4}, {3, 4, 4}, {4, 4, 4}}};
    // clang-format on

    Config config = expected_configs[0];
    for (size_t i = 1; i < expected_configs.size(); ++i) {
        EXPECT_TRUE(NextConfiguration(config.begin(), config.end(), uint64_t(2),
                                      uint64_t(4)));
        EXPECT_EQ(config, expected_configs[i]);
    }
}

template <size_t nDim>
uint64_t ComputeParallelCount(const Lattice<nDim, UpDownSpin> &lattice) {
    uint64_t parallelCount = 0;
    for (size_t si = 0; si < lattice.size(); ++si) {
        UpDownSpin spin = lattice[si];
        Index<nDim> i = lattice.getVectorIndex(si);
        for (size_t d = 0; d < nDim; ++d) {
            const typename Index<nDim>::value_type i_d = i[d];
            i[d] = (i_d + 1) % lattice.shape(d);
            UpDownSpin spin1 = lattice[i];
            if (MaskedEqual(spin, spin1)) {
                ++parallelCount;
            }
            i[d] = i_d;
        }
    }
    return parallelCount;
}

template <size_t nDim>
uint64_t GetMaxParallelCount(const Lattice<nDim, UpDownSpin> &lattice) {
    return nDim * lattice.size();
}

template <size_t nDim>
std::vector<uint64_t> ComputeEntropyHistogram(const Index<nDim> &shape) {
    Lattice<nDim, UpDownSpin> lattice(shape, 0);
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
    const std::mt19937::result_type iProb =
        std::floor(std::pow(2.0, 32) * prob);
    std::mt19937 rng;
    Lattice<nDim, UpDownSpin> lattice(shape, 0);
    std::queue<Index<nDim>> queue;
    const uint64_t max_n_parallel = GetMaxParallelCount(lattice);
    std::vector<uint64_t> histogram(max_n_parallel + 1, 0);

    for (size_t iStep0 = 0; iStep0 < nMeasure; ++iStep0) {
        for (size_t iStep1 = 0; iStep1 < measureEvery; ++iStep1) {
            const Index<nDim> i0 = GetRandomIndex(shape, &rng);
            FlipCluster(iProb, i0, &lattice, &rng, &queue);
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

double ComputeDistributionPValue(double prob,
                                 const std::vector<uint64_t> &entropy_histogram,
                                 const std::vector<uint64_t> &visit_histogram) {
    if (visit_histogram.size() != entropy_histogram.size()) {
        throw std::invalid_argument("incorrect entropy_histogram.size()");
    }
    if (visit_histogram.empty()) {
        throw std::invalid_argument("empty histogram");
    }
    const uint64_t max_n_parallel = entropy_histogram.size() - 1;
    const uint64_t n_measure =
        std::accumulate(visit_histogram.begin(), visit_histogram.end(), 0);
    double chi_squared = 0;
    const std::vector<double> expected_probability =
        ComputeExpectedVisitingProbability(prob, entropy_histogram);

    // auto cout_flags = std::cout.flags();
    // std::cout << std::fixed << std::setprecision(3);
    // std::cout << std::setw(5) << "entr" << std::setw(8) << "prob"
    //           << std::setw(12) << "lambda" << std::setw(5) << "visit"
    //           << std::setw(10) << "pValue" << std::endl;
    for (size_t n_parallel = 0; n_parallel <= max_n_parallel; ++n_parallel) {
        const double lambda = expected_probability[n_parallel] * n_measure;
        const double pValue =
            PoissonExactTest(visit_histogram[n_parallel], lambda);
        chi_squared -= 2.0 * std::log(std::max(pValue, 1.0e-15));
        //    std::cout << std::setw(5) << entropy_histogram[n_parallel]
        //              << std::setw(8) << expected_probability[n_parallel]
        //              << std::setw(12) << lambda << std::setw(5)
        //              << visit_histogram[n_parallel] << std::setw(10) <<
        //              pValue
        //              << std::endl;
    }
    // std::cout.flags(cout_flags);
    return 1.0 - ChiSquaredCdf(chi_squared, 2.0 * (max_n_parallel + 1));
}

template <size_t nDim>
void TestWolffAlgorithmCorrectDistribution(
    const Index<nDim> &shape, double true_prob,
    const std::vector<double> &counterfactual_probs, uint64_t n_measure,
    uint64_t measure_every) {
    const std::vector<uint64_t> entropy_hist = ComputeEntropyHistogram(shape);
    const std::vector<uint64_t> visit_hist =
        ComputeVisitHistogram(shape, true_prob, n_measure, measure_every);
    EXPECT_GT(ComputeDistributionPValue(true_prob, entropy_hist, visit_hist),
              0.05);
    for (const double prob : counterfactual_probs) {
        EXPECT_LT(ComputeDistributionPValue(prob, entropy_hist, visit_hist),
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

TEST(UpDownHoleSpin, VisitedFlag) {
    for (UpDownHoleSpin s0(0); s0.value < 3; ++s0.value) {
        UpDownHoleSpin s1 = s0;
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

TEST(UpDownHoleSpin, MaskedEqual) {
    UpDownHoleSpin sd = UpDownHoleSpin::Down();
    UpDownHoleSpin sh = UpDownHoleSpin::Hole();
    UpDownHoleSpin su = UpDownHoleSpin::Up();
    for (uint64_t i = 0; i < 2; ++i) {
        EXPECT_EQ(sd, sd);
        EXPECT_EQ(sh, sh);
        EXPECT_EQ(su, su);
        EXPECT_NE(sd, sh);
        EXPECT_NE(sd, su);
        EXPECT_NE(sh, sd);
        EXPECT_NE(sh, su);
        EXPECT_NE(su, sd);
        EXPECT_NE(su, sh);
        MarkVisited(&sd);
        MarkVisited(&sh);
        MarkVisited(&su);
    }
}

TEST(UpDownHoleSpin, Flip) {
    UpDownHoleSpin s = UpDownHoleSpin::Down();
    Flip(&s);
    EXPECT_EQ(s, UpDownHoleSpin::Up());
    Flip(&s);
    EXPECT_EQ(s, UpDownHoleSpin::Down());
}

TEST(UpDownHoleSpin, Increment) {
    UpDownHoleSpin s = UpDownHoleSpin::Down();
    ++s.value;
    EXPECT_EQ(s, UpDownHoleSpin::Hole());
    ++s.value;
    EXPECT_EQ(s, UpDownHoleSpin::Up());
    --s.value;
    EXPECT_EQ(s, UpDownHoleSpin::Hole());
    --s.value;
    EXPECT_EQ(s, UpDownHoleSpin::Down());
}
