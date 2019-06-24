#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <queue>
#include <random>
#include <thread>
#include <unordered_map>

#include <gtest/gtest.h>

#include "lattice.h"
#include "observables.h"
#include "poisson_distribution.h"
#include "wolff_algorithm.h"

std::mt19937 rng;

template <size_t nDim>
Index<nDim> GetRandomShape(typename Index<nDim>::value_type min,
                           typename Index<nDim>::value_type max) {
    std::uniform_int_distribution<size_t> rand_int(min, max);
    Index<nDim> shape;
    for (size_t d = 0; d < nDim; ++d) {
        shape[d] = rand_int(rng);
    }
    return shape;
}

template <size_t nDim> void TestIndexConversion() {
    std::uniform_int_distribution<size_t> rand_int;
    for (size_t it = 0; it < 10; ++it) {
        const auto shape = GetRandomShape<nDim>(2, 6);
        const size_t size = GetSize(shape);
        for (size_t it2 = 0; it2 < 10; ++it2) {
            const size_t i = rand_int(rng) % size;
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
    for (size_t it1 = 0; it1 < 10; ++it1) {
        const auto shape = GetRandomShape<nDim>(2, 6);
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
    Observables obs;
    for (size_t i = 0; i < 5; ++i) {
        const auto shape = GetRandomShape<3>(3, 8);
        const size_t size = GetSize(shape);
        std::vector<Node> nodes(size, 0);

        std::map<Index<2>, Node, IndexLess<2>> indexMap;

        size_t nFlipped = 0;
        for (size_t it = 0; it < size / 2; ++it) {
            const Index<3> i = GetRandomIndex<3>(shape, &rng);
            const size_t si = GetScalarIndex(i, shape);
            if (nodes[si] == 0) {
                nodes[si] = 1;
                Index<2> i2 = {i[0], i[1]};
                auto pair = indexMap.emplace(i2, 0);
                ++pair.first->second;
                ++nFlipped;
            }
        }

        std::vector<uint32_t> waveNumbers(10);
        std::iota(waveNumbers.begin(), waveNumbers.end(), 1);
        std::array<Eigen::MatrixXf, 2> ftTables;
        for (size_t i = 0; i < 2; ++i) {
            MakeFourierTable(shape[i], waveNumbers, &ftTables[i]);
        }

        Measure(shape, nodes.data(), ftTables, &obs);

        EXPECT_LT(std::abs(int64_t(nFlipped) - int64_t(obs.upCount)), 1.0e-5);
        EXPECT_LT(std::abs(2.0 * nFlipped / size - 1.0 -
                           obs.fourierTransform2d(0, 0)),
                  1.0e-5);
    }
}

TEST(Poission, Pmf) {
    for (double lambda : {0.5, 1.0, 1.5, 2.0, 2.5}) {
        double sum = 0.0;
        for (uint64_t k = 0; k < 30 * lambda; ++k) {
            sum += std::exp(poisson::LogPmf(k, lambda));
        }
        EXPECT_LT(1.0 - sum, 1.0e-7) << "lambda: " << lambda;
    }
}

TEST(Poission, SmallestOutcomeNotLessLikelyThan) {
    std::mt19937 rng;
    for (uint64_t i = 0; i < 32; ++i) {
        const double lambda = std::exponential_distribution<double>(0.1)(rng);
        const uint64_t k_max = 32 * std::ceil(lambda);
        for (uint64_t k = 0; k < k_max; ++k) {
            const uint64_t n =
                poisson::SmallestOutcomeNotLessLikelyThan(k, lambda);
            const double log_pk = poisson::LogPmf(k, lambda);
            const double log_pn = poisson::LogPmf(n, lambda);
            EXPECT_GE(log_pn, log_pk)
                << "n: " << n << " k: " << k << " lambda: " << lambda;
            if (n > 0) {
                const double log_pn1 = poisson::LogPmf(n - 1, lambda);
                EXPECT_LT(log_pn1, log_pk)
                    << "n: " << n << " k: " << k << " lambda: " << lambda;
            }
        }
    }
}

TEST(Poission, LargestOutcomeNotLessLikelyThan) {
    for (uint64_t i = 0; i < 32; ++i) {
        const double lambda = std::exponential_distribution<double>(0.1)(rng);
        const uint64_t k_max = 32 * std::ceil(lambda);
        for (uint64_t k = 0; k < k_max; ++k) {
            const uint64_t n =
                poisson::LargestOutcomeNotLessLikelyThan(k, lambda);
            const double log_pk = poisson::LogPmf(k, lambda);
            const double log_pn = poisson::LogPmf(n, lambda);
            const double log_pn1 = poisson::LogPmf(n + 1, lambda);
            EXPECT_GE(log_pn - log_pk, -1.0e-15)
                << "n: " << n << " k: " << k << " lambda: " << lambda;
            EXPECT_LT(log_pn1, log_pk)
                << "n: " << n << " k: " << k << " lambda: " << lambda;
        }
    }
}

template <size_t nDim>
uint64_t ComputeParallelCount(const Index<nDim> &shape, const Node *nodes) {
    const size_t size = GetSize(shape);
    uint64_t parallelCount = 0;
    for (size_t si = 0; si < size; ++si) {
        Node node = nodes[si];
        Index<nDim> i = GetVectorIndex(si, shape);
        for (size_t d = 0; d < nDim; ++d) {
            const typename Index<nDim>::value_type i_d = i[d];
            i[d] = (i_d + 1) % shape[d];
            Node node1 = nodes[GetScalarIndex(i, shape)];
            if (Parallel(node, node1)) {
                ++parallelCount;
            }
            i[d] = i_d;
        }
    }
    return parallelCount;
}

template <size_t nDim>
void TestWolfAlgorithmCorrectDistribution(Index<nDim> shape, double prob,
                                          uint64_t nMeasure,
                                          uint64_t measureEvery) {
    const std::mt19937::result_type iProb =
        std::floor(std::pow(2.0, 32) * prob);
    std::mt19937 rng;
    std::vector<Node> nodes(GetSize(shape), 0);
    std::queue<Index<nDim>> queue;
    constexpr uint64_t nTestingProbabilities = 3;
    std::array<double, nTestingProbabilities> testingProbabilities = {
        prob, prob + 0.05, prob - 0.05};
    struct ConfigurationStats {
        uint64_t nVisited;
        uint64_t nParallel;
        std::array<double, nTestingProbabilities> probabilities;
    };

    uint64_t maxNParallel = 0;
    std::unordered_map<std::string, ConfigurationStats> configurations;
    for (size_t iStep0 = 0; iStep0 < nMeasure; ++iStep0) {
        size_t cumulativeClusterSize = 0;
        for (size_t iStep1 = 0; iStep1 < measureEvery; ++iStep1) {
            const Index<nDim> i0 = GetRandomIndex(shape, &rng);
            FlipCluster(iProb, i0, shape, nodes.data(), &cumulativeClusterSize,
                        &rng, &queue);
            ClearVisitedFlag(i0, shape, nodes.data(), &queue);
        }
        auto pair =
            configurations.insert({std::string(nodes.begin(), nodes.end()),
                                   ConfigurationStats{0, 0, {}}});
        ConfigurationStats &stats = pair.first->second;
        if (pair.second) {
            stats.nParallel = ComputeParallelCount(shape, nodes.data());
            maxNParallel = std::max(maxNParallel, stats.nParallel);
        }
        stats.nVisited++;
    }

    std::array<double, nTestingProbabilities> totalProbabilities{};
    std::vector<ConfigurationStats> sorted_stats;
    for (auto &kvp : configurations) {
        ConfigurationStats &stats = kvp.second;
        for (uint64_t i = 0; i < nTestingProbabilities; ++i) {
            stats.probabilities[i] =
                std::pow(testingProbabilities[i],
                         double(maxNParallel - stats.nParallel));
            totalProbabilities[i] += stats.probabilities[i];
        }
        sorted_stats.emplace_back(stats);
    }
    std::sort(sorted_stats.begin(), sorted_stats.end(),
              [](const ConfigurationStats &c1, const ConfigurationStats &c2) {
                  return c1.nVisited > c2.nVisited;
              });

    std::array<double, nTestingProbabilities> chi_squared{};
    for (const auto &stats : sorted_stats) {
        for (uint64_t i = 0; i < nTestingProbabilities; ++i) {
            const double expectedProbability =
                stats.probabilities[i] / totalProbabilities[i];
            const double lambda = expectedProbability * nMeasure;
            const double pValue =
                poisson::PoissonPValue(stats.nVisited, lambda);
            chi_squared[i] -= 2.0 * std::log(std::max(pValue, 1.0e-15));
        }
    }
    const double dof = 2.0 * sorted_stats.size();
    for (uint64_t i = 0; i < nTestingProbabilities; ++i) {
        const double normal_z = (chi_squared[i] - dof) / std::sqrt(2.0 * dof);
        if (i == 0) {
            EXPECT_LT(std::abs(normal_z), 2.0);
        } else {
            EXPECT_GT(std::abs(normal_z), 3.0);
        }
    }
}

TEST(WolffAlgorithm, CorrectDistribution) {
    TestWolfAlgorithmCorrectDistribution(Index<3>{2, 2, 2}, 0.641978, 10000,
                                         10);
}
