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

#include "observables.h"
#include "progress.h"
#include "tensor.h"
#include "timer.h"
#include "udh_io.h"
#include "udh_measure.h"
#include "udh_metropolis_algorithm.h"
#include "udh_spin.h"
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

TEST(Tensor, TestIndexConversion) {
    TestIndexConversion<2>();
    TestIndexConversion<3>();
    TestIndexConversion<4>();
}

TEST(Tensor, IndexIsValid) {
    EXPECT_TRUE(IndexIsValid(Index<2>{0, 0}, Index<2>{2, 3}));
    EXPECT_TRUE(IndexIsValid(Index<2>{1, 0}, Index<2>{2, 3}));
    EXPECT_TRUE(IndexIsValid(Index<2>{0, 2}, Index<2>{2, 3}));
    EXPECT_FALSE(IndexIsValid(Index<2>{2, 0}, Index<2>{2, 3}));
    EXPECT_FALSE(IndexIsValid(Index<2>{0, 3}, Index<2>{2, 3}));

    EXPECT_TRUE(IndexIsValid(Index<3>{0, 0, 0}, Index<3>{2, 3, 4}));
    EXPECT_TRUE(IndexIsValid(Index<3>{1, 0, 0}, Index<3>{2, 3, 4}));
    EXPECT_TRUE(IndexIsValid(Index<3>{0, 2, 0}, Index<3>{2, 3, 4}));
    EXPECT_TRUE(IndexIsValid(Index<3>{0, 0, 3}, Index<3>{2, 3, 4}));
    EXPECT_FALSE(IndexIsValid(Index<3>{2, 0, 0}, Index<3>{2, 3, 4}));
    EXPECT_FALSE(IndexIsValid(Index<3>{0, 3, 0}, Index<3>{2, 3, 4}));
    EXPECT_FALSE(IndexIsValid(Index<3>{0, 0, 4}, Index<3>{2, 3, 4}));
}

TEST(Tensor, NextIndex2D) {
    const Index<2> shape{5, 4};
    // clang-format off
    std::array<Index<2>, 20> expected_indices = {{
        {0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0},
        {0, 1}, {1, 1}, {2, 1}, {3, 1}, {4, 1},
        {0, 2}, {1, 2}, {2, 2}, {3, 2}, {4, 2},
        {0, 3}, {1, 3}, {2, 3}, {3, 3}, {4, 3},
    }};
    // clang-format on
    Index<2> index{0, 0};
    size_t i = 0;
    do {
        EXPECT_EQ(index, expected_indices.at(i));
        ++i;
    } while (NextIndex(&index, shape));
    EXPECT_EQ(i, expected_indices.size());
}

TEST(Tensor, NextIndex3D) {
    const Index<3> shape{4, 3, 2};
    // clang-format off
    std::array<Index<3>, 24> expected_indices = {{
        {0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {3, 0, 0},
        {0, 1, 0}, {1, 1, 0}, {2, 1, 0}, {3, 1, 0},
        {0, 2, 0}, {1, 2, 0}, {2, 2, 0}, {3, 2, 0},
        {0, 0, 1}, {1, 0, 1}, {2, 0, 1}, {3, 0, 1},
        {0, 1, 1}, {1, 1, 1}, {2, 1, 1}, {3, 1, 1},
        {0, 2, 1}, {1, 2, 1}, {2, 2, 1}, {3, 2, 1},
    }};
    // clang-format on
    Index<3> index{0, 0, 0};
    size_t i = 0;
    do {
        EXPECT_EQ(index, expected_indices.at(i));
        ++i;
    } while (NextIndex(&index, shape));
    EXPECT_EQ(i, expected_indices.size());
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

TEST(Tensor, TestGetFirstNeighbors) {
    TestGetFirstNeighbors<2>();
    TestGetFirstNeighbors<3>();
    TestGetFirstNeighbors<4>();
}

template <size_t nDim>
void TestTileTensor(const Index<nDim> &tile_shape, const Index<nDim> &n_tiles,
                    const std::vector<uint64_t> &expected_values) {
    Tensor<nDim, uint64_t> tile(tile_shape, 0);
    for (uint64_t i = 0; i < tile.size(); ++i) {
        tile[i] = i;
    }
    const Tensor<nDim, uint64_t> tiled = TileTensor(tile, n_tiles);
    for (size_t i = 0; i < expected_values.size(); ++i) {
        EXPECT_EQ(tiled[i], expected_values[i]);
    }
    EXPECT_EQ(expected_values.size(), tiled.size());
}

TEST(Tensor, TileTensor1) {
    // clang-format off
    TestTileTensor<2>({2, 3}, {3, 2}, {
             0, 1, 0, 1, 0, 1,
             2, 3, 2, 3, 2, 3,
             4, 5, 4, 5, 4, 5,
             0, 1, 0, 1, 0, 1,
             2, 3, 2, 3, 2, 3,
             4, 5, 4, 5, 4, 5});
    // clang-format on
}

TEST(Tensor, TileTensor2) {
    // clang-format off
    TestTileTensor<2>({2, 3}, {2, 3}, {
             0, 1, 0, 1 ,
             2, 3, 2, 3 ,
             4, 5, 4, 5 ,
             0, 1, 0, 1 ,
             2, 3, 2, 3 ,
             4, 5, 4, 5 ,
             0, 1, 0, 1 ,
             2, 3, 2, 3 ,
             4, 5, 4, 5 });
    // clang-format on
}

TEST(Tensor, TileTensor3) {
    // clang-format off
    TestTileTensor<3>({2, 3, 2}, {3, 2, 2}, {
             0, 1, 0, 1, 0, 1,
             2, 3, 2, 3, 2, 3,
             4, 5, 4, 5, 4, 5,
             0, 1, 0, 1, 0, 1,
             2, 3, 2, 3, 2, 3,
             4, 5, 4, 5, 4, 5,

             6, 7, 6, 7, 6, 7,
             8, 9, 8, 9, 8, 9,
            10,11,10,11,10,11,
             6, 7, 6, 7, 6, 7,
             8, 9, 8, 9, 8, 9,
            10,11,10,11,10,11,

             0, 1, 0, 1, 0, 1,
             2, 3, 2, 3, 2, 3,
             4, 5, 4, 5, 4, 5,
             0, 1, 0, 1, 0, 1,
             2, 3, 2, 3, 2, 3,
             4, 5, 4, 5, 4, 5,

             6, 7, 6, 7, 6, 7,
             8, 9, 8, 9, 8, 9,
            10,11,10,11,10,11,
             6, 7, 6, 7, 6, 7,
             8, 9, 8, 9, 8, 9,
            10,11,10,11,10,11,
             });
    // clang-format on
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
        Tensor<nDim, UdSpin> lattice(GetRandomShape<nDim>(3, 8, &rng),
                                     UdSpinDown());

        std::map<Index<2>, UdSpin, IndexLess<2>> indexMap;

        size_t nFlipped = 0;
        for (size_t it = 0; it < lattice.size() / 2; ++it) {
            const Index<nDim> i = GetRandomIndex<nDim>(lattice.shape(), &rng);
            const size_t si = lattice.getScalarIndex(i);
            if (lattice[si].value == 0) {
                lattice[si].value = 1;
                Index<2> i2 = {i[0], i[1]};
                auto pair = indexMap.emplace(i2, UdSpinDown());
                Flip(&pair.first->second);
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

struct ConfigurationItem {
    uint64_t value;
    ConfigurationItem(uint64_t value) : value(value) {}
};

bool operator==(const ConfigurationItem &i1, const ConfigurationItem &i2) {
    return i1.value == i2.value;
}

TEST(NextConfiguration, NextConfiguration) {
    using Config = std::array<ConfigurationItem, 3>;
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

template <typename DoubleHistogram, typename IntHistogram>
double ComputeDistributionPValue(const DoubleHistogram &expected_probability,
                                 const IntHistogram &visit_histogram) {
    if (visit_histogram.size() != expected_probability.size()) {
        throw std::invalid_argument("incorrect entropy_histogram.size()");
    }
    const uint64_t n = visit_histogram.size();
    const uint64_t n_measure =
        std::accumulate(visit_histogram.begin(), visit_histogram.end(), 0);
    double chi_squared = 0;

    for (size_t i = 0; i < n; ++i) {
        const double lambda = expected_probability[i] * n_measure;
        const double pValue = PoissonExactTest(visit_histogram[i], lambda);
        chi_squared -= 2.0 * std::log(std::max(pValue, 1.0e-15));
    }
    return 1.0 - ChiSquaredCdf(chi_squared, 2.0 * n);
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
Index<2> ComputeEnergies(const Tensor<nDim, UdhSpin> &lattice) {
    Index<2> energies{};
    Index<nDim> i{};
    energies[0] = nDim * lattice.size();
    do {
        UdhSpin spin = lattice[i];
        if (spin == UdhSpinHole()) {
            continue;
        }
        ++energies[1];
        for (size_t d = 0; d < nDim; ++d) {
            const typename Index<nDim>::value_type i_d = i[d];
            i[d] = (i_d + 1) % lattice.shape(d);
            UdhSpin spin1 = lattice[i];
            energies[0] +=
                (int64_t(spin.value) - 1) * (int64_t(spin1.value) - 1);
            i[d] = i_d;
        }
    } while (NextIndex(&i, lattice.shape()));
    return energies;
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
    const uint32_t p_no_add = std::round((1ul << 32) * std::exp(-2.0 * J));

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

TEST(UdhIo, ReadWrite) {
    std::mt19937 rng;
    std::normal_distribution<double> normal_variate(0, 1);

    udh::Parameters parameters_out;
    parameters_out.set_j(normal_variate(rng));
    parameters_out.set_mu(normal_variate(rng));
    parameters_out.mutable_shape()->Add(rng());
    parameters_out.mutable_shape()->Add(rng());
    parameters_out.set_tag("foo bar");

    std::vector<udh::Observables> observables_out(4);
    for (udh::Observables &obs : observables_out) {
        obs.set_stamp(rng());
        obs.set_sum_si_sj(rng());
    }

    std::stringstream strm;
    Write(parameters_out, &strm);
    for (const udh::Observables &obs : observables_out) {
        Write(obs, &strm);
    }

    strm.seekg(0);
    udh::Parameters parameters_in;
    EXPECT_TRUE(Read(&parameters_in, &strm));

    EXPECT_EQ(parameters_out.j(), parameters_in.j());
    EXPECT_EQ(parameters_out.mu(), parameters_in.mu());
    EXPECT_EQ(parameters_out.shape(0), parameters_in.shape(0));
    EXPECT_EQ(parameters_out.shape(1), parameters_in.shape(1));
    EXPECT_EQ(parameters_out.tag(), parameters_in.tag());

    std::vector<udh::Observables> observables_in;
    do {
        observables_in.emplace_back();
    } while (Read(&observables_in.back(), &strm));
    observables_in.resize(observables_in.size() - 1);

    ASSERT_EQ(observables_out.size(), observables_in.size());
    for (size_t i = 0; i < observables_out.size(); ++i) {
        EXPECT_EQ(observables_out[i].stamp(), observables_in[i].stamp());
        EXPECT_EQ(observables_out[i].sum_si_sj(),
                  observables_in[i].sum_si_sj());
    }
};

template <size_t nDim>
void TestUdhMeasureRandomConfiguration(const Index<nDim> &shape) {
    std::mt19937 rng;
    std::uniform_int_distribution<uint8_t> random_spin(0, 2);
    Tensor<nDim, UdhSpin> lattice(shape, UdhSpinDown());
    for (uint64_t i = 0; i < 8; ++i) {
        for (UdhSpin &node : lattice) {
            node.value = random_spin(rng);
        }
        udh::Observables observables;
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
    udh::Observables observables;
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
    using time_point = uint64_t;
    using duration = int64_t;
    static uint64_t time;
    static uint64_t now() { return time; }
};
uint64_t MockClock::time{};

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
