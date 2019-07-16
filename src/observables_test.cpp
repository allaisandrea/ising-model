#include "observables.h"

#include <gtest/gtest.h>

namespace {
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
}
