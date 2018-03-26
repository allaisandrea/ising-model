#include "Queue.h"
#include "lattice.h"
#include "observables.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <queue>
#include <random>
#include <thread>

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

template <size_t nDim> bool TestIndexConversion() {
    std::uniform_int_distribution<size_t> rand_int;
    for (size_t it = 0; it < 10; ++it) {
        const auto shape = GetRandomShape<nDim>(2, 6);
        const size_t size = GetSize(shape);
        for (size_t it2 = 0; it2 < 10; ++it2) {
            const size_t i = rand_int(rng) % size;
            const Index<nDim> j = GetVectorIndex(i, shape);
            const size_t k = GetScalarIndex(j, shape);
            if (i != k) {
                return false;
            }
        }
    }

    return true;
}

template <size_t nDim> bool TestGetFirstNeighbors() {
    for (size_t it1 = 0; it1 < 10; ++it1) {
        const auto shape = GetRandomShape<nDim>(2, 6);
        const size_t size = GetSize(shape);
        for (size_t si = 0; si < size; ++si) {
            const Index<nDim> i = GetVectorIndex(si, shape);
            const auto n1 = GetFirstNeighbors(i, shape);
            for (const auto &j : n1) {
                const auto n2 = GetFirstNeighbors(j, shape);
                if (std::find(n2.begin(), n2.end(), i) == n2.end()) {
                    std::cout << "i: " << i << std::endl;
                    std::cout << "j: " << j << std::endl;
                    std::cout << "shape: " << shape << std::endl;
                    std::cout << "n1: " << std::endl;
                    for (const auto &k : n1) {
                        std::cout << k << std::endl;
                    }
                    std::cout << std::endl;
                    std::cout << "n2: " << std::endl;
                    for (const auto &k : n2) {
                        std::cout << k << std::endl;
                    }
                    std::cout << std::endl;
                    return false;
                }
            }
        }
    }
    return true;
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

bool TestMakeFourierTable(size_t period) {

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

    return s.norm() < 1.0e-5 && w.norm() < 1.0e-5;
}

bool TestMakeFourierTable() {
    for (size_t period = 3; period < 10; ++period) {
        if (!TestMakeFourierTable(period)) {
            std::cout << "TestMakeFourierTable failed at " << period
                      << std::endl;
            return false;
        }
    }
    return true;
}

bool TestMeasure(MeasureWorkspace *work, Observables *obs) {
    const auto shape = GetRandomShape<3>(3, 8);
    const size_t size = GetSize(shape);
    std::vector<Node> nodes(size, 0);

    std::map<Index<2>, Node, IndexLess<2>> indexMap;

    size_t nFlipped = 0;
    for (size_t it = 0; it < size / 2; ++it) {
        const Index<3> i = GetRandomIndex<3, decltype(rng)>(shape, &rng);
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

    Measure(shape, nodes.data(), ftTables, obs, work);

    for (const auto &pair : indexMap) {
        const Index<2> i = pair.first;
        const size_t sum = pair.second;
        if (sum != work->slice2dSum(i[0], i[1])) {
            std::cerr << "Failed slice2dSum (1)" << std::endl;
            return false;
        }
        work->slice2dSum(i[0], i[1]) = 0;
    }
    if (work->slice2dSum.norm() > 1.0e-7) {
        std::cerr << "Failed slice2dSum (2)" << std::endl;
        return false;
    }

    Eigen::MatrixXf M =
        ftTables[0].transpose() * work->slice2dMagnetization * ftTables[1] -
        obs->fourierTransform2d;
    if (M.norm() > 1.0e-7) {
        std::cerr << "Failed fourierTransform2d (1)" << std::endl;
        return false;
    }

    if (std::abs(nFlipped - obs->upCount) > 1.0e-5) {
        std::cerr << "Failed magnetization" << std::endl;
        return false;
    }

    if (std::abs(2.0 * nFlipped / size - 1.0 - obs->fourierTransform2d(0, 0)) >
        1.0e-5) {
        std::cerr << "Failed fourierTransform2d (2)" << std::endl;
        return false;
    }
    return true;
}

bool TestMeasure() {
    MeasureWorkspace work;
    Observables obs;
    for (size_t i = 0; i < 5; ++i) {
        if (!TestMeasure(&work, &obs)) {
            return false;
        }
    }
    return true;
}

bool TestQueue() {
    for (size_t it = 0; it < 100; ++it) {

        Queue<int64_t> queue1(1 + std::rand() % 32);
        std::queue<int64_t> queue2;

        for (size_t i = 0; i < 16; ++i) {
            const int64_t x = std::rand();
            queue1.emplace(x);
            queue2.emplace(x);
        }

        while (!queue1.empty()) {
            if (queue1.front() != queue2.front()) {
                return false;
            }
            queue1.pop();
            queue2.pop();
        }

        if (!queue2.empty()) {
            while (!queue2.empty()) {
                queue2.pop();
            }
            return false;
        }
    }
    return true;
}

int main() {
    if (!TestIndexConversion<2>()) {
        std::cerr << "Failed TestIndexConversion<2>" << std::endl;
    } else {
        std::cerr << "Passed TestIndexConversion<2>" << std::endl;
    }

    if (!TestIndexConversion<3>()) {
        std::cerr << "Failed TestIndexConversion<3>" << std::endl;
    } else {
        std::cerr << "Passed TestIndexConversion<3>" << std::endl;
    }

    if (!TestIndexConversion<4>()) {
        std::cerr << "Failed TestIndexConversion<4>" << std::endl;
    } else {
        std::cerr << "Passed TestIndexConversion<4>" << std::endl;
    }

    if (!TestGetFirstNeighbors<2>()) {
        std::cerr << "Failed TestGetFirstNeighbors<2>" << std::endl;
    } else {
        std::cerr << "Passed TestGetFirstNeighbors<2>" << std::endl;
    }

    if (!TestGetFirstNeighbors<3>()) {
        std::cerr << "Failed TestGetFirstNeighbors<3>" << std::endl;
    } else {
        std::cerr << "Passed TestGetFirstNeighbors<3>" << std::endl;
    }

    if (!TestGetFirstNeighbors<4>()) {
        std::cerr << "Failed TestGetFirstNeighbors<4>" << std::endl;
    } else {
        std::cerr << "Passed TestGetFirstNeighbors<4>" << std::endl;
    }

    if (!TestMakeFourierTable()) {
        std::cerr << "Failed TestMakeFourierTable" << std::endl;
    } else {
        std::cerr << "Passed TestMakeFourierTable" << std::endl;
    }

    if (!TestMeasure()) {
        std::cerr << "Failed TestMeasure" << std::endl;
    } else {
        std::cerr << "Passed TestMeasure" << std::endl;
    }

    if (!TestQueue()) {
        std::cerr << "Failed TestQueue" << std::endl;
    } else {
        std::cerr << "Passed TestQueue" << std::endl;
    }
    return 0;
}
