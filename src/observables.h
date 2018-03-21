#pragma once

#include "Node.h"
#include <eigen3/Eigen/Core>
#include <vector>

struct Observables {
    uint64_t flipClusterDuration;
    uint64_t clearFlagDuration;
    uint64_t measureDuration;
    uint64_t serializeDuration;
    uint64_t cumulativeClusterSize;
    uint64_t nClusters;
    uint64_t representativeState;
    std::vector<uint32_t> stateCount;
    uint64_t parallelCount;
    float magnetization;
    Eigen::MatrixXf fourierTransform2d;
};

struct MeasureWorkspace {
    Eigen::Matrix<uint_fast32_t, Eigen::Dynamic, Eigen::Dynamic> slice2dSum;
    Eigen::MatrixXf slice2dMagnetization;
    Eigen::MatrixXf partialFT;
};

template <size_t nDim>
void Measure(const Index<nDim> &shape, const Node *nodes,
             const std::array<Eigen::MatrixXf, 2> fourierTables,
             Observables *obs, MeasureWorkspace *work) {
    obs->representativeState = nodes[0];

    const size_t size = GetSize(shape);
    const size_t sliceSize = shape[0] * shape[1];
    const size_t nSlices = size / sliceSize;

    work->slice2dSum.resize(shape[0], shape[1]);
    work->slice2dSum.setZero();
    uint_fast32_t *slice2dSum = work->slice2dSum.data();
    obs->stateCount.assign(2, 0);
    obs->parallelCount = 0;
    for (size_t si = 0; si < size; ++si) {
        Node node = nodes[si];
        ++(obs->stateCount[node]);
        const size_t j = si % sliceSize;
        slice2dSum[j] += node;

        Index<nDim> i;
        GetVectorIndex(si, shape, &i);
        for (size_t d = 0; d < nDim; ++d) {
            const typename Index<nDim>::value_type i_d = i[d];
            i[d] = (i_d + 1) % shape[d];
            Node node1 = nodes[GetScalarIndex(i, shape)];
            if (Parallel(node, node1)) {
                ++obs->parallelCount;
            }
            i[d] = i_d;
        }
    }

    obs->magnetization =
        float(int64_t(obs->stateCount[1]) - int64_t(obs->stateCount[0])) /
        (obs->stateCount[1] + obs->stateCount[0]);

    work->slice2dMagnetization = work->slice2dSum.cast<float>();
    work->slice2dMagnetization *= 2.0f / nSlices;
    work->slice2dMagnetization.array() -= 1.0f;
    work->slice2dMagnetization /= std::sqrt(sliceSize);
    work->partialFT.noalias() =
        fourierTables[0].transpose() * work->slice2dMagnetization;
    obs->fourierTransform2d.noalias() = work->partialFT * fourierTables[1];
}

void MakeFourierTable(size_t period, const std::vector<uint32_t> &waveNumbers,
                      Eigen::MatrixXf *table) {
    const auto &wn = waveNumbers;

    const float norm0 = 1.0f / std::sqrt(period);
    const float normk = 1.0f / std::sqrt(0.5f * period);

    const size_t maxK = (period + 1) / 2;
    size_t nWaves = 0;
    for (size_t i = 0; i < wn.size() && wn[i] < maxK; ++i) {
        ++nWaves;
    }

    table->resize(period, 2 * nWaves + 1);

    for (size_t x = 0; x < period; ++x) {
        (*table)(x, 0) = norm0;
    }
    for (size_t i = 0, j = 1; i < wn.size() && wn[i] < maxK; ++i) {
        for (size_t x = 0; x < period; ++x) {
            const float theta = 2.0 * M_PI * x * wn[i] / period;
            (*table)(x, j) = normk * std::sin(theta);
            (*table)(x, j + 1) = normk * std::cos(theta);
        }
        j += 2;
    }
}
