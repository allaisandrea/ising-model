#pragma once

#include "tensor.h"
#include <queue>

inline uint32_t GetNoAddProbabilityFromJ(double J) {
    return std::round((1ul << 32) * std::exp(-2.0 * J));
}

template <size_t nDim, typename Spin>
size_t FlipCluster(uint64_t p_no_add, const Index<nDim> &i0,
                   Tensor<nDim, Spin> *pTensor, std::mt19937 *rng,
                   std::queue<Index<nDim>> *queue) {
    size_t clusterSize = 0;
    Tensor<nDim, Spin> &lattice = *pTensor;
    queue->emplace(i0);
    MarkVisited(&lattice[i0]);

    while (!queue->empty()) {
        Index<nDim> &i = queue->front();
        Spin *spin0 = &lattice[i];
        for (size_t d = 0; d < nDim; ++d) {
            const typename Index<nDim>::value_type i_d = i[d];
            const typename Index<nDim>::value_type s_d = lattice.shape(d);
            for (size_t dir = 0; dir < 4; dir += 2) {
                i[d] = (i_d + s_d + dir - 1) % s_d;
                Spin *spin1 = &lattice[i];
                const bool add = !Visited(*spin1) &&
                                 MaskedEqual(*spin0, *spin1) &&
                                 (*rng)() > p_no_add;
                if (add) {
                    queue->emplace(i);
                    MarkVisited(spin1);
                }
            }
            i[d] = i_d;
        }

        Flip(spin0);
        queue->pop();
        ++clusterSize;
    }
    return clusterSize;
}

template <size_t nDim, typename Spin>
void ClearVisitedFlag(const Index<nDim> &i0, Tensor<nDim, Spin> *pTensor,
                      std::queue<Index<nDim>> *queue) {

    Tensor<nDim, Spin> &lattice = *pTensor;
    queue->emplace(i0);
    ClearVisitedFlag(&lattice[i0]);

    while (!queue->empty()) {
        Index<nDim> &i = queue->front();
        for (size_t d = 0; d < nDim; ++d) {
            const typename Index<nDim>::value_type i_d = i[d];
            const typename Index<nDim>::value_type s_d = lattice.shape(d);
            for (size_t dir = 0; dir < 4; dir += 2) {
                i[d] = (i_d + s_d + dir - 1) % s_d;
                Spin *spin1 = &lattice[i];
                if (Visited(*spin1)) {
                    queue->emplace(i);
                    ClearVisitedFlag(spin1);
                }
            }
            i[d] = i_d;
        }
        queue->pop();
    }
}
