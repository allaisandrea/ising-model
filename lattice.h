#pragma once

#include <array>
#include <iostream>


template<size_t nDim>
using Index = std::array<uint_fast16_t, nDim>;


template<size_t nDim>
size_t GetSize(const Index<nDim>& shape) {
    size_t size = 1;
    for (size_t i = 0; i < nDim; ++i) {
        size *= shape[i];
    }
    return size;
}


template<size_t nDim>
void GetVectorIndex(size_t i, const Index<nDim>& shape, Index<nDim>* j) {
    for (size_t d = 0; d < nDim; ++d) {
        const size_t k = i / shape[d];
        (*j)[d] = i - k * shape[d];
        i = k;
    }
}


template<size_t nDim>
size_t GetScalarIndex(const Index<nDim>& j, const Index<nDim>& shape) {
    size_t i = 0;
    for (size_t d = nDim - 1; d < nDim; --d) {
        i = i * shape[d] + j[d];
    }
    return i;
}

                       
template<size_t nDim>
void GetFirstNeighbors(const Index<nDim>& i, 
                       const Index<nDim>& shape, 
                       std::array<Index<nDim>, 2 * nDim>* neighbors) {
    for (size_t d = 0; d < nDim; ++d) {
        (*neighbors)[d] = i;
        (*neighbors)[nDim + d] = i;
        (*neighbors)[d][d] = (i[d] + 1) % shape[d];
        (*neighbors)[nDim + d][d] = (i[d] + shape[d] - 1) % shape[d];
    }
}


template<size_t nDim, typename Generator>
void GetRandomIndex(const Index<nDim>& shape, Index<nDim>* i, Generator* rng) {
    for (size_t d = 0; d < nDim; ++d) {
        (*i)[d] = (*rng)() % shape[d];
    }
}


template <size_t nDim>
std::ostream& operator<< (std::ostream& os, const Index<nDim>& i) {
    for (size_t d = 0; d < nDim; ++d) {
        os << i[d] << " ";
    }
    return os;
}


