#pragma once

#include <array>
#include <iostream>
#include <random>

template <size_t nDim> using Index = std::array<uint_fast16_t, nDim>;

template <size_t nDim> size_t GetSize(const Index<nDim> &shape) {
    size_t size = 1;
    for (size_t i = 0; i < nDim; ++i) {
        size *= shape[i];
    }
    return size;
}

template <size_t nDim>
Index<nDim> GetVectorIndex(size_t i, const Index<nDim> &shape) {
    Index<nDim> j;
    for (size_t d = 0; d < nDim; ++d) {
        const size_t k = i / shape[d];
        j[d] = i - k * shape[d];
        i = k;
    }
    return j;
}

template <size_t nDim>
size_t GetScalarIndex(const Index<nDim> &j, const Index<nDim> &shape) {
    size_t i = 0;
    for (size_t d = nDim - 1; d < nDim; --d) {
        i = i * shape[d] + j[d];
    }
    return i;
}

template <size_t nDim>
std::array<Index<nDim>, 2 * nDim> GetFirstNeighbors(const Index<nDim> &i,
                                                    const Index<nDim> &shape) {
    std::array<Index<nDim>, 2 * nDim> neighbors;
    for (size_t d = 0; d < nDim; ++d) {
        neighbors[d] = i;
        neighbors[nDim + d] = i;
        neighbors[d][d] = (i[d] + 1) % shape[d];
        neighbors[nDim + d][d] = (i[d] + shape[d] - 1) % shape[d];
    }
    return neighbors;
}

template <size_t nDim, typename Generator>
Index<nDim> GetRandomIndex(const Index<nDim> &shape, Generator *rng) {
    Index<nDim> i;
    for (size_t d = 0; d < nDim; ++d) {
        i[d] = std::uniform_int_distribution<typename Index<nDim>::value_type>(
            0, shape[d] - 1)(*rng);
    }
    return i;
}

template <size_t Dim> struct IndexLess {
    bool operator()(const Index<Dim> &i1, const Index<Dim> &i2) const {
        return std::lexicographical_compare(i1.begin(), i1.end(), i2.begin(),
                                            i2.end());
    }
};

template <size_t nDim>
std::ostream &operator<<(std::ostream &os, const Index<nDim> &i) {
    for (size_t d = 0; d < nDim; ++d) {
        os << i[d] << " ";
    }
    return os;
}
