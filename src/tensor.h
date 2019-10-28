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
bool IndexIsValid(const Index<nDim> &i, const Index<nDim> &shape) {
    for (size_t d = 0; d < nDim; ++d) {
        if (i[d] >= shape[d]) {
            return false;
        }
    }
    return true;
}

template <size_t nDim>
Index<nDim> HypercubeShape(typename Index<nDim>::value_type L) {
    Index<nDim> shape;
    for (size_t d = 0; d < nDim; ++d) {
        shape[d] = L;
    }
    return shape;
}

template <size_t nDim>
Index<nDim> operator+(const Index<nDim> &i1, const Index<nDim> &i2) {
    Index<nDim> result = i1;
    for (size_t d = 0; d < nDim; ++d) {
        result[d] += i2[d];
    }
    return result;
}

template <size_t nDim>
Index<nDim> operator*(const Index<nDim> &i1, const Index<nDim> &i2) {
    Index<nDim> result = i1;
    for (size_t d = 0; d < nDim; ++d) {
        result[d] *= i2[d];
    }
    return result;
}

// Advance vector index in lexicographic order. Returns false if the highest
// possible index has been passed and index is now all zeros again, otherwise
// returns true.
template <size_t nDim>
bool NextIndex(Index<nDim> *i, const Index<nDim> &shape) {
    for (size_t d = 0; d < nDim; ++d) {
        ++(*i)[d];
        if ((*i)[d] < shape[d]) {
            return true;
        } else {
            (*i)[d] = 0;
        }
    }
    return false;
}

// Advance vector index in lexicographic order. Returns false if the highest
// possible index has been passed and index is now all zeros again, otherwise
// returns true.
template <size_t nDim>
bool NextIndex(Index<nDim> *i, const Index<nDim> &shape,
               const uint_fast16_t stride) {
    for (size_t d = 0; d < nDim; ++d) {
        (*i)[d] += stride;
        if ((*i)[d] < shape[d]) {
            return true;
        } else {
            (*i)[d] -= shape[d];
        }
    }
    return false;
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

template <size_t nDim>
Index<nDim> GetRandomIndex(const Index<nDim> &shape, std::mt19937 *rng) {
    Index<nDim> i;
    for (size_t d = 0; d < nDim; ++d) {
        i[d] = std::uniform_int_distribution<typename Index<nDim>::value_type>(
            0, shape[d] - 1)(*rng);
    }
    return i;
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

template <size_t nDim, typename Scalar> class Tensor {
    Index<nDim> _shape;
    size_t _size;
    std::vector<Scalar> _nodes;

  public:
    Tensor() : _shape{}, _size{}, _nodes{} {}
    Tensor(const Index<nDim> &shape, const Scalar &node)
        : _shape(shape), _size(GetSize(_shape)), _nodes(_size, node) {}

    void resize(const Index<nDim> &shape, const Scalar &node) {
        _shape = shape;
        _size = GetSize(_shape);
        _nodes.clear();
        _nodes.resize(_size, node);
    }

    friend void swap(Tensor<nDim, Scalar> &t1, Tensor<nDim, Scalar> &t2) {
        std::swap(t1._shape, t2._shape);
        std::swap(t1._size, t2._size);
        std::swap(t1._nodes, t2._nodes);
    }

    const Index<nDim> &shape() const { return _shape; }
    size_t shape(size_t d) const { return _shape[d]; }
    size_t size() const { return _size; };
    Scalar &operator[](size_t i) { return _nodes[i]; }
    const Scalar &operator[](size_t i) const { return _nodes[i]; }
    Scalar &operator[](const Index<nDim> &i) {
        return _nodes[GetScalarIndex(i, _shape)];
    }
    const Scalar &operator[](const Index<nDim> &i) const {
        return _nodes[GetScalarIndex(i, _shape)];
    }
    Index<nDim> getVectorIndex(size_t i) const {
        return GetVectorIndex(i, _shape);
    }
    size_t getScalarIndex(const Index<nDim> &i) const {
        return GetScalarIndex(i, _shape);
    }
    typename std::vector<Scalar>::iterator begin() { return _nodes.begin(); }
    typename std::vector<Scalar>::const_iterator begin() const {
        return _nodes.cbegin();
    }
    typename std::vector<Scalar>::iterator end() { return _nodes.end(); }
    typename std::vector<Scalar>::const_iterator end() const {
        return _nodes.cend();
    }
};

template <size_t nDim, typename Scalar>
void TileTensor(const Tensor<nDim, Scalar> &tile, const Index<nDim> &n_tiles,
                Tensor<nDim, Scalar> *result) {
    if (tile.size() == 0) {
        throw std::invalid_argument("Cannot tile tensor of zero size");
    }
    result->resize(tile.shape() * n_tiles, tile[0]);
    Index<nDim> tile_index{};
    do {
        Index<nDim> i{};
        do {
            (*result)[tile.shape() * tile_index + i] = tile[i];
        } while (NextIndex(&i, tile.shape()));
    } while (NextIndex(&tile_index, n_tiles));
}
