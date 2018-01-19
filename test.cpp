#include "lattice.h"
#include <random>
#include <iostream>
#include <algorithm>

std::mt19937 rng;

template<size_t nDim>
void GetRandomShape(typename Index<nDim>::value_type min, 
                    typename Index<nDim>::value_type max, 
                    Index<nDim>* shape) {
    std::uniform_int_distribution<size_t> rand_int(min, max);
    
    for (size_t d = 0; d < nDim; ++d) {
        (*shape)[d] = rand_int(rng);
    }
}


template<size_t nDim>
bool TestIndexConversion() {

    std::uniform_int_distribution<size_t> rand_int;
    
    Index<nDim> shape;
    Index<nDim> j;

    for (size_t it = 0; it < 10; ++it) {
        GetRandomShape(2, 6, &shape);
        const size_t size = GetSize(shape);
        for (size_t it2 = 0; it2 < 10; ++it2) {
            const size_t i = rand_int(rng) % size;
            GetVectorIndex(i, shape, &j);
            const size_t k = GetScalarIndex(j, shape);
            if (i != k) {
                return false;
            }
        }
    }

    return true;
}

template<size_t nDim>
bool TestGetFirstNeighbors() {
    
    Index<nDim> shape;
    Index<nDim> i;
    std::array<Index<nDim>, 2 * nDim> n1;
    std::array<Index<nDim>, 2 * nDim> n2;
    
    for(size_t it1 = 0; it1 < 10; ++it1) {
        GetRandomShape(2, 6, &shape);
        const size_t size = GetSize(shape);
        for(size_t si = 0; si < size; ++si) {
            GetVectorIndex(si, shape, &i);
            GetFirstNeighbors(i, shape, &n1);
            for (const auto& j : n1) {
                GetFirstNeighbors(j, shape, &n2);
                if (std::find(n2.begin(), n2.end(), i) == n2.end()) {
                    std::cout << "i: " << i << std::endl;
                    std::cout << "j: " << j << std::endl;
                    std::cout << "shape: " << shape << std::endl;
                    std::cout << "n1: " << std::endl;
                    for (const auto& k : n1) {
                        std::cout << k << std::endl;
                    }
                    std::cout << std::endl;
                    std::cout << "n2: " << std::endl;
                    for (const auto& k : n2) {
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

    return 0;

}