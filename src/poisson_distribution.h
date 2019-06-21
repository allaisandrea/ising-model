#pragma once
#include <cmath>
#include <cstdint>
#include <stdexcept>

double PoissonCDF(uint64_t n, double lambda) {
    if (lambda <= 0.0) {
        throw std::invalid_argument("Lambda should be positive");
    }
    double pn = std::exp(-lambda);
    double result = pn;
    for (uint64_t i = 1; i < n; ++i) {
        pn *= lambda / i;
        result += pn;
    }
    return result;
}

double PoissonPValue(uint64_t n, double lambda) {
    if (lambda <= 0.0) {
        throw std::invalid_argument("Lambda should be positive");
    }
    double pn = std::exp(-lambda);
    for (uint64_t i = 1; i <= n; ++i) {
        pn *= lambda / i;
    }

    double pk = std::exp(-lambda);
    double result = 0.0f;
    for (uint64_t i = 1; i <= lambda + 3 || pk > pn; ++i) {
        if (pk > pn) {
            result += pk;
        }
        pk *= lambda / i;
    }
    return 1.0 - result;
}
