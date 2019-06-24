#pragma once
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace poisson {

double LogPmf(uint64_t k, double lambda) {
    return k * std::log(lambda) - std::lgamma(k + 1) - lambda;
}

uint64_t SmallestOutcomeNotLessLikelyThan(uint64_t k, double lambda) {
    if (k < lambda) {
        return k;
    }
    const double log_pk = LogPmf(k, lambda);
    const double log_p0 = LogPmf(0, lambda);
    if (log_p0 >= log_pk) {
        return 0;
    }
    uint64_t n0 = 0;
    uint64_t n1 = std::ceil(lambda) - 1;
    while (n1 - n0 > 1) {
        uint64_t n = n0 + (n1 - n0) / 2;
        if (LogPmf(n, lambda) >= log_pk) {
            n1 = n;
        } else {
            n0 = n;
        }
    }
    return n1;
}

uint64_t LargestOutcomeNotLessLikelyThan(uint64_t k, double lambda) {
    if (k >= lambda) {
        return k;
    }
    const double log_pk = LogPmf(k, lambda);
    uint64_t n0 = std::floor(lambda);
    uint64_t n1 = n0 + 1;
    while (LogPmf(n1, lambda) >= log_pk) {
        n1 = 2 * n1;
    }
    while (n1 - n0 > 1) {
        uint64_t n = n0 + (n1 - n0) / 2;
        if (LogPmf(n, lambda) >= log_pk) {
            n0 = n;
        } else {
            n1 = n;
        }
    }
    return n0;
}

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
} // namespace poisson
