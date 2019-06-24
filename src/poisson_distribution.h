#pragma once
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace poisson {

double LogPmf(uint64_t k, double lambda) {
    return k * std::log(lambda) - std::lgamma(k + 1) - lambda;
}

double ExactTest(uint64_t n, double lambda) {

    const double log_pn = LogPmf(n, lambda);
    double result = 0.0;
    double log_pk = 0.0;
    for (uint64_t k = 0; k <= lambda + 3 || log_pk > log_pn; ++k) {
        log_pk = LogPmf(k, lambda);
        if (log_pk > log_pn) {
            result += std::exp(log_pk);
        }
    }
    return 1.0 - result;
}
} // namespace poisson
