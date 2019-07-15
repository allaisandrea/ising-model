#pragma once

#include <boost/math/special_functions/gamma.hpp>

inline double ChiSquaredCdf(double x, double dof) {
    return boost::math::gamma_p(0.5 * dof, x);
}

inline double PoissonLogPmf(uint64_t k, double lambda) {
    return k * std::log(lambda) - std::lgamma(k + 1) - lambda;
}

inline double PoissonExactTest(uint64_t n, double lambda) {
    const double log_pn = PoissonLogPmf(n, lambda);
    double result = 0.0;
    double log_pk = 0.0;
    for (uint64_t k = 0; k <= lambda + 3 || log_pk > log_pn; ++k) {
        log_pk = PoissonLogPmf(k, lambda);
        if (log_pk > log_pn) {
            result += std::exp(log_pk);
        }
    }
    return 1.0 - result;
}
