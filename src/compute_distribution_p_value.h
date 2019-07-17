#pragma once
#include "distributions.h"

#include <numeric>

template <typename DoubleHistogram, typename IntHistogram>
double ComputeDistributionPValue(const DoubleHistogram &expected_probability,
                                 const IntHistogram &visit_histogram) {
    if (visit_histogram.size() != expected_probability.size()) {
        throw std::invalid_argument("incorrect entropy_histogram.size()");
    }
    const uint64_t n = visit_histogram.size();
    const uint64_t n_measure =
        std::accumulate(visit_histogram.begin(), visit_histogram.end(), 0);
    double chi_squared = 0;

    for (size_t i = 0; i < n; ++i) {
        const double lambda = expected_probability[i] * n_measure;
        const double pValue = PoissonExactTest(visit_histogram[i], lambda);
        chi_squared -= 2.0 * std::log(std::max(pValue, 1.0e-15));
    }
    return 1.0 - ChiSquaredCdf(chi_squared, 2.0 * n);
}
