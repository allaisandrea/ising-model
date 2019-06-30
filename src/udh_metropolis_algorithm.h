#pragma once
#include <cstdint>

struct UdhTransitionProbs {
    uint32_t p_dh;
    uint32_t p_hd;
    uint32_t p_hu;
    uint32_t p_uh;
};

UdhTransitionProbs ComputeUdhTransitionProbs(const double p_d, const double p_h,
                                             const double p_u) {
    const double s = std::min(1.0 / p_h, 1.0 / (p_u + p_d));
    auto MakeProb = [s](double p) {
        return uint32_t(
            std::round(s * p * std::numeric_limits<uint32_t>::max()));
    };
    return {MakeProb(p_h), MakeProb(p_d), MakeProb(p_u), MakeProb(p_h)};
}

template <size_t nDim>
std::array<UdhTransitionProbs, 4 * nDim + 1>
ComputeUdhTransitionProbs(const double J, const double r) {
    std::array<UdhTransitionProbs, 4 * nDim + 1> result;
    const int64_t max_n_val = 2 * nDim;
    for (int64_t n = -max_n_val; n <= max_n_val; ++n) {
        const double p_d = std::exp(-J * n - r);
        const double p_h = 1.0;
        const double p_u = std::exp(+J * n - r);
        result.at(max_n_val + n) = ComputeUdhTransitionProbs(p_d, p_h, p_u);
    }
    return result;
}
