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
