#pragma once
#include <cstdint>

struct UdhTransitionProbs {
    uint64_t p_dh_or_uh;
    uint64_t p_hd;
    uint64_t p_hd_plus_p_hu;
};

UdhTransitionProbs ComputeUdhTransitionProbs(const double p_d, const double p_h,
                                             const double p_u) {
    return {uint64_t((1ul << 32) * std::min(1.0, p_h / (p_u + p_d))),
            uint64_t((1ul << 32) * std::min(p_d / p_h, p_d / (p_u + p_d))),
            uint64_t((1ul << 32) * std::min((p_d + p_u) / p_h, 1.0))};
}

template <size_t nDim>
using UdhTransitionProbsArray = std::array<UdhTransitionProbs, 4 * nDim + 1>;

template <size_t nDim>
UdhTransitionProbsArray<nDim> ComputeUdhTransitionProbs(const double J,
                                                        const double mu) {
    std::array<UdhTransitionProbs, 4 * nDim + 1> result;
    const int64_t max_n_val = 2 * nDim;
    for (int64_t n = -max_n_val; n <= max_n_val; ++n) {
        const double p_d = std::exp(-J * n - mu);
        const double p_h = 1.0;
        const double p_u = std::exp(+J * n - mu);
        result.at(max_n_val + n) = ComputeUdhTransitionProbs(p_d, p_h, p_u);
    }
    return result;
}

template <size_t nDim>
void UdhMetropolisMove(
    const UdhTransitionProbsArray<nDim> &transition_probs_array,
    const Index<nDim> &i, Lattice<nDim, UdhSpin> *pLattice, std::mt19937 *rng) {
    Lattice<nDim, UdhSpin> &lattice = *pLattice;

    // Total magnetization of the neighbors
    uint64_t n = 0;
    Index<nDim> j = i;
    for (size_t d = 0; d < nDim; ++d) {
        const typename Index<nDim>::value_type s_d = lattice.shape(d);
        j[d] = (i[d] + s_d - 1) % s_d;
        n += lattice[j].value;
        j[d] = (i[d] + 1) % s_d;
        n += lattice[j].value;
        j[d] = i[d];
    }

    // Effect transition
    const UdhTransitionProbs &transition_probs = transition_probs_array.at(n);
    UdhSpin &s = lattice[i];
    const uint32_t random_u32 = (*rng)();
    if (s == UdhSpinDown()) {
        if (random_u32 < transition_probs.p_dh_or_uh) {
            ++s.value;
        }
    } else if (s == UdhSpinUp()) {
        if (random_u32 < transition_probs.p_dh_or_uh) {
            --s.value;
        }
    } else if (s == UdhSpinHole()) {
        if (random_u32 < transition_probs.p_hd) {
            --s.value;
        } else if (random_u32 < transition_probs.p_hd_plus_p_hu) {
            ++s.value;
        }
    }
}

template <size_t nDim>
void UdhMetropolisSweep(
    const UdhTransitionProbsArray<nDim> &transition_probs_array,
    Lattice<nDim, UdhSpin> *lattice, std::mt19937 *rng) {
    Index<nDim> i{};
    do {
        UdhMetropolisMove(transition_probs_array, i, lattice, rng);
    } while (NextIndex(&i, lattice->shape()));
}
