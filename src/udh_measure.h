#include "udh_simulation.pb.h"
#include "udh_spin.h"

template <size_t nDim>
void Measure(const Tensor<nDim, UdhSpin> &lattice,
             udh::Observables *observables) {
    Index<nDim> i{};
    std::array<uint64_t, 3> udh_count{};
    int64_t sum_si_sj{};
    do {
        UdhSpin spin = lattice[i];
        ++udh_count[spin.value];
        for (size_t d = 0; d < nDim; ++d) {
            const typename Index<nDim>::value_type i_d = i[d];
            i[d] = (i_d + 1) % lattice.shape(d);
            UdhSpin spin1 = lattice[i];
            sum_si_sj += spin.value * spin1.value;
            i[d] = i_d;
        }
    } while (NextIndex(&i, lattice.shape()));

    const int64_t sum_si = udh_count[2] - udh_count[0];
    sum_si_sj -= 2 * nDim * sum_si + nDim * lattice.size();
    observables->set_n_down(udh_count[0]);
    observables->set_n_holes(udh_count[1]);
    observables->set_n_up(udh_count[2]);
    observables->set_sum_si_sj(sum_si_sj);
}
