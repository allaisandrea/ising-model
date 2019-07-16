#include "udh_io.h"

#include <gtest/gtest.h>
#include <random>

TEST(UdhIo, ReadWrite) {
    std::mt19937 rng;
    std::normal_distribution<double> normal_variate(0, 1);

    UdhParameters parameters_out;
    parameters_out.set_j(normal_variate(rng));
    parameters_out.set_mu(normal_variate(rng));
    parameters_out.mutable_shape()->Add(rng());
    parameters_out.mutable_shape()->Add(rng());
    parameters_out.set_tag("foo bar");

    std::vector<UdhObservables> observables_out(4);
    for (UdhObservables &obs : observables_out) {
        obs.set_stamp(rng());
        obs.set_sum_si_sj(rng());
    }

    std::stringstream strm;
    Write(parameters_out, &strm);
    for (const UdhObservables &obs : observables_out) {
        Write(obs, &strm);
    }

    strm.seekg(0);
    UdhParameters parameters_in;
    EXPECT_TRUE(Read(&parameters_in, &strm));

    EXPECT_EQ(parameters_out.j(), parameters_in.j());
    EXPECT_EQ(parameters_out.mu(), parameters_in.mu());
    EXPECT_EQ(parameters_out.shape(0), parameters_in.shape(0));
    EXPECT_EQ(parameters_out.shape(1), parameters_in.shape(1));
    EXPECT_EQ(parameters_out.tag(), parameters_in.tag());

    std::vector<UdhObservables> observables_in;
    do {
        observables_in.emplace_back();
    } while (Read(&observables_in.back(), &strm));
    observables_in.resize(observables_in.size() - 1);

    ASSERT_EQ(observables_out.size(), observables_in.size());
    for (size_t i = 0; i < observables_out.size(); ++i) {
        EXPECT_EQ(observables_out[i].stamp(), observables_in[i].stamp());
        EXPECT_EQ(observables_out[i].sum_si_sj(),
                  observables_in[i].sum_si_sj());
    }
};
