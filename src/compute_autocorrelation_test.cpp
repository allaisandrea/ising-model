#include "compute_autocorrelation.h"
#include "eigen_expect_near.h"
#include "mock_file_system.h"

#include <gtest/gtest.h>
#include <random>

namespace {

Eigen::Array3d SampleExponentialDistribution(const Eigen::Array3d &lambda,
                                             std::mt19937 &rng) {
    Eigen::Array3d result;
    for (int i = 0; i < result.rows(); ++i) {
        result(i) = std::exponential_distribution<double>(lambda(i))(rng);
    }
    return result;
}

void SetObservables(uint64_t i, const Eigen::Vector3d &values,
                    UdhObservables *observables) {
    observables->set_n_down(std::round(values(0)));
    observables->set_n_holes(std::round(values(1)));
    observables->set_n_up(std::round(values(2)));
    observables->set_stamp(i);
}

Eigen::Array3d NextValues(const Eigen::Array3d &lambda,
                          const Eigen::Array3d &alpha,
                          const Eigen::Array3d &values, std::mt19937 &rng) {
    return alpha * values + SampleExponentialDistribution(lambda, rng);
}

std::string MakeMockData(uint64_t n, const Eigen::Array3d &lambda,
                         const Eigen::Array3d &alpha, std::mt19937 &rng) {
    std::ostringstream strm;
    UdhParameters params;
    Write(params, &strm);
    UdhObservables observables;
    Eigen::Array3d values;
    values.setZero();
    for (uint64_t i = 0; i < 256; ++i) {
        values = NextValues(lambda, alpha, values, rng);
    }
    for (uint64_t i = 0; i < n; ++i) {
        SetObservables(i, values, &observables);
        Write(observables, &strm);
        values = NextValues(lambda, alpha, values, rng);
    }
    return strm.str();
}

TEST(ComputeAutocorrelation, ComputeAutocorrelation) {
    std::mt19937 rng;
    MockFileSystem mock_file_system;
    for (int i = 0; i < 8; ++i) {
        Eigen::Array3d lambda;
        const uint64_t log_n_obs =
            std::uniform_int_distribution<uint64_t>(16, 16)(rng);
        const uint64_t n_obs = 1ul << log_n_obs;
        lambda << std::uniform_real_distribution<double>(0.0, 1.0e-4)(rng),
            std::uniform_real_distribution<double>(0.0, 1.0e-4)(rng),
            std::uniform_real_distribution<double>(0.0, 1.0e-4)(rng);
        lambda(2) = lambda(0);
        Eigen::Array3d alpha;
        alpha.setZero();
        alpha << std::uniform_real_distribution<double>(0.0, 0.3)(rng),
            std::uniform_real_distribution<double>(0.0, 0.3)(rng),
            std::uniform_real_distribution<double>(0.0, 0.3)(rng);
        alpha(2) = alpha(0);
        mock_file_system.files["file1"] =
            MakeMockData(n_obs, lambda, alpha, rng);
        UdhFileGroup file_group({{"file1", 1}}, 0, mock_file_system);
        const Eigen::ArrayXd result =
            ComputeAutocorrelation(2, n_obs, &file_group);
        EXPECT_NEAR(result(0), 1.0, 1.0e-12);
        EXPECT_NEAR(result(1), 1.0, 1.0e-12);
        EXPECT_NEAR(result(2), alpha(0), 3.0 / std::sqrt(n_obs));
        EXPECT_NEAR(result(3), alpha(1), 3.0 / std::sqrt(n_obs));
    }
}
} // namespace
