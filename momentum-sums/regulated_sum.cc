#include <numeric>
#include <array>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

std::array<double, 2> ComputeSum(double eps, int k_max) {
    const double eps2 = eps * eps;
    std::array<double, 2> sums{};
    for (int k0 = -k_max; k0 <= k_max; ++k0) {
        for (int k1 = -k_max; k1 <= k_max; ++k1) {
            for (int k2 = -k_max; k2 <= k_max; ++k2) {
                for (int k3 = -k_max; k3 <= k_max; ++k3) {
                    const int k_sq = k0 * k0 + k1 * k1 + k2 * k2 + k3 * k3;
                    if (k_sq > 0) {
                        sums[0] += std::exp(-eps2 * k_sq) / k_sq;
                        sums[1] += std::exp(-eps * std::sqrt(k_sq)) / k_sq;
                    }
                }
            }
        }
    }
    return sums;
}

std::array<double, 1> ComputeSum1(int k_max) {
    std::array<double, 1> sums{};
    for (int k0 = 0; k0 < k_max; ++k0) {
        const double s0 = 2.0 * std::sin(M_PI * k0 / k_max);
        for (int k1 = 0; k1 < k_max; ++k1) {
            const double s1 = 2.0 * std::sin(M_PI * k1 / k_max);
            for (int k2 = 0; k2 < k_max; ++k2) {
                const double s2 = 2.0 * std::sin(M_PI * k2 / k_max);
                for (int k3 = 0; k3 < k_max; ++k3) {
                    const double s3 = 2.0 * std::sin(M_PI * k3 / k_max);
                    const double k_sq = s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3;
                    if (k_sq > 0.0) {
                        sums[0] += 1.0 / k_sq;
                    }
                }
            }
        }
    }
    return sums;
}

void PrintSumTable() {
    const auto flags = std::cout.flags();
    std::cout.precision(20);
    std::vector<double> lambda_values = {1, 2, 4, 8, 16, 32};
    lambda_values.resize(20);
    std::iota(lambda_values.begin(), lambda_values.end(), 1);
    std::vector<int> k_max_values = {4, 8, 16, 32, 64, 128};
    for (int k_max : k_max_values) {
        const auto sums1 = ComputeSum1(k_max);
        for (int lambda : lambda_values) {
            const auto sums = ComputeSum(1.0 / lambda, k_max);
            std::cout << k_max << "," << lambda << "," << sums[0] << ","
                      << sums[1] << "," << sums1[0] << std::endl;
        }
    }
    std::cout.flags(flags);
}

int main(int, char **) {
    PrintSumTable();
    return 0;
}
