#include <array>
#include <cmath>
#include <iostream>
#include <vector>

std::array<double, 3> ComputeIntegrals(const double eta, const double s,
                                       int n_max) {
    std::array<double, 3> res{};
    for (int k1 = -n_max; k1 <= n_max; ++k1) {
        for (int k2 = -n_max; k2 <= n_max; ++k2) {
            for (int k3 = -n_max; k3 <= n_max; ++k3) {
                for (int k4 = -n_max; k4 <= n_max; ++k4) {
                    if (k1 == 0 && k2 == 0 && k3 == 0 && k4 == 0) {
                        continue;
                    }
                    const double p =
                        k1 * k1 + k2 * k2 + k3 * k3 + k4 * k4 + eta;
                    const double reg = std::exp(-s * p);
                    res[0] += reg / p;
                    res[1] += reg / (p * p);
                    res[2] += reg / (p * p * p);
                }
            }
        }
    }
    return res;
}
int main() {
    const std::vector<double> eta_values = {-0.5, -0.25, 0.0, 0.25,
                                            0.5,  0.75,  1.0};
    std::cout.precision(15);
    std::cout << "eta,s,n_max,I1,I2,I3\n";
    for (int i_s = 0; i_s < 9; ++i_s) {
        for (const double eta : eta_values) {
            const double s = std::pow(2.0, -i_s);
            for (int j : {4, 6, 8}) {
                const int n_max = j / std::sqrt(s);
                auto integrals = ComputeIntegrals(eta, s, n_max);
                std::cout << eta << "," << s << "," << n_max << ","
                          << integrals[0] << "," << integrals[1] << ","
                          << integrals[2] << std::endl;
            }
        }
    }
}
