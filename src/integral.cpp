#include <array>
#include <cmath>
#include <iostream>
#include <vector>

double Propagator(int n0, int n1, double ma2,
                  const std::vector<double> &sinTable) {
    return 1.0 / (sinTable[n0] + sinTable[n1] + ma2);
}

double Propagator(int n0, int n1, int n2, double ma2,
                  const std::vector<double> &sinTable) {
    return 1.0 / (sinTable[n0] + sinTable[n1] + sinTable[n2] + ma2);
}

std::array<double, 2> I2(double ma, int N, std::vector<double> *sinTable) {
    sinTable->clear();
    sinTable->reserve(N);
    for (int i = 0; i < N; ++i) {
        const double x = 2.0 * std::sin(M_PI * double(i) / N);
        (*sinTable)[i] = x * x;
    }

    std::array<double, 2> sum = {0.0, 0.0};
    const double ma2 = ma * ma;
    for (int n0 = 0; n0 < N; ++n0)
        for (int n1 = 0; n1 < N; ++n1) {
            const double Delta = Propagator(n0, n1, ma2, *sinTable);
            sum[0] += Delta;
            sum[1] += Delta * Delta;
        }
    sum[0] /= N * N;
    sum[1] /= N * N;
    sum[1] *= ma2;
    return sum;
}

std::array<double, 2> I3(double ma, int N, std::vector<double> *sinTable) {
    sinTable->clear();
    sinTable->reserve(N);
    for (int i = 0; i < N; ++i) {
        const double x = 2.0 * std::sin(M_PI * double(i) / N);
        (*sinTable)[i] = x * x;
    }

    std::array<double, 2> sum = {0.0};
    const double ma2 = ma * ma;
    for (int n0 = 0; n0 < N; ++n0) {
        for (int n1 = 0; n1 < N; ++n1) {
            for (int n2 = 0; n2 < N; ++n2) {
                const double Delta = Propagator(n0, n1, n2, ma2, *sinTable);
                sum[0] += Delta;
                sum[1] += Delta * Delta;
            }
        }
    }
    const double N3 = std::pow(double(N), 3);
    sum[0] *= 1.0 / (ma * N3);
    sum[1] *= ma / N3;
    return sum;
}

int main() {
    std::vector<double> sinTable;
    std::vector<double> lValues;
    std::vector<int> nValues = {/*32, 64, 128, 256,*/ 512, 1024, 2048, 4096};

    const uint64_t nL = 64;
    lValues.reserve(nL);
    for (uint64_t i = 0; i < nL; ++i) {
        lValues.emplace_back(0.5 + 8.5 * i / (nL - 1));
    }

    const double m = 1.0;
    for (const auto &N : nValues)
        for (const auto &L : lValues) {
            const double a = L / N;
            const auto sum = I3(m * a, N, &sinTable);
            std::cout << L << "," << N << "," << sum[0] << "," << sum[1]
                      << std::endl;
        }
    return 0;
}
