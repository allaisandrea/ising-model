#include <Eigen/Core>
#include <array>
#include <boost/program_options.hpp>
#include <random>

#include "cross_validate.h"
#include "udh_critical_line.h"
#include "udh_file_group.h"
#include "udh_io.h"

UdhObservables GetObservables(UdhFileGroup *file_group) {
    UdhObservables observables;
    if (!file_group->NextObservables(&observables)) {
        throw std::runtime_error("Reached end of data");
    }
    return observables;
}

struct ParameterRange {
    int64_t begin;
    int64_t end;
    double increment;
};

Eigen::ArrayXd ReweightObservables(const ParameterRange &J_range,
                                   const ParameterRange &mu_range,
                                   uint64_t n_read, UdhFileGroup *file_group) {
    const UdhFileGroup::Position position = file_group->GetPosition();

    const double J0 = file_group->parameters().j();
    const double mu0 = file_group->parameters().mu();
    const int64_t n_J = J_range.end - J_range.begin;
    const int64_t n_mu = mu_range.end - mu_range.begin;

    Eigen::ArrayXXd max_log_weight =
        Eigen::ArrayXXd::Constant(n_J, n_mu, -INFINITY);
    for (uint64_t i = 0; i < n_read; ++i) {
        const UdhObservables observables = GetObservables(file_group);
        const int64_t sum_si_sj = observables.sum_si_sj();
        const int64_t sum_si_si = observables.n_down() + observables.n_up();
        for (int64_t i_J = 0; i_J < n_J; ++i_J) {
            const double J = (J_range.begin + i_J) * J_range.increment;
            for (int64_t i_mu = 0; i_mu < n_mu; ++i_mu) {
                const double mu = (mu_range.begin + i_mu) * mu_range.increment;
                const double log_weight =
                    (J - J0) * sum_si_sj - (mu - mu0) * sum_si_si;
                max_log_weight(i_J, i_mu) =
                    std::max(max_log_weight(i_J, i_mu), log_weight);
            }
        }
    }

    std::array<Eigen::ArrayXXd, 5> moments;
    for (auto &slice : moments) {
        slice.setZero(n_J, n_mu);
    }

    file_group->SetPosition(position);
    for (uint64_t i = 0; i < n_read; ++i) {
        const UdhObservables observables = GetObservables(file_group);
        const int64_t sum_si_sj = observables.sum_si_sj();
        const int64_t n_down = observables.n_down();
        const int64_t n_holes = observables.n_holes();
        const int64_t n_up = observables.n_up();
        const int64_t sum_si_si = n_down + n_up;
        const int64_t volume = n_down + n_holes + n_up;
        const double rho = double(n_holes) / volume;
        const double phi = double(n_up - n_down) / volume;
        const double phi2 = std::pow(phi, 2);
        const double phi4 = std::pow(phi2, 2);
        const double sum_si_sj_intensive = double(sum_si_sj) / volume;
        for (int64_t i_J = 0; i_J < n_J; ++i_J) {
            const double J = (J_range.begin + i_J) * J_range.increment;
            for (int64_t i_mu = 0; i_mu < n_mu; ++i_mu) {
                const double mu = (mu_range.begin + i_mu) * mu_range.increment;
                const double log_weight =
                    (J - J0) * sum_si_sj - (mu - mu0) * sum_si_si;
                const double weight =
                    std::exp(log_weight - max_log_weight(i_J, i_mu));
                moments[0](i_J, i_mu) += weight;
                moments[1](i_J, i_mu) += weight * phi2;
                moments[2](i_J, i_mu) += weight * phi4;
                moments[3](i_J, i_mu) += weight * rho;
                moments[4](i_J, i_mu) += weight * sum_si_sj_intensive;
            }
        }
    }
    for (uint64_t i = 1; i < moments.size(); ++i) {
        moments[i] /= moments[0];
    }

    Eigen::ArrayXd result(4 * n_J * n_mu);
    Eigen::Map<Eigen::ArrayXXd> chi_view(result.data(), n_J, n_mu);
    Eigen::Map<Eigen::ArrayXXd> u_view(result.data() + n_J * n_mu, n_J, n_mu);
    Eigen::Map<Eigen::ArrayXXd> rho_view(result.data() + 2 * n_J * n_mu, n_J,
                                         n_mu);
    Eigen::Map<Eigen::ArrayXXd> sum_si_sj_view(result.data() + 3 * n_J * n_mu,
                                               n_J, n_mu);
    chi_view = moments[1];
    u_view = 1.0 - moments[2] / (3.0 * moments[1].square());
    rho_view = moments[3];
    sum_si_sj_view = moments[4];
    return result;
}

struct Arguments {
    uint64_t n_J;
    uint64_t n_mu;
    int64_t log2_J_increment;
    int64_t log2_mu_increment;
    std::vector<std::string> file_names;
};

bool ParseArgs(int argc, const char *argv[], Arguments *args) {
    using namespace boost::program_options;

    options_description description{"Options"};
    auto &&add_option = description.add_options();
    add_option("help,h", "Show usage");
    add_option("n-J", value<uint64_t>(&(args->n_J))->required());
    add_option("n-mu", value<uint64_t>(&(args->n_mu))->required());
    add_option("log2-mu-increment",
               value<int64_t>(&(args->log2_mu_increment))->required());
    add_option("files", value<std::vector<std::string>>(&(args->file_names))
                            ->required()
                            ->multitoken());
    variables_map vm;
    store(parse_command_line(argc, argv, description), vm);
    if (vm.count("help") > 0) {
        std::cout << description << std::endl;
        return false;
    }
    notify(vm);

    return true;
}

std::string GetFilesString(const UdhFileGroup &group) {
    std::ostringstream strm;
    strm << "\"";
    for (const UdhFileGroup::Entry &entry : group.entries()) {
        strm << entry.file_name << ";";
    }
    strm << "\"";
    return strm.str();
}

int main(int argc, const char **argv) {
    Arguments args;
    if (!ParseArgs(argc, argv, &args)) {
        return -1;
    }

    const uint32_t seed = std::random_device()();
    std::mt19937 rng(seed);

    const int64_t n_J = args.n_J;
    ParameterRange mu_range;
    mu_range.increment = std::pow(2.0, args.log2_mu_increment);
    auto pairs = ReadUdhParametersFromFiles(args.file_names.begin(),
                                            args.file_names.end());
    std::vector<UdhFileGroup> file_groups = GroupFiles(pairs, 3);

    // clang-format off
    std::cout << "J0" << ","
              << "mu0" << ","
              << "L0" << ","
              << "n_wolff" << ","
              << "n_measure" << ","
              << "files" << ","
              << "log2_J_increment" << ","
              << "log2_mu_increment" << ","
              << "i_J" << ","
              << "i_mu" << ","
              << "J" << ","
              << "mu" << ","
              << "susc" << ","
              << "susc_std" << ","
              << "binder" << ","
              << "binder_std" << ","
              << "hole_density" << ","
              << "hole_density_std" << ","
              << "sum_si_sj" << ","
              << "sum_si_sj_std" << "\n";
    // clang-format on

    for (auto &group : file_groups) {
        const UdhParameters &parameters = group.parameters();
        const double mu = parameters.mu();
        const uint64_t n_dim = parameters.shape().size();
        mu_range.begin = std::round(mu / mu_range.increment) - args.n_mu / 2;
        mu_range.end = mu_range.begin + args.n_mu;

        for (int64_t i_mu = mu_range.begin; i_mu < mu_range.end; ++i_mu) {
            const double mu = mu_range.increment * i_mu;
            const auto pair = GetCriticalJ(n_dim, mu);
            const double J = pair(0);
            const double sJ = pair(1);
            ParameterRange J_range;
            J_range.increment =
                std::pow(2.0, std::ceil(std::log2(6 * sJ / n_J)));
            J_range.begin = std::round(J / J_range.increment) - n_J / 2;
            J_range.end = J_range.begin + n_J;

            ParameterRange mu_range_1{i_mu, i_mu + 1, mu_range.increment};
            const CrossValidationStats stats = CrossValidate(
                /*n_batches=*/16,
                [J_range, mu_range_1](uint64_t n_read,
                                      UdhFileGroup *file_group) {
                    return ReweightObservables(J_range, mu_range_1, n_read,
                                               file_group);
                },
                &group);
            const uint64_t n_measure = group.CountObservables();

            constexpr int n_cols = 4;
            if (stats.mean.size() != n_cols * n_J) {
                throw std::logic_error("Unexpected size");
            }
            Eigen::Map<const Eigen::ArrayXXd> mean(stats.mean.data(), n_J,
                                                   n_cols);
            Eigen::Map<const Eigen::ArrayXXd> std_dev(stats.std_dev.data(), n_J,
                                                      n_cols);
            for (int64_t i = 0; i < n_J; ++i) {
                const int64_t i_J = J_range.begin + i;
                const double J = i_J * J_range.increment;
                // clang-format off
                std::cout << std::setprecision(12);
                std::cout << parameters.j() << ","
                          << parameters.mu() << ","
                          << parameters.shape(0) << ","
                          << parameters.n_wolff() << ","
                          << n_measure << ","
                          << GetFilesString(group) << ","
                          << args.log2_J_increment << ","
                          << args.log2_mu_increment << ","
                          << i_J << ","
                          << i_mu << ","
                          << J << ","
                          << mu << ","
                          << mean(i, 0) << ","
                          << std_dev(i, 0) << ","
                          << mean(i, 1) << ","
                          << std_dev(i, 1) << ","
                          << mean(i, 2) << ","
                          << std_dev(i, 2) << ","
                          << mean(i, 3) << ","
                          << std_dev(i, 3) << "\n";
                // clang-format on
            }
        }
    }

    return 0;
}
