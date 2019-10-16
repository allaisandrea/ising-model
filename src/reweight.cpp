#include "cross_validate.h"
#include "google/protobuf/text_format.h"
#include "udh_critical_line.h"
#include "udh_file_group.h"
#include "udh_io.h"
#include <Eigen/Core>
#include <array>
#include <boost/program_options.hpp>
#include <random>

UdhObservables GetObservables(UdhFileGroup *file_group) {
    UdhObservables observables;
    if (!file_group->NextObservables(&observables)) {
        throw std::runtime_error("Reached end of data");
    }
    return observables;
}

Eigen::ArrayXd ReweightObservables(double mu, double J_begin, double J_end,
                                   uint64_t n_J, uint64_t n_read,
                                   UdhFileGroup *file_group) {
    const UdhFileGroup::Position position = file_group->GetPosition();

    const double J0 = file_group->parameters().j();
    const double mu0 = file_group->parameters().mu();

    Eigen::ArrayXd max_log_weight = Eigen::ArrayXd::Constant(n_J, -INFINITY);
    for (uint64_t i = 0; i < n_read; ++i) {
        const UdhObservables observables = GetObservables(file_group);
        const int64_t sum_si_sj = observables.sum_si_sj();
        const int64_t sum_si_si = observables.n_down() + observables.n_up();
        for (uint64_t i_J = 0; i_J < n_J; ++i_J) {
            const double J = J_begin + (J_end - J_begin) * i_J / n_J;
            const double log_weight =
                (J - J0) * sum_si_sj - (mu - mu0) * sum_si_si;
            max_log_weight(i_J) = std::max(max_log_weight(i_J), log_weight);
        }
    }

    std::array<Eigen::ArrayXd, 5> moments;
    for (auto &slice : moments) {
        slice.setZero(n_J);
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
        for (uint64_t i_J = 0; i_J < n_J; ++i_J) {
            const double J = J_begin + (J_end - J_begin) * i_J / n_J;
            const double log_weight =
                (J - J0) * sum_si_sj - (mu - mu0) * sum_si_si;
            const double weight = std::exp(log_weight - max_log_weight(i_J));
            moments[0](i_J) += weight;
            moments[1](i_J) += weight * phi2;
            moments[2](i_J) += weight * phi4;
            moments[3](i_J) += weight * rho;
            moments[4](i_J) += weight * sum_si_sj_intensive;
        }
    }
    for (uint64_t i = 1; i < moments.size(); ++i) {
        moments[i] /= moments[0];
    }

    Eigen::ArrayXd result(4 * n_J);
    Eigen::Map<Eigen::ArrayXd> chi_view(result.data(), n_J, 1);
    Eigen::Map<Eigen::ArrayXd> u_view(result.data() + n_J, n_J, 1);
    Eigen::Map<Eigen::ArrayXd> rho_view(result.data() + 2 * n_J, n_J, 1);
    Eigen::Map<Eigen::ArrayXd> sum_si_sj_view(result.data() + 3 * n_J, n_J, 1);
    chi_view = moments[1];
    u_view = 1.0 - moments[2] / (3.0 * moments[1].square());
    rho_view = moments[3];
    sum_si_sj_view = moments[4];
    return result;
}

struct Arguments {
    double mu;
    double J_begin;
    double J_end;
    uint64_t n_J;
    uint64_t measure_every;
    uint64_t skip_first_n;
    std::vector<std::string> file_names;
    std::string out_file;
};

bool ParseArgs(int argc, const char *argv[], Arguments *args) {
    using namespace boost::program_options;

    options_description description{"Options"};
    auto &&add_option = description.add_options();
    add_option("help,h", "Show usage");
    add_option("mu", value<double>(&(args->mu))->required());
    add_option("J-begin", value<double>(&(args->J_begin))->required());
    add_option("J-end", value<double>(&(args->J_end))->required());
    add_option("n-J", value<uint64_t>(&(args->n_J))->required());
    add_option("measure-every",
               value<uint64_t>(&(args->measure_every))->required());
    add_option("skip_first_n",
               value<uint64_t>(&(args->skip_first_n))->default_value(4));
    add_option("out-file", value<std::string>(&(args->out_file))->required());
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

int main(int argc, const char **argv) {
    Arguments args;
    if (!ParseArgs(argc, argv, &args)) {
        return -1;
    }
    UdhFileGroup group(
        GetUdhFileGroupEntries(args.file_names, args.measure_every),
        args.skip_first_n);
    const UdhParameters &parameters = group.parameters();

    const CrossValidationStats stats = CrossValidate(
        /*n_batches=*/16,
        [args](uint64_t n_read, UdhFileGroup *file_group) {
            return ReweightObservables(args.mu, args.J_begin, args.J_end,
                                       args.n_J, n_read, file_group);
        },
        &group);
    const uint64_t count = group.CountObservables();

    constexpr int n_cols = 4;
    if (stats.mean.size() != int64_t(n_cols * args.n_J)) {
        throw std::logic_error("Unexpected size");
    }
    Eigen::Map<const Eigen::ArrayXXd> mean(stats.mean.data(), args.n_J, n_cols);
    Eigen::Map<const Eigen::ArrayXXd> std_dev(stats.std_dev.data(), args.n_J,
                                              n_cols);
    std::ofstream out_file(args.out_file, std::ios_base::out |
                                              std::ios_base::app |
                                              std::ios_base::binary);
    if (!out_file.good()) {
        throw std::runtime_error("Unable to open \"" + args.out_file + "\"");
    }
    for (uint64_t i_J = 0; i_J < args.n_J; ++i_J) {
        const double J =
            args.J_begin + (args.J_end - args.J_begin) * i_J / args.n_J;
        UdhAggregateObservables obs;
        obs.set_j0(parameters.j());
        obs.set_mu0(parameters.mu());
        *obs.mutable_shape() = parameters.shape();
        obs.set_n_wolff(parameters.n_wolff());
        obs.set_n_metropolis(parameters.n_metropolis());
        obs.set_measure_every(args.measure_every);
        for (const auto &file_name : args.file_names) {
            obs.add_origin_files(file_name);
        }
        obs.set_mu(args.mu);
        obs.set_j(J);
        obs.set_count(count);
        obs.set_susceptibility(mean(i_J, 0));
        obs.set_susceptibility_std(std_dev(i_J, 0));
        obs.set_binder_cumulant(mean(i_J, 1));
        obs.set_binder_cumulant_std(std_dev(i_J, 1));
        obs.set_hole_density(mean(i_J, 2));
        obs.set_hole_density_std(std_dev(i_J, 2));
        obs.set_sum_si_sj(mean(i_J, 3));
        obs.set_sum_si_sj_std(std_dev(i_J, 3));
        Write(obs, &out_file);
    }

    return 0;
}
