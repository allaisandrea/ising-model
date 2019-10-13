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
    int64_t log2_J_increment;
    uint64_t measure_every;
    uint64_t skip_first_n;
    bool print_header;
    std::vector<std::string> file_names;
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
    add_option("print-header",
               value<bool>(&(args->print_header))->default_value(false));
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

std::vector<UdhFileGroup::Entry>
GetUdhFileGroupEntries(const std::vector<std::string> &file_names,
                       uint64_t measure_every) {
    std::vector<UdhFileGroup::Entry> result;
    UdhParameters params, prev_params;
    for (uint64_t i = 0; i < file_names.size(); ++i) {
        const auto &file_name = file_names[i];
        std::ifstream file(file_name);
        if (!Read(&params, &file)) {
            throw std::runtime_error("Unable to read parameters from file \"" +
                                     file_name + "\"");
        }
        if (measure_every % params.measure_every() != 0) {
            throw std::runtime_error(
                "Incompatible value of measure_every for file \"" + file_name +
                "\"");
        }
        if (i > 0 && !OutputCanBeJoined(params, prev_params)) {
            throw std::runtime_error("File \"" + file_name + "\" and \"" +
                                     file_names[i - 1] +
                                     "\" have incompatible parameters \"");
        }
        result.emplace_back(UdhFileGroup::Entry{
            .file_name = file_name,
            .read_every = measure_every / params.measure_every()});
        prev_params = params;
    }
    return result;
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

    if (args.print_header) {
        // clang-format off
        std::cout << "J0" << ","
                  << "mu0" << ","
                  << "L0" << ","
                  << "n_wolff" << ","
                  << "n_measure" << ","
                  << "files" << ","
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
    }

    const CrossValidationStats stats = CrossValidate(
        /*n_batches=*/16,
        [args](uint64_t n_read, UdhFileGroup *file_group) {
            return ReweightObservables(args.mu, args.J_begin, args.J_end,
                                       args.n_J, n_read, file_group);
        },
        &group);
    const uint64_t n_measure = group.CountObservables();

    constexpr int n_cols = 4;
    if (stats.mean.size() != int64_t(n_cols * args.n_J)) {
        throw std::logic_error("Unexpected size");
    }
    Eigen::Map<const Eigen::ArrayXXd> mean(stats.mean.data(), args.n_J, n_cols);
    Eigen::Map<const Eigen::ArrayXXd> std_dev(stats.std_dev.data(), args.n_J,
                                              n_cols);
    for (uint64_t i_J = 0; i_J < args.n_J; ++i_J) {
        const double J =
            args.J_begin + (args.J_end - args.J_begin) * i_J / args.n_J;
        // clang-format off
                std::cout << std::setprecision(12);
                std::cout << parameters.j() << ","
                          << parameters.mu() << ","
                          << parameters.shape(0) << ","
                          << parameters.n_wolff() << ","
                          << n_measure << ","
                          << GetFilesString(group) << ","
                          << J << ","
                          << args.mu << ","
                          << mean(i_J, 0) << ","
                          << std_dev(i_J, 0) << ","
                          << mean(i_J, 1) << ","
                          << std_dev(i_J, 1) << ","
                          << mean(i_J, 2) << ","
                          << std_dev(i_J, 2) << ","
                          << mean(i_J, 3) << ","
                          << std_dev(i_J, 3) << "\n";
        // clang-format on
    }

    return 0;
}
