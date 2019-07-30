#include <boost/program_options.hpp>
#include <random>

#include "compute_phase_diagram.h"
#include "cross_validate.h"
#include "phase_diagram.pb.h"
#include "udh_io.h"

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
    add_option("log2-J-increment",
               value<int64_t>(&(args->log2_J_increment))->required());
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

void CopyArray(const Eigen::ArrayXd &source, uint64_t offset, uint64_t n_copy,
               google::protobuf::RepeatedField<double> *dest) {
    if (source.size() < int64_t(offset + n_copy)) {
        throw std::runtime_error("Source is not big enough");
    }
    dest->Resize(n_copy, 0.0);
    std::memcpy(dest->mutable_data(), source.data() + offset,
                n_copy * sizeof(double));
}

std::string GetRandomFileName(std::mt19937 *rng) {
    std::ostringstream strm;
    strm << std::hex << (*rng)() << (*rng)();
    return strm.str() + ".pd";
}

Eigen::ArrayXd MakeDistanceArray(uint64_t n_J, uint64_t n_mu) {
    Eigen::ArrayXd result(n_J * n_mu);
    Eigen::Map<Eigen::ArrayXXd> view(result.data(), n_J, n_mu);
    for (uint64_t i_mu = 0; i_mu < n_mu; ++i_mu) {
        for (uint64_t i_J = 0; i_J < n_J; ++i_J) {
            view(i_J, i_mu) = std::sqrt(std::pow((i_J - 0.5 * n_J), 2) +
                                        std::pow((i_mu - 0.5 * n_mu), 2));
        }
    }
    return result;
}

int main(int argc, const char **argv) {
    Arguments args;
    if (!ParseArgs(argc, argv, &args)) {
        return -1;
    }

    const uint32_t seed = std::random_device()();
    std::mt19937 rng(seed);

    ParameterRange J_range, mu_range;
    J_range.increment = std::pow(2.0, args.log2_J_increment);
    mu_range.increment = std::pow(2.0, args.log2_mu_increment);
    auto pairs = ReadUdhParametersFromFiles(args.file_names.begin(),
                                            args.file_names.end());
    std::vector<UdhFileGroup> file_groups = GroupFiles(pairs, 3);

    for (auto &group : file_groups) {
        const UdhParameters &parameters = group.parameters();
        const double J = parameters.j();
        const double mu = parameters.mu();
        J_range.begin = std::round(J / J_range.increment) - args.n_J / 2;
        mu_range.begin = std::round(mu / mu_range.increment) - args.n_mu / 2;
        J_range.end = J_range.begin + args.n_J;
        mu_range.end = mu_range.begin + args.n_mu;

        const CrossValidationStats stats = CrossValidate(
            /*n_batches=*/16,
            [J_range, mu_range](uint64_t n_read, UdhFileGroup *file_group) {
                return ComputePhaseDiagram(J_range, mu_range, n_read,
                                           file_group);
            },
            &group);
        const uint64_t n_observables = group.CountObservables();
        const Eigen::ArrayXXd distance_array =
            MakeDistanceArray(args.n_J, args.n_mu);

        PhaseDiagramParams pd_params;
        pd_params.set_j(parameters.j());
        pd_params.set_mu(parameters.mu());
        *pd_params.mutable_shape() = parameters.shape();
        pd_params.set_n_wolff(parameters.n_wolff());
        pd_params.set_n_metropolis(parameters.n_metropolis());
        pd_params.set_measure_every(parameters.measure_every());
        pd_params.set_n_measure(n_observables);
        for (const UdhFileGroup::Entry &entry : group.entries()) {
            pd_params.mutable_file_names()->Add(std::string(entry.file_name));
        }
        pd_params.set_j_begin(J_range.begin);
        pd_params.set_j_end(J_range.end);
        pd_params.set_log2_j_increment(args.log2_J_increment);
        pd_params.set_mu_begin(mu_range.begin);
        pd_params.set_mu_end(mu_range.end);
        pd_params.set_log2_mu_increment(args.log2_mu_increment);

        PhaseDiagram pd;
        const uint64_t n_copy = args.n_J * args.n_mu;
        CopyArray(distance_array, 0, n_copy, pd.mutable_distance());
        CopyArray(stats.mean, 0, n_copy, pd.mutable_susceptibility());
        CopyArray(stats.std_dev, 0, n_copy, pd.mutable_susceptibility_std());
        CopyArray(stats.mean, n_copy, n_copy, pd.mutable_binder_cumulant());
        CopyArray(stats.std_dev, n_copy, n_copy,
                  pd.mutable_binder_cumulant_std());
        CopyArray(stats.mean, 2 * n_copy, n_copy, pd.mutable_hole_density());
        CopyArray(stats.std_dev, 2 * n_copy, n_copy,
                  pd.mutable_hole_density_std());
        CopyArray(stats.mean, 3 * n_copy, n_copy, pd.mutable_si_sj());
        CopyArray(stats.std_dev, 2 * n_copy, n_copy, pd.mutable_si_sj_std());

        std::ofstream out_file(GetRandomFileName(&rng), std::ios_base::binary);
        Write(pd_params, &out_file);
        Write(pd, &out_file);
    }

    return 0;
}
