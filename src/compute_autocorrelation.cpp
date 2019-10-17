#include "compute_autocorrelation.h"
#include "compute_timing.h"
#include "cross_validate.h"
#include "udh_file_group.h"
#include <boost/program_options.hpp>
#include <cmath>

struct Arguments {
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

int main(int argc, const char *argv[]) {
    Arguments args;
    if (!ParseArgs(argc, argv, &args)) {
        return -1;
    }

    UdhFileGroup group(
        GetUdhFileGroupEntries(args.file_names, args.measure_every),
        args.skip_first_n);

    const CrossValidationStats autocorrelation = CrossValidate(
        /*n_batches=*/16, &ComputeAutocorrelation, &group);
    const CrossValidationStats timing = CrossValidate(
        /*n_batches=*/16, &ComputeTiming, &group);
    const uint64_t count = group.CountObservables();
    const uint64_t n_tau = autocorrelation.mean.size() / 2;
    std::ofstream out_file(args.out_file, std::ios_base::out |
                                              std::ios_base::app |
                                              std::ios_base::binary);
    if (!out_file.good()) {
        throw std::runtime_error("Unable to open \"" + args.out_file + "\"");
    }
    for (uint64_t tau = 1; tau < n_tau; ++tau) {
        UdhAutocorrelationPoint pt;
        pt.set_j(group.parameters().j());
        pt.set_mu(group.parameters().mu());
        *pt.mutable_shape() = group.parameters().shape();
        pt.set_n_wolff(group.parameters().n_wolff());
        pt.set_n_metropolis(group.parameters().n_metropolis());
        for (const auto &file_name : args.file_names) {
            pt.add_origin_files(file_name);
        }
        pt.set_count(count);
        pt.set_tau(tau);
        pt.set_ud_autocorrelation(autocorrelation.mean(2 * tau));
        pt.set_ud_autocorrelation_std(autocorrelation.std_dev(2 * tau));
        pt.set_h_autocorrelation(autocorrelation.mean(2 * tau + 1));
        pt.set_h_autocorrelation_std(autocorrelation.std_dev(2 * tau + 1));
        pt.set_t_wolff(timing.mean(0));
        pt.set_t_metropolis(timing.mean(1));
        pt.set_t_measure(timing.mean(2));
        pt.set_t_serialize(timing.mean(3));
        pt.set_t_residual(timing.mean(4));
        Write(pt, &out_file);
    }
}
