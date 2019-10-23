#pragma once
#include "udh.pb.h"
#include <boost/program_options.hpp>
#include <iomanip>
#include <random>

bool ParseArgs(int argc, const char *argv[], UdhParameters *parameters) {
    using namespace boost::program_options;

    std::vector<uint32_t> shape;
    double J, mu;
    uint32_t n_wolff, n_metropolis, measure_every, n_measure, seed,
        metropolis_stride;
    bool quenched_holes;
    std::string id, tag;
    seed = std::random_device()();
    std::mt19937 rng(seed);
    std::ostringstream strm;
    strm << std::hex << std::setfill('0') << std::setw(8) << rng()
         << std::setw(8) << rng();
    id = strm.str();

    options_description description{"Options"};
    auto &&add_option = description.add_options();
    add_option("help,h", "Show usage");
    add_option("J", value<double>(&J)->required());
    add_option("mu", value<double>(&mu)->required());
    add_option("shape",
               value<std::vector<uint32_t>>(&shape)->required()->multitoken());
    add_option("n-wolff", value<uint32_t>(&n_wolff)->required());
    add_option("n-metropolis", value<uint32_t>(&n_metropolis)->required());
    add_option("metropolis-stride",
               value<uint32_t>(&metropolis_stride)->default_value(0));
    add_option("measure-every", value<uint32_t>(&measure_every)->required());
    add_option("n-measure", value<uint32_t>(&n_measure)->required());
    add_option("quenched-holes",
               value<bool>(&quenched_holes)->default_value(false));
    add_option("seed", value<uint32_t>(&seed));
    add_option("id", value<std::string>(&id));
    add_option("tag", value<std::string>(&tag));

    variables_map vm;
    store(parse_command_line(argc, argv, description), vm);
    if (vm.count("help") > 0) {
        std::cout << description << std::endl;
        return false;
    }
    notify(vm);

    if (quenched_holes) {
        if (mu < 0.0 || mu > 1.0) {
            throw std::invalid_argument(
                "For quenched holes mu must be between 0.0 and 1.0");
        }
        if (n_metropolis != 0) {
            throw std::invalid_argument(
                "Metropolis algorithm is not compatible with quenched holes.");
        }
        if (n_wolff != 1) {
            throw std::invalid_argument(
                "With quenched holes, n_wolff must be 1.");
        }
    }

    parameters->set_j(J);
    parameters->set_mu(mu);
    parameters->mutable_shape()->Clear();
    for (const uint32_t i : shape) {
        if (i % 8 != 0) {
            throw std::invalid_argument("Dimensions should be multiples of 8");
        }
        parameters->mutable_shape()->Add(i);
    }
    parameters->set_n_wolff(n_wolff);
    parameters->set_n_metropolis(n_metropolis);
    parameters->set_metropolis_stride(metropolis_stride);
    parameters->set_measure_every(measure_every);
    parameters->set_n_measure(n_measure);
    parameters->set_seed(seed);
    parameters->set_id(id);
    parameters->set_tag(tag);
    parameters->set_quenched_holes(quenched_holes);
    return true;
}
