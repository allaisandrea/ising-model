#include <fstream>

#include "google/protobuf/text_format.h"
#include "progress.h"
#include "tensor.h"
#include "throttle.h"
#include "timer.h"
#include "udh_arguments.h"
#include "udh_io.h"
#include "udh_measure.h"
#include "udh_metropolis_algorithm.h"
#include "udh_spin.h"
#include "wolff_algorithm.h"

namespace {
template <size_t nDim>
Index<nDim> GetStaticShape(const UdhParameters &parameters) {
    if (parameters.shape_size() != nDim) {
        throw std::logic_error(
            "\"nDim\" does not match \"parameters\" specification");
    }
    Index<nDim> shape;
    for (size_t d = 0; d < nDim; ++d) {
        shape[d] = parameters.shape(d);
    }
    return shape;
}

template <size_t nDim>
bool ShapeCanBeTiled(const Index<nDim> &shape, uint64_t tile_size) {
    for (size_t d = 0; d < nDim; ++d) {
        if (shape[d] % tile_size != 0) {
            return false;
        }
    }
    return true;
}

template <size_t nDim>
Index<nDim> GetNumberOfTiles(const Index<nDim> &shape, uint64_t tile_size) {
    Index<nDim> n_tiles;
    for (size_t d = 0; d < nDim; ++d) {
        if (shape[d] % tile_size != 0) {
            throw std::invalid_argument("dimensions should be multiples of " +
                                        std::to_string(tile_size));
        }
        n_tiles[d] = shape[d] / tile_size;
    }
    return n_tiles;
}

template <size_t nDim> struct GetNextConfigurationParameters {
    uint32_t p_no_add;
    UdhTransitionProbsArray<nDim> transition_probs;
    uint64_t measure_every;
    uint64_t n_wolff;
    uint64_t n_metropolis;
};

struct MonteCarloDurations {
    HiResTimer::Clock::duration::rep flip_cluster;
    HiResTimer::Clock::duration::rep clear_flag;
    HiResTimer::Clock::duration::rep metropolis_sweep;
};

template <size_t nDim>
MonteCarloDurations
GetNextConfiguration(const GetNextConfigurationParameters<nDim> &params,
                     Tensor<nDim, UdhSpin> *lattice,
                     std::queue<Index<nDim>> *queue, std::mt19937 *rng) {
    HiResTimer flip_cluster_timer, clear_flag_timer, metropolis_sweep_timer;
    for (uint64_t i1 = 0; i1 < params.measure_every; ++i1) {
        for (uint64_t i2 = 0; i2 < params.n_wolff; ++i2) {
            const Index<nDim> i0 = GetRandomIndex(lattice->shape(), rng);
            if ((*lattice)[i0] != UdhSpinHole()) {
                flip_cluster_timer.start();
                FlipCluster(params.p_no_add, i0, lattice, rng, queue);
                flip_cluster_timer.stop();
                clear_flag_timer.start();
                ClearVisitedFlag(i0, lattice, queue);
                clear_flag_timer.stop();
            }
        }
        metropolis_sweep_timer.start();
        for (uint64_t i2 = 0; i2 < params.n_metropolis; ++i2) {
            UdhMetropolisSweep(params.transition_probs, lattice, rng);
        }
        metropolis_sweep_timer.stop();
    }
    return {.flip_cluster = flip_cluster_timer.elapsed().count(),
            .clear_flag = clear_flag_timer.elapsed().count(),
            .metropolis_sweep = metropolis_sweep_timer.elapsed().count()};
}

template <size_t nDim>
Tensor<nDim, UdhSpin>
GetInitialConfiguration(const Index<nDim> &shape,
                        const GetNextConfigurationParameters<nDim> &params,
                        std::mt19937 *rng) {

    Tensor<nDim, UdhSpin> tile(HypercubeShape<nDim>(2), UdhSpinDown());
    std::queue<Index<nDim>> queue;
    while (ShapeCanBeTiled(shape, 2 * tile.shape(0))) {
        tile = TileTensor(tile, HypercubeShape<nDim>(2));
        for (int i = 0; i < 8; ++i) {
            GetNextConfiguration(params, &tile, &queue, rng);
        }
    }
    return TileTensor(tile, GetNumberOfTiles<nDim>(shape, tile.shape(0)));
}

template <size_t nDim> int Run(const UdhParameters &parameters) {
    const std::string out_file_name = parameters.id() + ".udh";
    std::ofstream out_file(out_file_name, std::ios_base::binary);
    if (!out_file.good()) {
        std::cout << "Failed to open\"" << out_file_name << "\" for output"
                  << std::endl;
        return -1;
    }

    const std::string log_file_name = parameters.id() + ".log";
    std::ofstream log_file(log_file_name);
    if (!log_file.good()) {
        std::cout << "Failed to open\"" << log_file_name << "\" for output"
                  << std::endl;
        return -1;
    }

    Write(parameters, &out_file);
    out_file.flush();

    std::string parameters_str;
    google::protobuf::TextFormat::PrintToString(parameters, &parameters_str);
    log_file << parameters_str;
    log_file.flush();

    const Index<nDim> shape = GetStaticShape<nDim>(parameters);
    const GetNextConfigurationParameters<nDim>
        get_next_configuration_parameters{
            .p_no_add = GetNoAddProbabilityFromJ(parameters.j()),
            .transition_probs = ComputeUdhTransitionProbs<nDim>(
                parameters.j(), parameters.mu()),
            .measure_every = parameters.measure_every(),
            .n_wolff = parameters.n_wolff(),
            .n_metropolis = parameters.n_metropolis()};

    log_file << TimeString(std::time(nullptr)) << " Starting warm-up"
             << std::endl;

    std::mt19937 rng(parameters.seed());
    Tensor<nDim, UdhSpin> lattice =
        GetInitialConfiguration(shape, get_next_configuration_parameters, &rng);

    std::queue<Index<nDim>> queue;
    UdhObservables observables;
    ProgressIndicator progress_indicator(std::time(nullptr),
                                         parameters.n_measure());
    log_file << TimeString(std::time(nullptr)) << " Starting simulation"
             << std::endl;
    HiResTimer simulation_timer;
    simulation_timer.start();
    uint64_t serialize_duration = 0;
    using period = std::chrono::steady_clock::period;
    Throttle<std::chrono::steady_clock> throttle(
        std::chrono::steady_clock::duration(1 * period::den / period::num));
    for (uint64_t i0 = 0; i0 < parameters.n_measure(); ++i0) {
        HiResTimer measure_timer, serialize_timer;
        const auto durations = GetNextConfiguration<nDim>(
            get_next_configuration_parameters, &lattice, &queue, &rng);
        measure_timer.start();
        Measure(lattice, &observables);
        measure_timer.stop();

        observables.set_sequence_id(i0);
        observables.set_stamp(simulation_timer.elapsed().count());
        observables.set_flip_cluster_duration(durations.flip_cluster);
        observables.set_clear_flag_duration(durations.clear_flag);
        observables.set_metropolis_sweep_duration(durations.metropolis_sweep);
        observables.set_measure_duration(measure_timer.elapsed().count());
        observables.set_serialize_duration(serialize_duration);

        serialize_timer.start();
        Write(observables, &out_file);
        out_file.flush();
        serialize_timer.stop();
        serialize_duration = serialize_timer.elapsed().count();

        throttle([&log_file, i0, &progress_indicator]() {
            log_file << progress_indicator.string(std::time(nullptr), i0 + 1)
                     << std::endl;
        });
    }
    return 0;
}

} // namespace

int main(int argc, const char **argv) {
    UdhParameters parameters;
    try {
        if (!ParseArgs(argc, argv, &parameters)) {
            return -1;
        }
    } catch (std::exception e) {
        std::cout << e.what() << std::endl;
        return -1;
    }

    const std::map<uint32_t, int (*)(const UdhParameters &)> dispatch_table = {
        {2u, &Run<2>}, {3u, &Run<3>}, {4u, &Run<4>}, {5u, &Run<5>}};

    const auto pair = dispatch_table.find(parameters.shape().size());
    if (pair == dispatch_table.end()) {
        std::cout << "Unsupported number of dimensions: "
                  << parameters.shape().size() << std::endl;
        return -1;
    } else {
        return pair->second(parameters);
    }
}
