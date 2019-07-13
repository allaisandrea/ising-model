#include <fstream>

#include "progress.h"
#include "tensor.h"
#include "timer.h"
#include "udh_arguments.h"
#include "udh_io.h"
#include "udh_measure.h"
#include "udh_metropolis_algorithm.h"
#include "udh_spin.h"
#include "wolff_algorithm.h"

template <size_t nDim>
Index<nDim> GetNumberOfTiles(const udh::Parameters &parameters,
                             uint64_t tile_size) {
    Index<nDim> n_tiles;
    for (size_t d = 0; d < nDim; ++d) {
        if (parameters.shape(d) % tile_size != 0) {
            throw std::invalid_argument("dimensions should be multiples of " +
                                        std::to_string(tile_size));
        }
        n_tiles[d] = parameters.shape(d) / tile_size;
    }
    return n_tiles;
}

template <size_t nDim> int Run(const udh::Parameters &parameters) {
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
    log_file << TimeString(std::time(nullptr)) << " Starting warm-up"
             << std::endl;

    Write(parameters, &out_file);
    std::mt19937 rng(parameters.seed());

    const UdhTransitionProbsArray<nDim> transition_probs =
        ComputeUdhTransitionProbs<nDim>(parameters.j(), parameters.mu());

    Tensor<nDim, UdhSpin> lattice4(HypercubeShape<nDim>(4), UdhSpinDown());
    for (uint64_t i = 0; i < 16; ++i) {
        UdhMetropolisSweep(transition_probs, &lattice4, &rng);
    }

    Tensor<nDim, UdhSpin> lattice8 =
        TileTensor(lattice4, HypercubeShape<nDim>(2));
    for (uint64_t i = 0; i < 512; ++i) {
        UdhMetropolisSweep(transition_probs, &lattice8, &rng);
    }

    const Index<nDim> n_tiles = GetNumberOfTiles<nDim>(parameters, 8);
    Tensor<nDim, UdhSpin> lattice = TileTensor(lattice8, n_tiles);

    const uint32_t p_no_add = GetNoAddProbabilityFromJ(parameters.j());

    std::queue<Index<nDim>> queue;
    udh::Observables observables;
    ProgressIndicator progress_indicator(std::time(nullptr),
                                         parameters.n_measure());
    log_file << TimeString(std::time(nullptr)) << " Starting simulation"
             << std::endl;
    HiResTimer simulation_timer;
    simulation_timer.start();
    for (uint64_t i0 = 0; i0 < parameters.n_measure(); ++i0) {
        HiResTimer flip_cluster_timer, clear_flag_timer, metropolis_sweep_timer,
            measure_timer, serialize_timer;
        for (uint64_t i1 = 0; i1 < parameters.measure_every(); ++i1) {
            for (uint64_t i2 = 0; i2 < parameters.n_wolff(); ++i2) {
                const Index<nDim> i0 = GetRandomIndex(lattice.shape(), &rng);
                if (lattice[i0] != UdhSpinHole()) {
                    flip_cluster_timer.start();
                    FlipCluster(p_no_add, i0, &lattice, &rng, &queue);
                    flip_cluster_timer.stop();
                    clear_flag_timer.start();
                    ClearVisitedFlag(i0, &lattice, &queue);
                    clear_flag_timer.stop();
                }
            }
            metropolis_sweep_timer.start();
            for (uint64_t i2 = 0; i2 < parameters.n_metropolis(); ++i2) {
                UdhMetropolisSweep(transition_probs, &lattice, &rng);
            }
            metropolis_sweep_timer.stop();
        }
        measure_timer.start();
        Measure(lattice, &observables);
        measure_timer.stop();

        observables.set_stamp(simulation_timer.elapsed().count());
        observables.set_flip_cluster_duration(
            flip_cluster_timer.elapsed().count());
        observables.set_clear_flag_duration(clear_flag_timer.elapsed().count());
        observables.set_metropolis_sweep_duration(
            metropolis_sweep_timer.elapsed().count());
        observables.set_measure_duration(measure_timer.elapsed().count());
        observables.set_serialize_duration(serialize_timer.elapsed().count());

        serialize_timer.start();
        Write(observables, &out_file);
        serialize_timer.stop();

        log_file << progress_indicator.string(std::time(nullptr), i0 + 1)
                 << std::endl;
    }
    return 0;
}

int main(int argc, const char **argv) {
    udh::Parameters parameters;
    try {
        if (!ParseArgs(argc, argv, &parameters)) {
            return -1;
        }
    } catch (std::exception e) {
        std::cout << e.what() << std::endl;
        return -1;
    }

    const std::map<uint32_t, int (*)(const udh::Parameters &)> dispatch_table =
        {{2u, &Run<2>}, {3u, &Run<3>}, {4u, &Run<4>}, {5u, &Run<5>}};

    const auto pair = dispatch_table.find(parameters.shape().size());
    if (pair == dispatch_table.end()) {
        std::cout << "Unsupported number of dimensions: "
                  << parameters.shape().size() << std::endl;
        return -1;
    } else {
        return pair->second(parameters);
    }
}
