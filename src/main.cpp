#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <random>
#include <signal.h>
#include <vector>

#include <boost/program_options.hpp>

#include "UdSpin.h"
#include "lattice.h"
#include "observables.h"
#include "wolff_algorithm.h"

struct Arguments {
    std::string outputFileName;
    std::vector<uint32_t> shape;
    std::mt19937::result_type iProb;
    std::mt19937::result_type seed;
    std::vector<uint32_t> waveNumbers;
    uint64_t measureEvery;
    uint64_t nMeasure;
    std::string tag;
};

std::ostream &operator<<(std::ostream &os, const Arguments &args) {
    std::cout << "outputFileName: " << args.outputFileName << std::endl;
    std::cout << "shape: ";
    for (const auto &i : args.shape) {
        std::cout << i << " ";
    }
    std::cout << std::endl;

    std::cout << "iProb: " << std::hex << args.iProb << std::dec << std::endl;
    std::cout << "seed: " << std::hex << args.seed << std::dec << std::endl;
    std::cout << "measureEvery: " << args.measureEvery << std::endl;
    std::cout << "nMeasure: " << args.nMeasure << std::endl;
    std::cout << "tag: " << args.tag << std::endl;

    return os;
}

template <typename T> bool Serialize(const T &x, std::ofstream *file) {
    file->write((const char *)(&x), sizeof(T));
    if (!file->good()) {
        return false;
    }
    return true;
}

template <typename T>
bool Serialize(const T *v, uint64_t n, std::ofstream *file) {
    if (!Serialize<uint64_t>(n, file)) {
        return false;
    }
    file->write((const char *)(v), n * sizeof(T));
    if (!file->good()) {
        return false;
    }
    return true;
}

template <typename T>
bool Serialize(const T *v, uint64_t n1, uint64_t n2, std::ofstream *file) {
    if (!Serialize<uint64_t>(n1, file)) {
        return false;
    }
    if (!Serialize<uint64_t>(n2, file)) {
        return false;
    }
    file->write((const char *)(v), n1 * n2 * sizeof(T));
    if (!file->good()) {
        return false;
    }
    return true;
}

bool Serialize(const Arguments &args, std::ofstream *file) {
    const uint64_t version = 2;
    // clang-format off
    return Serialize(version, file) &&
           Serialize(args.shape.data(), args.shape.size(), file) &&
           Serialize<uint64_t>(args.iProb, file) &&
           Serialize<uint64_t>(args.seed, file) &&
           Serialize(args.waveNumbers.data(), args.waveNumbers.size(), file) &&
           Serialize(args.measureEvery, file) &&
           Serialize(args.nMeasure, file) &&
           Serialize(args.tag.data(), args.tag.size(), file);
    // clang-format on
}

bool Serialize(const Observables &obs, std::ofstream *file) {
    return Serialize(obs.flipClusterDuration, file) &&
           Serialize(obs.clearFlagDuration, file) &&
           Serialize(obs.measureDuration, file) &&
           Serialize(obs.serializeDuration, file) &&
           Serialize(obs.cumulativeClusterSize, file) &&
           Serialize(obs.upCount, file) && Serialize(obs.parallelCount, file) &&
           Serialize(obs.fourierTransform2d.data(),
                     obs.fourierTransform2d.rows(),
                     obs.fourierTransform2d.cols(), file);
}

bool ProbabilityStringIsValid(const std::string &probStr) {
    if (probStr.empty() || probStr.size() > std::mt19937::word_size / 4) {
        return false;
    }
    for (const auto &c : probStr) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) {
            return false;
        }
    }
    return true;
}

bool ParseArgs(int argc, const char *argv[], Arguments *args) {
    using namespace boost::program_options;

    std::string probStr;
    options_description description{"Options"};
    description.add_options()("help,h", "Show usage")(
        "output", value<std::string>(&(args->outputFileName))->required())(
        "shape",
        value<std::vector<uint32_t>>(&(args->shape))->required()->multitoken())(
        "i-prob", value<std::string>(&(probStr))->required())(
        "seed", value<std::mt19937::result_type>(&(args->seed))->required())(
        "measure-every", value<uint64_t>(&(args->measureEvery))->required())(
        "n-measure", value<uint64_t>(&(args->nMeasure))->required())(
        "tag", value<std::string>(&(args->tag))->required());

    variables_map vm;
    try {
        store(parse_command_line(argc, argv, description), vm);
    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
        std::cout << description << std::endl;
        return false;
    }

    if (vm.count("help") > 0) {
        std::cout << description << std::endl;
        return false;
    }
    try {
        notify(vm);
    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
        std::cout << description << std::endl;
        return false;
    }

    if (!ProbabilityStringIsValid(probStr)) {
        std::cout << "Unable to parse \"" << probStr << "\" as probability"
                  << std::endl;
        return false;
    }

    probStr.resize(std::mt19937::word_size / 4, '0');
    args->iProb = std::stoull(probStr, 0, 16);
    return true;
}

std::atomic<bool> quit(false); // signal flag

template <size_t nDim> void run(const Arguments &args) {
    std::cout << "Arguments: " << std::endl;
    std::cout << args << std::endl;

    std::mt19937 rng(args.seed);

    Index<nDim> shape;
    for (size_t i = 0; i < nDim; ++i) {
        shape[i] = args.shape.at(i);
    }
    Lattice<nDim, UdSpin> lattice(shape, UdSpin{0});
    std::queue<Index<nDim>> queue;
    Observables observables;

    std::array<Eigen::MatrixXf, 2> ftTables;
    for (size_t i = 0; i < 2; ++i) {
        MakeFourierTable(shape[i], args.waveNumbers, &ftTables[i]);
    }

    std::ofstream outFile(args.outputFileName,
                          std::ios_base::trunc | std::ios_base::binary);
    if (!outFile.good()) {
        std::cerr << "Unable to open \"" << args.outputFileName
                  << "\" for output." << std::endl;
        return;
    }
    if (!Serialize(args, &outFile)) {
        std::cerr << "Unable to serialize arguments to \""
                  << args.outputFileName << "\"." << std::endl;
        return;
    }
    outFile.flush();

    std::cout << "Begin simulation" << std::endl;
    auto time0 = std::chrono::high_resolution_clock::now();
    double step0MeanDuration = 0.0;
    for (uint64_t iStep0 = 0; iStep0 < args.nMeasure; ++iStep0) {
        const auto time1 = std::chrono::high_resolution_clock::now();

        auto time2 = time1;
        std::chrono::high_resolution_clock::duration flipClusterDuration(0);
        std::chrono::high_resolution_clock::duration clearFlagDuration(0);
        size_t cumulativeClusterSize = 0;
        for (uint64_t iStep1 = 0; iStep1 < args.measureEvery && !quit.load();
             ++iStep1) {
            const auto i0 = GetRandomIndex(shape, &rng);
            cumulativeClusterSize +=
                FlipCluster(args.iProb, i0, &lattice, &rng, &queue);
            const auto time3 = std::chrono::high_resolution_clock::now();
            flipClusterDuration += time3 - time2;

            ClearVisitedFlag(i0, &lattice, &queue);
            time2 = std::chrono::high_resolution_clock::now();
            clearFlagDuration += time2 - time3;
        }
        if (quit.load())
            break;

        const auto time4 = std::chrono::high_resolution_clock::now();
        Measure(lattice, ftTables, &observables);

        const auto time5 = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double> step0Duration = (time5 - time0);

        observables.cumulativeClusterSize = cumulativeClusterSize;
        observables.flipClusterDuration = flipClusterDuration.count();
        observables.clearFlagDuration = clearFlagDuration.count();
        observables.measureDuration = (time5 - time4).count();
        observables.serializeDuration = (time1 - time0).count();
        time0 = time5;

        if (!Serialize(observables, &outFile)) {
            std::cerr << "Unable to serialize observables" << std::endl;
        }
        outFile.flush();

        step0MeanDuration +=
            (step0Duration.count() - step0MeanDuration) / (iStep0 + 1);
        const double eta = (args.nMeasure - iStep0 - 1) * step0MeanDuration;

        const auto flags = std::cout.flags();
        std::cout << std::fixed << std::setprecision(1) << "Step " << iStep0 + 1
                  << "; " << step0Duration.count() << " seconds per step; ETA ";
        if (eta < 5 * 60) {
            std::cout << eta << " seconds." << std::endl;
        } else if (eta < 5 * 60 * 60) {
            std::cout << eta / 60 << " minutes." << std::endl;
        } else if (eta < 2 * 24 * 60 * 60) {
            std::cout << eta / (60 * 60) << " hours." << std::endl;
        } else {
            std::cout << eta / (24 * 60 * 60) << " days." << std::endl;
        }
        std::cout.flags(flags);
    }
    std::cout << "End" << std::endl;
}

void HandleSignal(int signal) {
    std::cerr << "Caught signal " << signal << std::endl;
    quit.store(true);
}

void RegisterSignalCallback() {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = &HandleSignal;
    sigfillset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
}

int main(int argc, const char *argv[]) {
    RegisterSignalCallback();

    Arguments args;
    bool success = ParseArgs(argc, argv, &args);
    if (!success) {
        return -1;
    }
    args.waveNumbers = {1,  2,  3,  4,  6,  8,   12,  16,
                        24, 32, 48, 64, 96, 128, 192, 256};

    std::vector<void (*)(const Arguments &)> dispatchTable = {
        &run<0>, &run<1>, &run<2>, &run<3>, &run<4>, &run<5>};

    dispatchTable.at(args.shape.size())(args);

    return 0;
}
