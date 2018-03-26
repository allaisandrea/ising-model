#include "Node.h"
#include "lattice.h"
#include "observables.h"
#include <atomic>
#include <boost/program_options.hpp>
#include <chrono>
#include <eigen3/Eigen/Core>
#include <fstream>
#include <iostream>
#include <queue>
#include <random>
#include <signal.h>
#include <vector>

struct Arguments {
    std::string outputFileName;
    std::vector<uint32_t> shape;
    double prob;
    uint64_t seed;
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

    std::cout << "prob: " << args.prob << std::endl;
    std::cout << "seed: " << args.seed << std::endl;
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
    return Serialize(version, file) &&
           Serialize(args.shape.data(), args.shape.size(), file) &&
           Serialize(args.prob, file) && Serialize(args.seed, file) &&
           Serialize(args.waveNumbers.data(), args.waveNumbers.size(), file) &&
           Serialize(args.measureEvery, file) &&
           Serialize(args.nMeasure, file) &&
           Serialize(args.tag.data(), args.tag.size(), file);
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

bool ParseArgs(int argc, const char *argv[], Arguments *args) {
    using namespace boost::program_options;

    options_description description{"Options"};
    description.add_options()("help,h", "Show usage")(
        "output", value<std::string>(&(args->outputFileName))->required())(
        "shape",
        value<std::vector<uint32_t>>(&(args->shape))->required()->multitoken())(
        "prob", value<double>(&(args->prob))->required())(
        "seed", value<uint64_t>(&(args->seed))->required())(
        "measure-every", value<uint64_t>(&(args->measureEvery))->required())(
        "n-measure", value<uint64_t>(&(args->nMeasure))->required())(
        "tag", value<std::string>(&(args->tag))->required());

    variables_map vm;
    store(parse_command_line(argc, argv, description), vm);

    if (vm.count("help") > 0) {
        std::cout << description << std::endl;
        return false;
    }
    try {
        notify(vm);
    } catch (std::exception &e) {
        std::cout << e.what() << std::endl;
        std::cout << description << std::endl;
        return false;
    }
    return true;
}

bool AtomicallyAcquired(Node *node) {
    return __sync_fetch_and_or(node, 128) & 128;
}

bool Visited(Node node) { return node & 128; }

void MarkVisited(Node *node) { (*node) |= 128; }

void ClearVisitedFlag(Node *node) { (*node) &= (~128); }

void Flip(Node *node) { (*node) ^= 1; }

template <size_t nDim, typename Generator, typename Queue>
void FlipCluster(double prob, const Index<nDim> &i0, const Index<nDim> &shape,
                 Node *nodes, size_t *clusterSize, Generator *rng,
                 Queue *queue) {

    const typename Generator::result_type iProb =
        rng->min() +
        typename Generator::result_type(prob * (rng->max() - rng->min()));

    queue->emplace(i0);
    MarkVisited(nodes + GetScalarIndex(i0, shape));

    while (!queue->empty()) {
        Index<nDim> &i = queue->front();
        Node *node0 = nodes + GetScalarIndex(i, shape);
        for (size_t d = 0; d < nDim; ++d) {
            const typename Index<nDim>::value_type i_d = i[d];
            const typename Index<nDim>::value_type s_d = shape[d];
            for (size_t dir = 0; dir < 4; dir += 2) {
                i[d] = (i_d + s_d + dir - 1) % s_d;
                Node *node1 = nodes + GetScalarIndex(i, shape);
                const bool add = !Visited(*node1) && Parallel(*node0, *node1) &&
                                 (*rng)() > iProb;
                if (add) {
                    queue->emplace(i);
                    MarkVisited(node1);
                }
            }
            i[d] = i_d;
        }

        Flip(node0);
        queue->pop();
        ++(*clusterSize);
    }
}

template <size_t nDim, typename Queue>
void ClearVisitedFlag(const Index<nDim> &i0, const Index<nDim> &shape,
                      Node *nodes, Queue *queue) {

    queue->emplace(i0);
    ClearVisitedFlag(nodes + GetScalarIndex(i0, shape));

    while (!queue->empty()) {
        Index<nDim> &i = queue->front();
        for (size_t d = 0; d < nDim; ++d) {
            const typename Index<nDim>::value_type i_d = i[d];
            const typename Index<nDim>::value_type s_d = shape[d];
            for (size_t dir = 0; dir < 4; dir += 2) {
                i[d] = (i_d + s_d + dir - 1) % s_d;
                Node *node1 = nodes + GetScalarIndex(i, shape);
                if (Visited(*node1)) {
                    queue->emplace(i);
                    ClearVisitedFlag(node1);
                }
            }
            i[d] = i_d;
        }
        queue->pop();
    }
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
    std::vector<Node> nodes(GetSize(shape), 0);
    std::queue<Index<nDim>> queue;
    Observables observables;

    for (size_t i = 0; i < nodes.size(); ++i) {
        nodes[i] = uint32_t(1664525ul * i + 1013904223ul) & 1u;
    }

    MeasureWorkspace measureWorkspace;

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
    for (uint64_t it = 0; it < args.nMeasure; ++it) {
        auto time1 = std::chrono::high_resolution_clock::now();

        auto time2 = time1;
        std::chrono::high_resolution_clock::duration flipClusterDuration(0);
        std::chrono::high_resolution_clock::duration clearFlagDuration(0);
        size_t cumulativeClusterSize = 0;
        for (uint64_t it1 = 0; it1 < args.measureEvery && !quit.load(); ++it1) {
            const auto i0 = GetRandomIndex(shape, &rng);
            FlipCluster(args.prob, i0, shape, nodes.data(),
                        &cumulativeClusterSize, &rng, &queue);
            auto time3 = std::chrono::high_resolution_clock::now();
            flipClusterDuration += time3 - time2;

            ClearVisitedFlag(i0, shape, nodes.data(), &queue);
            time2 = std::chrono::high_resolution_clock::now();
            clearFlagDuration += time2 - time3;
        }
        if (quit.load())
            break;

        auto time4 = std::chrono::high_resolution_clock::now();
        Measure(shape, nodes.data(), ftTables, &observables, &measureWorkspace);

        auto time5 = std::chrono::high_resolution_clock::now();
        observables.cumulativeClusterSize = cumulativeClusterSize;
        observables.flipClusterDuration = flipClusterDuration.count();
        observables.clearFlagDuration = clearFlagDuration.count();
        observables.measureDuration = (time5 - time4).count();
        observables.serializeDuration = (time1 - time0).count();
        time0 = time5;

        if (!Serialize(observables, &outFile)) {
            std::cerr << "Unable to serialize observables" << std::endl;
        }
        std::cout << "Iteration " << it + 1 << " done" << std::endl;
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
