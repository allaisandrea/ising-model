#include "Node.h"
#include "lattice.h"
#include "observables.h"
#include "simulation.pb.h"
#include <boost/program_options.hpp>
#include <Eigen/Core>
#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <random>
#include <chrono>

struct Arguments {
    std::string outputFileName;
    std::vector<size_t> shape;
    double prob;
    size_t measureEvery;
    size_t runFor;
    size_t saveEvery;
};

std::ostream &operator<<(std::ostream &os, const Arguments &args) {
    std::cout << "outputFileName: " << args.outputFileName << std::endl;
    std::cout << "shape: ";
    for (const auto &i : args.shape) {
        std::cout << i << " ";
    }
    std::cout << std::endl;

    std::cout << "prob: " << args.prob << std::endl;
    std::cout << "measureEvery: " << args.measureEvery << std::endl;
    std::cout << "runFor: " << args.runFor << std::endl;
    std::cout << "saveEvery: " << args.saveEvery << std::endl;

    return os;
}

bool ParseArgs(int argc, const char *argv[], Arguments *args) {
    using namespace boost::program_options;

    options_description description{"Options"};
    description.add_options()("help,h", "Show usage")(
        "output", value<std::string>(&(args->outputFileName))->required())(
        "shape",
        value<std::vector<size_t>>(&(args->shape))->required()->multitoken())(
        "prob", value<double>(&(args->prob))->required())(
        "measure-every", value<size_t>(&(args->measureEvery))->required())(
        "run-for", value<size_t>(&(args->runFor))->required())(
        "save-every", value<size_t>(&(args->saveEvery))->required());

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

void ToProtobuf(const Arguments &args, pb::Simulation *sim) {

    sim->clear_shape();
    for (const auto &shape_i : args.shape) {
        sim->add_shape(shape_i);
    }
    sim->set_prob(args.prob);
    sim->set_measure_every(args.measureEvery);
}

void AppendToProtobuf(const Observables &obs, double stamp,
                      pb::Simulation *sim) {
    pb::Observables *pb_obs = sim->add_observables();
    pb_obs->set_stamp(stamp);
    pb_obs->set_representative_state(obs.representativeState);
    for (const auto &c : obs.stateCount) {
        pb_obs->add_state_count(c);
    }
    pb_obs->set_magnetization(obs.magnetization);
}

bool AtomicallyAcquired(Node *node) {
    return __sync_fetch_and_or(node, 128) & 128;
}

bool Visited(Node node) { return node & 128; }

void MarkVisited(Node *node) { (*node) |= 128; }

void ClearVisitedFlag(Node *node) { (*node) &= (~128); }

bool Parallel(Node node1, Node node2) { return ((node1 ^ node2) & 1) == 0; }

void Flip(Node *node) { (*node) ^= 1; }

template <size_t nDim, typename Generator>
void FlipCluster(double prob, const Index<nDim> &i0, const Index<nDim> &shape,
                 Node *nodes, size_t *clusterSize, Generator *rng,
                 std::queue<Index<nDim>> *queue) {

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

void ClearVisitedFlag(Node *begin, Node *end) {
    for (Node *node = begin; node < end; ++node) {
        ClearVisitedFlag(node);
    }
}

bool Serialize(const pb::Simulation &simulationPb,
               const std::string &fileName) {
    std::ofstream file;
    file.open(fileName, std::ios_base::trunc | std::ios_base::binary);
    if (!simulationPb.SerializeToOstream(&file)) {
        std::cerr << "Unable to save simulation protobuf to \"" << fileName
                  << "\"." << std::endl;
        return false;
    }
    return true;
}

template <size_t nDim> void run(const Arguments &args) {
    std::cout << "Arguments: " << std::endl;
    std::cout << args << std::endl;

    std::mt19937 rng;

    pb::Simulation simulationPb;
    ToProtobuf(args, &simulationPb);
    Serialize(simulationPb, args.outputFileName);

    Index<nDim> shape;
    for (size_t i = 0; i < nDim; ++i) {
        shape[i] = args.shape.at(i);
    }
    std::vector<Node> nodes(GetSize(shape), 0);
    Index<nDim> i0;
    std::queue<Index<nDim>> queue;
    Observables observables;

    const auto startTime = std::chrono::steady_clock::now();
    const auto hiResStartTime = std::chrono::high_resolution_clock::now();
    auto timeOfLastSave = std::chrono::steady_clock::now();

    MeasureWorkspace measureWorkspace;

    const std::vector<size_t> waveNumbers = {1,  2,  3,  4,  6,  8,   12,  16,
                                             24, 32, 48, 64, 96, 128, 192, 256};
    std::array<Eigen::MatrixXf, 2> ftTables;
    for (size_t i = 0; i < 2; ++i) {
        MakeFourierTable(shape[i], waveNumbers, &ftTables[i]);
    }

    while (std::chrono::steady_clock::now() - startTime <
           std::chrono::seconds(args.runFor)) {

        size_t clusterSize = 0;
        while (clusterSize < args.measureEvery * nodes.size()) {
            GetRandomIndex(shape, &i0, &rng);
            FlipCluster(args.prob, i0, shape, nodes.data(), &clusterSize, &rng,
                        &queue);
            ClearVisitedFlag(nodes.data(), nodes.data() + nodes.size());
        }
        Measure(shape, nodes.data(), ftTables, &observables, &measureWorkspace);

        double stamp = (std::chrono::high_resolution_clock::now() -
                        hiResStartTime).count();

        AppendToProtobuf(observables, stamp, &simulationPb);

        auto now = std::chrono::steady_clock::now();
        if (now - timeOfLastSave > std::chrono::seconds(args.saveEvery)) {
            Serialize(simulationPb, args.outputFileName);
            timeOfLastSave = now;
        }
    };
    Serialize(simulationPb, args.outputFileName);
}

int main(int argc, const char *argv[]) {
    Arguments args;
    bool success = ParseArgs(argc, argv, &args);
    if (!success) {
        return -1;
    }

    std::vector<void (*)(const Arguments &)> dispatchTable = {
        &run<0>, &run<1>, &run<2>, &run<3>, &run<4>, &run<5>};

    dispatchTable.at(args.shape.size())(args);

    return 0;
}