#pragma once
#include "udh_io.h"
#include "udh_simulation.pb.h"
#include <fstream>
#include <set>

struct ParametersComp {
    bool operator()(const UdhParameters &p1,
                    const UdhParameters &p2) const {
        if (p1.shape().size() < p2.shape().size()) {
            return true;
        } else if (p1.shape().size() > p2.shape().size()) {
            return false;
        } else if (p1.j() < p2.j()) {
            return true;
        } else if (p1.j() > p2.j()) {
            return false;
        } else if (p1.mu() < p2.mu()) {
            return true;
        } else if (p1.mu() > p2.mu()) {
            return false;
        }
        for (int i = 0; i < p1.shape().size(); ++i) {
            if (p1.shape(i) < p2.shape(i)) {
                return true;
            } else if (p1.shape(i) > p2.shape(i)) {
                return false;
            }
        }
        if (p1.n_wolff() < p2.n_wolff()) {
            return true;
        } else if (p1.n_wolff() > p2.n_wolff()) {
            return false;
        } else if (p1.n_metropolis() < p2.n_metropolis()) {
            return true;
        } else if (p1.n_metropolis() > p2.n_metropolis()) {
            return false;
        } else if (p1.measure_every() > p2.measure_every()) {
            return true;
        } else if (p1.measure_every() < p2.measure_every()) {
            return false;
        } else if (p1.seed() < p2.seed()) {
            return true;
        } else if (p1.seed() > p2.seed()) {
            return false;
        } else if (p1.id() < p2.id()) {
            return true;
        } else if (p1.id() > p2.id()) {
            return false;
        } else {
            return false;
        }
    }
};

using ParametersSet = std::set<UdhParameters, ParametersComp>;

inline bool OutputCanBeJoined(const UdhParameters &p1,
                              const UdhParameters &p2) {
    bool result = true;
    // clang-format off
    result = p1.shape().size() == p2.shape().size() && 
             p1.j() == p2.j() &&
             p1.mu() == p2.mu() && 
             p1.n_wolff() == p2.n_wolff() &&
             p1.n_metropolis() == p2.n_metropolis() &&
             p1.measure_every() % p2.measure_every() == 0 &&
             p1.seed() != p2.seed();
    // clang-format on
    if (!result)
        return false;
    for (int i = 0; i < p1.shape().size(); ++i) {
        result = result && p1.shape(i) == p2.shape(i);
    }
    return result;
}

template <typename FileNamesIt>
ParametersSet ParametersSetFromFileNames(FileNamesIt begin, FileNamesIt end) {
    ParametersSet result;
    UdhParameters parameters;
    for (FileNamesIt file_name = begin; file_name != end; ++file_name) {
        std::ifstream file(*file_name);
        if (!Read(&parameters, &file)) {
            throw std::runtime_error("Unable to read parameters from file \"" +
                                     std::string(*file_name) + "\"");
        }
        result.emplace(parameters);
    }
    return result;
}
