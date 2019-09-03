#pragma once
#include "udh.pb.h"
#include "udh_io.h"
#include <fstream>
#include <set>

struct ParametersComp {
    bool operator()(const UdhParameters &p1, const UdhParameters &p2) const {
        if (p1.shape().size() < p2.shape().size()) {
            return true;
        } else if (p1.shape().size() > p2.shape().size()) {
            return false;
        } else if (p1.mu() < p2.mu()) {
            return true;
        } else if (p1.mu() > p2.mu()) {
            return false;
        } else if (p1.j() < p2.j()) {
            return true;
        } else if (p1.j() > p2.j()) {
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

template <typename FilenameIt>
std::vector<std::pair<std::string, UdhParameters>>
ReadUdhParametersFromFiles(FilenameIt begin, FilenameIt end,
                           OpenFunctionT open_function = &OpenFile) {
    using Pair = std::pair<std::string, UdhParameters>;
    std::vector<Pair> result;
    for (FilenameIt file_name = begin; file_name != end; ++file_name) {
        auto file = open_function(*file_name);
        result.emplace_back(Pair{*file_name, UdhParameters{}});
        if (!Read(&result.back().second, file.get())) {
            throw std::runtime_error("Unable to read parameters from file \"" +
                                     std::string(*file_name) + "\"");
        }
    }
    return result;
}

inline void
SortParametersArray(std::vector<std::pair<std::string, UdhParameters>> *pairs) {
    using Pair = std::pair<std::string, UdhParameters>;
    std::sort(pairs->begin(), pairs->end(), [](const Pair &p1, const Pair &p2) {
        return ParametersComp()(p1.second, p2.second);
    });
}
