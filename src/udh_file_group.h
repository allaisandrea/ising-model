#pragma once
#include "udh_io.h"
#include "udh_parameters_set.h"
#include <fstream>
#include <stdexcept>

class UdhFileGroup {
    struct Entry {
        std::string file_name;
        uint64_t read_every;
    };
    const uint64_t _skip_first_n;
    const std::vector<Entry> _entries;
    const udh::Parameters _parameters;
    size_t _current_entry;
    std::ifstream _current_file;

    static std::vector<Entry> MakeEntries(ParametersSet::const_iterator begin,
                                          ParametersSet::const_iterator end);

    bool OpenNextFile();

    const Entry &current_entry() { return _entries.at(_current_entry); }

  public:
    UdhFileGroup(ParametersSet::const_iterator begin,
                 ParametersSet::const_iterator end, uint64_t skip_first_n = 0);

    bool NextObservables(udh::Observables *observables);
    uint64_t CountObservables();
    const udh::Parameters &parameters() const { return _parameters; }
};

inline std::vector<UdhFileGroup::Entry>
UdhFileGroup::MakeEntries(ParametersSet::const_iterator begin,
                          ParametersSet::const_iterator end) {
    if (begin == end) {
        throw std::invalid_argument("Cannot make empty file group");
    }
    std::vector<Entry> entries;
    for (auto params = begin; params != end; ++params) {
        if ((params != begin) && !OutputCanBeJoined(*begin, *params)) {
            throw std::invalid_argument("Outputs cannot be joined:\n" +
                                        GetCsvString(*begin) + "\n" +
                                        GetCsvString(*params));
        }
        entries.emplace_back(
            Entry{params->id() + ".udh",
                  begin->measure_every() / params->measure_every()});
        if (entries.back().read_every == 0) {
            throw std::runtime_error("read every is zero");
        }
    }
    return entries;
}

inline UdhFileGroup::UdhFileGroup(ParametersSet::const_iterator begin,
                                  ParametersSet::const_iterator end,
                                  uint64_t skip_first_n)
    : _skip_first_n(skip_first_n), _entries(MakeEntries(begin, end)),
      _parameters(*begin), _current_entry{_entries.size()} {}

inline bool UdhFileGroup::OpenNextFile() {
    _current_file.close();
    while (true) {
        ++_current_entry;
        if (_current_entry >= _entries.size()) {
            return false;
        }
        const std::string file_name = current_entry().file_name;
        _current_file.open(file_name);
        if (!_current_file.good()) {
            throw std::runtime_error("Unable to open file \"" + file_name +
                                     "\"");
        }
        if (!Skip(1, &_current_file)) {
            throw std::runtime_error(
                "File \"" + file_name +
                "\" does not contain the parameters header");
        }
        const uint64_t n_skip = _skip_first_n * current_entry().read_every;
        if (Skip(n_skip, &_current_file)) {
            return true;
        }
    }
    return false;
}

inline bool UdhFileGroup::NextObservables(udh::Observables *observables) {
    if (_current_entry >= _entries.size()) {
        _current_entry = -1;
        if (!OpenNextFile()) {
            return false;
        }
    }
    while (true) {
        const uint64_t n_skip = current_entry().read_every - 1;
        if (!Skip(n_skip, &_current_file)) {
            if (!OpenNextFile()) {
                return false;
            }
            continue;
        }
        if (!Read(observables, &_current_file)) {
            if (!OpenNextFile()) {
                return false;
            }
            continue;
        } else {
            break;
        }
    }
    return _current_entry < _entries.size();
}

inline uint64_t UdhFileGroup::CountObservables() {
    _current_entry = -1;
    if (!OpenNextFile()) {
        return 0;
    }
    uint64_t result = 0;
    while (true) {
        const uint64_t n_skip = current_entry().read_every;
        if (!Skip(n_skip, &_current_file)) {
            if (!OpenNextFile()) {
                return result;
            }
            continue;
        }
        ++result;
    }
    return result;
}

template <typename FilenameIt>
inline std::vector<UdhFileGroup> GroupFiles(FilenameIt begin, FilenameIt end,
                                            uint64_t skip_first_n = 0) {
    const ParametersSet parameters_set = ParametersSetFromFileNames(begin, end);
    auto group_begin = parameters_set.begin();
    std::vector<UdhFileGroup> result;
    while (group_begin != parameters_set.end()) {
        auto group_end = std::next(group_begin);
        while (group_end != parameters_set.end() &&
               OutputCanBeJoined(*group_begin, *group_end)) {
            ++group_end;
        }
        result.emplace_back(UdhFileGroup(group_begin, group_end, skip_first_n));
        group_begin = group_end;
    }
    return result;
}
