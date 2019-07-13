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
    std::cout << "OpenNextFile" << std::endl;
    _current_file.close();
    ++_current_entry;
    if (_current_entry < _entries.size()) {
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
        return true;
    } else {
        return false;
    }
}

inline bool UdhFileGroup::NextObservables(udh::Observables *observables) {
    if (_current_entry >= _entries.size()) {
        _current_entry = -1;
        if (!OpenNextFile()) {
            return false;
        }
    }
    const uint64_t n_skip = current_entry().read_every - 1;
    while (!Skip(n_skip, &_current_file) && OpenNextFile()) {
    }
    while (!Read(observables, &_current_file) && OpenNextFile()) {
        const uint64_t n_skip =
            (current_entry().read_every - 1) * _skip_first_n;
        while (!Skip(n_skip, &_current_file) && OpenNextFile()) {
        }
    }
    return _current_entry < _entries.size();
}

template <typename FilenameIt>
inline std::vector<UdhFileGroup> GroupFiles(FilenameIt begin, FilenameIt end) {
    const ParametersSet parameters_set = ParametersSetFromFileNames(begin, end);
    auto group_begin = parameters_set.begin();
    std::vector<UdhFileGroup> result;
    while (group_begin != parameters_set.end()) {
        auto group_end = std::next(group_begin);
        while (group_end != parameters_set.end() &&
               OutputCanBeJoined(*group_begin, *group_end)) {
            ++group_end;
        }
        result.emplace_back(UdhFileGroup(group_begin, group_end));
        group_begin = group_end;
    }
    return result;
}
