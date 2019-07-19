#pragma once
#include "udh_io.h"
#include "udh_parameters_set.h"
#include <fstream>
#include <stdexcept>

class UdhFileGroup {
  public:
    struct Entry {
        std::string file_name;
        uint64_t read_every;
    };

    UdhFileGroup(const std::vector<Entry> &entries, uint64_t skip_first_n = 0,
                 OpenFunctionT open_function = &OpenFile);

    bool NextObservables(UdhObservables *observables);
    uint64_t CountObservables();
    const UdhParameters &parameters() const { return _parameters; }

  private:
    static UdhParameters GetParameters(const std::vector<Entry> &entries,
                                       OpenFunctionT open_function);
    const UdhParameters _parameters;
    const uint64_t _skip_first_n;
    const std::vector<Entry> _entries;
    size_t _current_entry;
    std::unique_ptr<std::istream> _current_file;
    OpenFunctionT _open_function;

    bool OpenNextFile();

    const Entry &current_entry() { return _entries.at(_current_entry); }
};

inline UdhParameters
UdhFileGroup::GetParameters(const std::vector<Entry> &entries,
                            OpenFunctionT open_function) {
    if (entries.empty()) {
        return {};
    } else {
        auto file = open_function(entries.front().file_name);
        UdhParameters params;
        Read(&params, file.get());
        return params;
    }
}

inline UdhFileGroup::UdhFileGroup(const std::vector<Entry> &entries,
                                  uint64_t skip_first_n,
                                  OpenFunctionT open_function)
    : _parameters(GetParameters(entries, open_function)),
      _skip_first_n(skip_first_n), _entries(entries),
      _current_entry(entries.size()), _open_function(open_function) {}

inline bool UdhFileGroup::OpenNextFile() {
    while (true) {
        ++_current_entry;
        if (_current_entry >= _entries.size()) {
            return false;
        }
        _current_file = _open_function(current_entry().file_name);
        if (!Skip(1, _current_file.get())) {
            throw std::runtime_error(
                "File \"" + current_entry().file_name +
                "\" does not contain the parameters header");
        }
        const uint64_t n_skip = _skip_first_n * current_entry().read_every;
        if (Skip(n_skip, _current_file.get())) {
            return true;
        }
    }
    return false;
}

inline bool UdhFileGroup::NextObservables(UdhObservables *observables) {
    if (_current_entry >= _entries.size()) {
        _current_entry = -1;
        if (!OpenNextFile()) {
            return false;
        }
    }
    while (true) {
        const uint64_t n_skip = current_entry().read_every - 1;
        if (!Skip(n_skip, _current_file.get())) {
            if (!OpenNextFile()) {
                return false;
            }
            continue;
        }
        if (!Read(observables, _current_file.get())) {
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
        if (!Skip(n_skip, _current_file.get())) {
            if (!OpenNextFile()) {
                return result;
            }
            continue;
        }
        ++result;
    }
    return result;
}

inline std::vector<std::vector<UdhFileGroup::Entry>> MakeUdhFileGroupEntries(
    std::vector<std::pair<std::string, UdhParameters>> pairs) {
    SortParametersArray(&pairs);
    auto group_begin = pairs.begin();
    std::vector<std::vector<UdhFileGroup::Entry>> result;
    while (group_begin != pairs.end()) {
        result.emplace_back();
        std::vector<UdhFileGroup::Entry> &entries = result.back();
        entries.emplace_back(UdhFileGroup::Entry{group_begin->first, 1});
        auto pair = std::next(group_begin);
        for (; pair != pairs.end() &&
               OutputCanBeJoined(group_begin->second, pair->second);
             ++pair) {
            const uint64_t read_every = group_begin->second.measure_every() /
                                        pair->second.measure_every();
            entries.emplace_back(UdhFileGroup::Entry{pair->first, read_every});
        }
        group_begin = pair;
    }
    return result;
}

inline std::vector<UdhFileGroup>
GroupFiles(const std::vector<std::pair<std::string, UdhParameters>> &pairs,
           uint64_t skip_first_n) {
    const auto entries_array = MakeUdhFileGroupEntries(pairs);
    std::vector<UdhFileGroup> groups;
    for (const auto entries : entries_array) {
        groups.emplace_back(entries, skip_first_n);
    }
    return groups;
}
