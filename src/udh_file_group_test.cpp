#include "mock_file_system.h"
#include "udh_file_group.h"

#include <array>
#include <gtest/gtest.h>
#include <random>

namespace {

struct TestFileDef {
    uint64_t n_observables;
    uint64_t read_every;
};

std::string MakeTestingFile(uint64_t id, const TestFileDef &test_file_def) {
    std::ostringstream strm;
    Write(UdhParameters(), &strm);
    UdhObservables observables;
    for (uint64_t i = 0; i < test_file_def.n_observables; ++i) {
        observables.set_sequence_id(i);
        observables.set_stamp(id);
        Write(observables, &strm);
    }
    return strm.str();
}

MockFileSystem
MakeTestingFiles(const std::vector<TestFileDef> &test_file_defs) {
    MockFileSystem mock_file_system;
    for (uint64_t id = 0; id < test_file_defs.size(); ++id) {
        mock_file_system.files[std::to_string(id)] =
            MakeTestingFile(id, test_file_defs[id]);
    }
    return mock_file_system;
}

struct ObsSummary {
    uint64_t sequence_id;
    uint64_t file_id;
};

bool operator==(const ObsSummary &lhs, const ObsSummary &rhs) {
    return lhs.sequence_id == rhs.sequence_id && lhs.file_id == rhs.file_id;
}

std::ostream &operator<<(std::ostream &strm, const ObsSummary &x) {
    strm << "(" << x.sequence_id << "," << x.file_id << ")";
    return strm;
}

std::vector<ObsSummary>
GetExpectedResult(uint64_t skip_first_n,
                  const std::vector<TestFileDef> &test_file_defs) {
    std::vector<ObsSummary> result;
    for (uint64_t id = 0; id < test_file_defs.size(); ++id) {
        const auto &def = test_file_defs[id];
        for (uint64_t i = (skip_first_n + 1) * def.read_every - 1;
             i < def.n_observables; i += def.read_every) {
            result.emplace_back(ObsSummary{i, id});
        }
    }
    return result;
}

std::vector<UdhFileGroup::Entry>
GetEntries(const std::vector<TestFileDef> &test_file_defs) {
    std::vector<UdhFileGroup::Entry> entries;
    for (uint64_t id = 0; id < test_file_defs.size(); ++id) {
        const auto &def = test_file_defs[id];
        entries.emplace_back(
            UdhFileGroup::Entry{std::to_string(id), def.read_every});
    }
    return entries;
}

std::array<std::vector<ObsSummary>, 2>
GetActualResult(uint64_t skip_first_n,
                const std::vector<TestFileDef> &test_file_defs) {
    MockFileSystem mock_file_system = MakeTestingFiles(test_file_defs);
    UdhFileGroup file_group(GetEntries(test_file_defs), skip_first_n,
                            mock_file_system);
    UdhObservables observables;
    std::vector<ObsSummary> result1;
    while (file_group.NextObservables(&observables)) {
        result1.emplace_back(
            ObsSummary{observables.sequence_id(), observables.stamp()});
    }

    std::vector<ObsSummary> result2;
    while (file_group.NextObservables(&observables)) {
        result2.emplace_back(
            ObsSummary{observables.sequence_id(), observables.stamp()});
    }
    return {result1, result2};
}

std::array<uint64_t, 2>
GetObservablesCount(uint64_t skip_first_n,
                    const std::vector<TestFileDef> &test_file_defs) {
    MockFileSystem mock_file_system = MakeTestingFiles(test_file_defs);
    UdhFileGroup file_group(GetEntries(test_file_defs), skip_first_n,
                            mock_file_system);
    return {file_group.CountObservables(), file_group.CountObservables()};
}

TEST(UdhFileGroup, UdhFileGroup) {
    std::mt19937 rng;
    for (uint64_t n_tries = 0; n_tries < 256; ++n_tries) {
        for (uint64_t n_files = 0; n_files < 3; ++n_files) {
            std::vector<TestFileDef> file_defs;
            for (uint64_t i_file = 0; i_file < n_files; ++i_file) {
                file_defs.emplace_back(TestFileDef{
                    std::uniform_int_distribution<uint64_t>(0, 16)(rng),
                    std::uniform_int_distribution<uint64_t>(1, 16)(rng)});
            }
            const uint64_t skip_first_n =
                std::uniform_int_distribution<uint64_t>(0, 4)(rng);
            const auto exp_result = GetExpectedResult(skip_first_n, file_defs);
            const auto act_results = GetActualResult(skip_first_n, file_defs);
            std::ostringstream strm;
            strm << "skip_first_n: " << skip_first_n << "\nfile_defs:\n"
                 << std::setw(5) << "id"
                 << "," << std::setw(15) << "n_observables"
                 << "," << std::setw(15) << "read_every"
                 << "\n";
            for (uint64_t id = 0; id < file_defs.size(); ++id) {
                const auto &def = file_defs[id];
                strm << std::setw(5) << id << "," << std::setw(15)
                     << def.n_observables << "," << std::setw(15)
                     << def.read_every << "\n";
            }
            EXPECT_EQ(act_results[0], exp_result) << strm.str();
            EXPECT_EQ(act_results[1], exp_result) << strm.str();

            const auto counts = GetObservablesCount(skip_first_n, file_defs);
            EXPECT_EQ(counts[0], exp_result.size());
            EXPECT_EQ(counts[1], exp_result.size());
        }
    }
}
} // namespace
