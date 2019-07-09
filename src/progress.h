#pragma once
#include <ctime>
#include <string>

struct ProgressString {
    ProgressString(const std::time_t start_time, const std::time_t now,
                   uint64_t progress, uint64_t max_progress)
        : start_time(start_time), now(now), progress(progress),
          max_progress(max_progress) {}
    std::time_t start_time;
    std::time_t now;
    uint64_t progress;
    uint64_t max_progress;
};

std::ostream &operator<<(std::ostream &strm, const ProgressString &ps) {
    const auto flags = strm.flags();
    strm << std::fixed << std::setprecision(2) << std::setw(6)
         << 100.0 * ps.progress / ps.max_progress << "%";
    if (ps.progress > 0) {
        const double elapsed = std::difftime(ps.now, ps.start_time);
        const std::time_t eta =
            ps.start_time + ps.max_progress * elapsed / ps.progress;
        const char *time_string = std::asctime(std::localtime(&eta));
        strm << " ETA " << std::string(time_string, time_string + 24);
    }
    strm.flags(flags);
    return strm;
}
