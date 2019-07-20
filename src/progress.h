#pragma once
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>

inline std::string TimeString(std::time_t time) {
    const char *time_string = std::asctime(std::gmtime(&time));
    return std::string(time_string, time_string + 24);
}

class ProgressIndicator {
    std::time_t _start_time;
    uint64_t _max_progress;

  public:
    ProgressIndicator(const std::time_t start_time, uint64_t max_progress)
        : _start_time(start_time), _max_progress(max_progress) {}

    std::string string(const std::time_t now, uint64_t progress) {
        std::ostringstream strm;
        strm << std::fixed << std::setprecision(2) << std::setw(6)
             << 100.0 * progress / _max_progress << "%";
        if (progress > 0) {
            const double elapsed = std::difftime(now, _start_time);
            const std::time_t eta =
                _start_time + _max_progress * elapsed / progress;
            strm << " ETA " << TimeString(eta);
        }
        return strm.str();
    }
};
