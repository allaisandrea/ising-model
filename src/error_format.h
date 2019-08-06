#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <algorithm>

inline std::string ErrorFormat(double x, double s) {
    const int64_t log_s =
        std::floor(std::log10(std::max<double>({s, 1.0e-8 * std::abs(x), 1.0e-100})));
    const int64_t log_x =
        std::floor(std::log10(std::max(std::abs(x), 1.0e-100)));
    std::ostringstream strm;
    if (log_s <= log_x) {
        strm << std::fixed << std::setprecision(log_x - log_s)
             << x * std::pow(10, -log_x) << "(" << std::setprecision(0)
             << s * std::pow(10, -log_s) << ")E" << log_x;
    } else {
        strm << "0(" << std::setprecision(0) << s * std::pow(10, -log_s) << ")E"
             << log_s;
    }
    return strm.str();
}
