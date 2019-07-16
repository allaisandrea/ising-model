
#pragma once
#include <cstdint>
struct MockClock {
    using time_point = int64_t;
    using duration = int64_t;
    static int64_t time;
    static int64_t now() { return time; }
};
