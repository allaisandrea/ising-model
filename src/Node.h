#pragma once

#include <cstdint>

using Node = uint8_t;

inline bool Parallel(Node node1, Node node2) {
    return ((node1 ^ node2) & 1) == 0;
}
