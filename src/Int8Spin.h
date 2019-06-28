#pragma once

#include <cstdint>

struct Int8Spin {
    Int8Spin(int8_t value) : value(value) {}
    int8_t value;
};

bool operator==(Int8Spin s1, Int8Spin s2) { return s1.value == s2.value; };

bool Parallel(Int8Spin s1, Int8Spin s2) { return s1.value * s2.value > 0; }

bool Visited(Int8Spin s) { return ((s.value ^ s.value << 1) & 128); }

void MarkVisited(Int8Spin *s) {
    s->value = (128 & ~s->value << 1) | (s->value & 127);
}

void ClearVisitedFlag(Int8Spin *s) {
    s->value = (128 & s->value << 1) | (s->value & 127);
}

void Flip(Int8Spin *s) { s->value ^= 1; }
