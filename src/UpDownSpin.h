#pragma once

#include <cstdint>

struct UpDownSpin {
    UpDownSpin(uint8_t value) : value(value) {}
    uint8_t value;
};

bool operator==(UpDownSpin s1, UpDownSpin s2) { return s1.value == s2.value; };

bool Parallel(UpDownSpin s1, UpDownSpin s2) {
    return ((s1.value ^ s2.value) & 1) == 0;
}

bool Visited(UpDownSpin s) { return s.value & 128; }

void MarkVisited(UpDownSpin *s) { s->value |= 128; }

void ClearVisitedFlag(UpDownSpin *s) { s->value &= (~128); }

void Flip(UpDownSpin *s) { s->value ^= 1; }
