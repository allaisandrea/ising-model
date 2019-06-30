#pragma once

#include <cstdint>

struct UdhSpin {
    static UdhSpin Down() { return 0; };
    static UdhSpin Hole() { return 1; };
    static UdhSpin Up() { return 2; };
    UdhSpin(uint8_t value) : value(value) {}
    uint8_t value;
};

bool operator==(UdhSpin s1, UdhSpin s2) { return s1.value == s2.value; };
bool operator!=(UdhSpin s1, UdhSpin s2) { return s1.value != s2.value; };

bool MaskedEqual(UdhSpin s1, UdhSpin s2) {
    return (s1.value & 3) == (s2.value & 3);
}

bool Visited(UdhSpin s) { return s.value & 128; }

void MarkVisited(UdhSpin *s) { s->value |= 128; }

void ClearVisitedFlag(UdhSpin *s) { s->value &= (~128); }

void Flip(UdhSpin *s) { s->value ^= 2; }
