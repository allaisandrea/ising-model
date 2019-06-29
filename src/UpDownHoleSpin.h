#pragma once

#include <cstdint>

struct UpDownHoleSpin {
    static UpDownHoleSpin Down() { return 0; };
    static UpDownHoleSpin Hole() { return 1; };
    static UpDownHoleSpin Up() { return 2; };
    UpDownHoleSpin(uint8_t value) : value(value) {}
    uint8_t value;
};

bool operator==(UpDownHoleSpin s1, UpDownHoleSpin s2) {
    return s1.value == s2.value;
};
bool operator!=(UpDownHoleSpin s1, UpDownHoleSpin s2) {
    return s1.value != s2.value;
};

bool MaskedEqual(UpDownHoleSpin s1, UpDownHoleSpin s2) {
    return (s1.value & 3) == (s2.value & 3);
}

bool Visited(UpDownHoleSpin s) { return s.value & 128; }

void MarkVisited(UpDownHoleSpin *s) { s->value |= 128; }

void ClearVisitedFlag(UpDownHoleSpin *s) { s->value &= (~128); }

void Flip(UpDownHoleSpin *s) { s->value ^= 2; }
