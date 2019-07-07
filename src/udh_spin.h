#pragma once

#include <cstdint>

struct UdhSpin {
    uint8_t value;
};

UdhSpin UdhSpinDown() { return {0}; };
UdhSpin UdhSpinHole() { return {1}; };
UdhSpin UdhSpinUp() { return {2}; };

bool operator==(UdhSpin s1, UdhSpin s2) { return s1.value == s2.value; };
bool operator!=(UdhSpin s1, UdhSpin s2) { return s1.value != s2.value; };

bool MaskedEqual(UdhSpin s1, UdhSpin s2) {
    return (s1.value & 3) == (s2.value & 3);
}

bool Visited(UdhSpin s) { return s.value & 128; }

void MarkVisited(UdhSpin *s) { s->value |= 128; }

void ClearVisitedFlag(UdhSpin *s) { s->value &= (~128); }

void Flip(UdhSpin *s) { s->value ^= 2; }
