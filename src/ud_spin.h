#pragma once

#include <cstdint>

struct UdSpin {
    uint8_t value;
};

UdSpin UdSpinDown() { return {0}; }
UdSpin UdSpinUp() { return {1}; }

bool operator==(UdSpin s1, UdSpin s2) { return s1.value == s2.value; }

bool MaskedEqual(UdSpin s1, UdSpin s2) {
    return (s1.value & 1) == (s2.value & 1);
}

bool Visited(UdSpin s) { return s.value & 128; }

void MarkVisited(UdSpin *s) { s->value |= 128; }

void ClearVisitedFlag(UdSpin *s) { s->value &= (~128); }

void Flip(UdSpin *s) { s->value ^= 1; }
