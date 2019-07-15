#pragma once

template <typename Iterator, typename ValueType>
bool NextConfiguration(Iterator begin, Iterator end, ValueType min_value,
                       ValueType max_value) {
    Iterator it = begin;
    while (it != end) {
        ++(it->value);
        if (it->value > max_value) {
            it->value = min_value;
            ++it;
        } else {
            return true;
        }
    }
    return false;
}
