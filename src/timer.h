#pragma once
#include <chrono>
#include <stdexcept>

template <typename ClockT> class Timer {
    typename ClockT::time_point _start_time;
    typename ClockT::duration _elapsed_time;
    enum State { STARTED, STOPPED } _state;

  public:
    using Clock = ClockT;
    Timer() : _elapsed_time{}, _state(STOPPED) {}
    void start() {
        if (_state == STOPPED) {
            _start_time = ClockT::now();
            _state = STARTED;
        }
    }

    void stop() {
        if (_state == STARTED) {
            _elapsed_time += ClockT::now() - _start_time;
            _state = STOPPED;
        }
    }

    typename ClockT::duration elapsed() {
        if (_state == STOPPED) {
            return _elapsed_time;
        } else if (_state == STARTED) {
            return _elapsed_time + ClockT::now() - _start_time;
        } else {
            throw std::runtime_error("Corrupt state");
        }
    }
};

using HiResTimer = Timer<std::chrono::high_resolution_clock>;
