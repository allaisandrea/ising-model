#pragma once
#include <chrono>

template <typename Clock> class Timer {
    typename Clock::time_point _start_time;
    typename Clock::duration _elapsed_time;
    enum State { STARTED, STOPPED } _state;

  public:
    Timer() : _elapsed_time{}, _state(STOPPED) {}
    void start() {
        if (_state == STOPPED) {
            _start_time = Clock::now();
            _state = STARTED;
        }
    }

    void stop() {
        if (_state == STARTED) {
            _elapsed_time += Clock::now() - _start_time;
            _state = STOPPED;
        }
    }

    typename Clock::duration elapsed() {
        if (_state == STOPPED) {
            return _elapsed_time;
        } else if (_state == STARTED) {
            return _elapsed_time + Clock::now() - _start_time;
        } else {
            throw std::runtime_error("Corrupt state");
        }
    }
};
