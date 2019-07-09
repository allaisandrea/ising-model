#pragma once

template <typename Clock> class Throttle {
    typename Clock::duration _duration;
    typename Clock::time_point _last_execution_time;

  public:
    Throttle(const typename Clock::duration &duration)
        : _duration(duration), _last_execution_time{} {};
    template <typename Callable> void operator()(Callable callable) {
        typename Clock::time_point now = Clock::now();
        if (now - _last_execution_time >= _duration) {
            callable();
            _last_execution_time = now;
        }
    }
};
