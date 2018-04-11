#pragma once
#include <iostream>
#include <stdexcept>

template <typename T> class Queue {
  public:
    Queue(std::size_t capacity)
        : _data(new T[capacity]), _begin(_data), _end(_begin),
          _dataEnd(_data + capacity) {}

    ~Queue() { delete[] _data; }

    template <class... Args> void emplace(Args &&... args) {
        new (_end) T(std::forward<Args>(args)...);
        ++_end;
        if (_end == _dataEnd)
            _end = _data;
        if (_end == _begin) {
            const std::size_t newCapacity = 3 * (_dataEnd - _data) / 2;
            T *const newData = new T[newCapacity];
            _end = std::copy(_data, _end, std::copy(_begin, _dataEnd, newData));
            delete[] _data;
            _data = newData;
            _begin = _data;
            _dataEnd = _data + newCapacity;
        }
    }

    void pop() {
        if (_begin == _end) {
            throw std::logic_error("Called pop() on empty queue");
        }
        ++_begin;
        if (_begin == _dataEnd)
            _begin = _data;
    }

    T &front() {
        if (_begin == _end) {
            throw std::logic_error("Called front() on empty queue");
        }
        return *_begin;
    }

    bool empty() { return _begin == _end; }

  private:
    T *_data;
    T *_begin;
    T *_end;
    T *_dataEnd;
};
