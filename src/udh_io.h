#pragma once
#include "udh_simulation.pb.h"
#include <stdexcept>

template <typename ProtoMessage>
void Write(const ProtoMessage &message, std::ostream *stream) {
    thread_local std::string buffer;
    buffer.clear();
    if (!message.SerializeToString(&buffer)) {
        throw std::runtime_error("Failed to serialize");
    }
    const uint64_t size = buffer.size();
    stream->write(reinterpret_cast<const char *>(&size), sizeof(size));
    if (!stream->good()) {
        throw std::runtime_error("Failed to write size");
    }
    stream->write(buffer.data(), buffer.size());
    if (!stream->good()) {
        throw std::runtime_error("Failed to write message");
    }
}

template <typename ProtoMessage>
bool Read(ProtoMessage *message, std::istream *stream) {
    thread_local std::string buffer;
    buffer.clear();
    uint64_t size;
    stream->read(reinterpret_cast<char *>(&size), sizeof(size));
    if (stream->eof()) {
        return false;
    }
    if (!stream->good()) {
        throw std::runtime_error("Failed to read size");
    }
    if (size > (1 << 20)) {
        throw std::runtime_error("Excessive size: " + std::to_string(size) +
                                 ", corrupt file");
    }
    buffer.resize(size);
    stream->read(&buffer[0], buffer.size());
    if (stream->eof()) {
        return false;
    }
    if (!stream->good()) {
        throw std::runtime_error("Failed to read message");
    }
    if (!message->ParseFromString(buffer)) {
        throw std::runtime_error("Failed to parse message");
    }
    return true;
}
