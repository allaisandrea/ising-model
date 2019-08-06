#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <functional>

#include "udh.pb.h"

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

inline bool Skip(uint64_t n_skip, std::istream *stream) {
    for (uint64_t i = 0; i < n_skip; ++i) {
        uint64_t size;
        stream->read(reinterpret_cast<char *>(&size), sizeof(size));
        if (stream->eof()) {
            return false;
        }
        if (!stream->good()) {
            throw std::runtime_error("Failed to read size");
        }
        stream->seekg(size, std::ios_base::cur);
        if (stream->eof()) {
            return false;
        }
        if (!stream->good()) {
            throw std::runtime_error("Failed to seek forward");
        }
    }
    return true;
}

inline void PrintAsCsv(const UdhParameters &params, std::ostream *pStrm) {
    std::ostream &strm = *pStrm;
    // clang-format off
    strm << params.shape().size() << ","
         << params.j() << ","
         << params.mu() << ",";
    // clang-format on
    for (const uint32_t i : params.shape()) {
        strm << i << ",";
    }
    for (uint64_t i = params.shape().size(); i < 5; ++i) {
        strm << "1,";
    }

    // clang-format off
    strm << params.n_wolff() << ","
         << params.n_metropolis() << ","
         << params.measure_every() << ","
         << params.seed() << ","
         << "\"" << params.id() << "\","
         << params.n_measure() << ","
         << "\"" << params.tag() << "\"";
    // clang-format on
}

inline void PrintAsCsv(const UdhObservables &observables, std::ostream *pStrm) {
    std::ostream &strm = *pStrm;
    // clang-format off
    strm << observables.sequence_id() << ","
         << observables.stamp() << ","
         << observables.flip_cluster_duration() << ","
         << observables.clear_flag_duration() << ","
         << observables.metropolis_sweep_duration() << ","
         << observables.measure_duration() << ","
         << observables.serialize_duration() << ","
         << observables.n_down() << ","
         << observables.n_holes() << ","
         << observables.n_up() << ","
         << observables.sum_si_sj();
    // clang-format on
}

template <typename ProtoMessage>
inline void PrintCsvHeader(std::ostream *pStrm);

template <> inline void PrintCsvHeader<UdhObservables>(std::ostream *pStrm) {
    std::ostream &strm = *pStrm;
    // clang-format off
    strm << "sequence_id,"
         << "stamp,"
         << "flip_cluster_duration,"
         << "clear_flag_duration,"
         << "metropolis_sweep_duration,"
         << "measure_duration,"
         << "serialize_duration,"
         << "n_down,"
         << "n_holes,"
         << "n_up,"
         << "sum_si_sj";
    // clang-format on
}

template <typename ProtoMessage>
inline std::string GetCsvString(const ProtoMessage &message) {
    std::ostringstream strm;
    PrintAsCsv(message, &strm);
    return strm.str();
}

template <typename ParametersIt>
inline void PrintParametersAsCsv(ParametersIt begin, ParametersIt end,
                                 std::ostream *pStrm) {
    std::ostream &strm = *pStrm;
    // clang-format off
    strm << "file_name"
         << "n_dim,"
         << "J,"
         << "L0,"
         << "L1,"
         << "L2,"
         << "L3,"
         << "L4,"
         << "n_wolff,"
         << "n_metropolis,"
         << "measure_every,"
         << "seed,"
         << "id,"
         << "n_measure,"
         << "tag" << std::endl;
    // clang-format on
    for (ParametersIt params = begin; params != end; ++params) {
        strm << params->first << ",";
        PrintAsCsv(params->second, pStrm);
        strm << "\n";
    }
}

using OpenFunctionT =
    std::function<std::unique_ptr<std::istream>(const std::string &)>;

inline std::unique_ptr<std::istream> OpenFile(const std::string &file_name) {
    std::unique_ptr<std::istream> file(
        new std::ifstream(file_name, std::ios_base::binary));
    if (!file->good()) {
        throw std::runtime_error("Unable to open \"" + file_name + "\".");
    }
    return file;
}
