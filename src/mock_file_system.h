#pragma once
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

struct MockFileSystem {
    std::unique_ptr<std::istream> operator()(const std::string &file_name) {
        return std::unique_ptr<std::istream>(
            new std::istringstream(files.at(file_name)));
    }
    std::unordered_map<std::string, std::string> files;
};
