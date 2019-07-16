#include "next_configuration.h"

#include <array>
#include <gtest/gtest.h>
namespace {
struct ConfigurationItem {
    uint64_t value;
    ConfigurationItem(uint64_t value) : value(value) {}
};

bool operator==(const ConfigurationItem &i1, const ConfigurationItem &i2) {
    return i1.value == i2.value;
}

TEST(NextConfiguration, NextConfiguration) {
    using Config = std::array<ConfigurationItem, 3>;
    // clang-format off
    std::array<Config, 27> expected_configs = {{
        {2, 2, 2}, {3, 2, 2}, {4, 2, 2},
        {2, 3, 2}, {3, 3, 2}, {4, 3, 2},
        {2, 4, 2}, {3, 4, 2}, {4, 4, 2},
        {2, 2, 3}, {3, 2, 3}, {4, 2, 3},
        {2, 3, 3}, {3, 3, 3}, {4, 3, 3},
        {2, 4, 3}, {3, 4, 3}, {4, 4, 3},
        {2, 2, 4}, {3, 2, 4}, {4, 2, 4},
        {2, 3, 4}, {3, 3, 4}, {4, 3, 4},
        {2, 4, 4}, {3, 4, 4}, {4, 4, 4}}};
    // clang-format on

    Config config = expected_configs[0];
    for (size_t i = 1; i < expected_configs.size(); ++i) {
        EXPECT_TRUE(NextConfiguration(config.begin(), config.end(), uint64_t(2),
                                      uint64_t(4)));
        EXPECT_EQ(config, expected_configs[i]);
    }
}
} // namespace
