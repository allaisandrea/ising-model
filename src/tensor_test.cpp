#include "tensor.h"

#include <gtest/gtest.h>

namespace {
template <size_t nDim> void TestIndexConversion() {
    std::mt19937 rng;
    for (size_t it = 0; it < 10; ++it) {
        const auto shape = GetRandomShape<nDim>(2, 6, &rng);
        const size_t size = GetSize(shape);
        std::uniform_int_distribution<size_t> rand_int(0, size - 1);
        for (size_t it2 = 0; it2 < 10; ++it2) {
            const size_t i = rand_int(rng);
            const Index<nDim> j = GetVectorIndex(i, shape);
            const size_t k = GetScalarIndex(j, shape);
            EXPECT_EQ(i, k);
        }
    }
}

TEST(Tensor, TestIndexConversion) {
    TestIndexConversion<2>();
    TestIndexConversion<3>();
    TestIndexConversion<4>();
}

TEST(Tensor, IndexIsValid) {
    EXPECT_TRUE(IndexIsValid(Index<2>{0, 0}, Index<2>{2, 3}));
    EXPECT_TRUE(IndexIsValid(Index<2>{1, 0}, Index<2>{2, 3}));
    EXPECT_TRUE(IndexIsValid(Index<2>{0, 2}, Index<2>{2, 3}));
    EXPECT_FALSE(IndexIsValid(Index<2>{2, 0}, Index<2>{2, 3}));
    EXPECT_FALSE(IndexIsValid(Index<2>{0, 3}, Index<2>{2, 3}));

    EXPECT_TRUE(IndexIsValid(Index<3>{0, 0, 0}, Index<3>{2, 3, 4}));
    EXPECT_TRUE(IndexIsValid(Index<3>{1, 0, 0}, Index<3>{2, 3, 4}));
    EXPECT_TRUE(IndexIsValid(Index<3>{0, 2, 0}, Index<3>{2, 3, 4}));
    EXPECT_TRUE(IndexIsValid(Index<3>{0, 0, 3}, Index<3>{2, 3, 4}));
    EXPECT_FALSE(IndexIsValid(Index<3>{2, 0, 0}, Index<3>{2, 3, 4}));
    EXPECT_FALSE(IndexIsValid(Index<3>{0, 3, 0}, Index<3>{2, 3, 4}));
    EXPECT_FALSE(IndexIsValid(Index<3>{0, 0, 4}, Index<3>{2, 3, 4}));
}

TEST(Tensor, NextIndex2D) {
    const Index<2> shape{5, 4};
    // clang-format off
    std::array<Index<2>, 20> expected_indices = {{
        {0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0},
        {0, 1}, {1, 1}, {2, 1}, {3, 1}, {4, 1},
        {0, 2}, {1, 2}, {2, 2}, {3, 2}, {4, 2},
        {0, 3}, {1, 3}, {2, 3}, {3, 3}, {4, 3},
    }};
    // clang-format on
    Index<2> index{0, 0};
    size_t i = 0;
    do {
        EXPECT_EQ(index, expected_indices.at(i));
        ++i;
    } while (NextIndex(&index, shape));
    EXPECT_EQ(i, expected_indices.size());
}

TEST(Tensor, NextIndex3D) {
    const Index<3> shape{4, 3, 2};
    // clang-format off
    std::array<Index<3>, 24> expected_indices = {{
        {0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {3, 0, 0},
        {0, 1, 0}, {1, 1, 0}, {2, 1, 0}, {3, 1, 0},
        {0, 2, 0}, {1, 2, 0}, {2, 2, 0}, {3, 2, 0},
        {0, 0, 1}, {1, 0, 1}, {2, 0, 1}, {3, 0, 1},
        {0, 1, 1}, {1, 1, 1}, {2, 1, 1}, {3, 1, 1},
        {0, 2, 1}, {1, 2, 1}, {2, 2, 1}, {3, 2, 1},
    }};
    // clang-format on
    Index<3> index{0, 0, 0};
    size_t i = 0;
    do {
        EXPECT_EQ(index, expected_indices.at(i));
        ++i;
    } while (NextIndex(&index, shape));
    EXPECT_EQ(i, expected_indices.size());
}

template <size_t nDim>
void TestNextIndexStrided(const Index<nDim> &shape, uint64_t stride,
                          const Index<nDim> &starting_index,
                          const std::vector<Index<nDim>> &expected_indices) {
    Index<nDim> index = starting_index;
    size_t i = 0;
    do {
        EXPECT_EQ(index, expected_indices.at(i));
        ++i;
    } while (NextIndex(&index, shape, /*stride=*/stride));
    EXPECT_EQ(i, expected_indices.size());
}

TEST(Tensor, NextIndexStrided2D) {
    // clang-format off
    TestNextIndexStrided<2>({6, 4}, 2, {0, 0}, {{
        {0, 0}, {2, 0}, {4, 0},
        {0, 2}, {2, 2}, {4, 2}
    }});
    TestNextIndexStrided<2>({6, 4}, 2, {1, 0}, {{
        {1, 0}, {3, 0}, {5, 0},
        {1, 2}, {3, 2}, {5, 2}
    }});
    TestNextIndexStrided<2>({6, 4}, 2, {0, 1}, {{
        {0, 1}, {2, 1}, {4, 1},
        {0, 3}, {2, 3}, {4, 3}
    }});
    TestNextIndexStrided<2>({6, 4}, 2, {1, 1}, {{
        {1, 1}, {3, 1}, {5, 1},
        {1, 3}, {3, 3}, {5, 3}
    }});
    // clang-format on
}

TEST(Tensor, HypercubeShape) {
    EXPECT_EQ(HypercubeShape<2>(5), (Index<2>{5, 5}));
    EXPECT_EQ(HypercubeShape<3>(7), (Index<3>{7, 7, 7}));
}

template <size_t nDim> void TestGetFirstNeighbors() {
    std::mt19937 rng;
    for (size_t it1 = 0; it1 < 10; ++it1) {
        const auto shape = GetRandomShape<nDim>(2, 6, &rng);
        const size_t size = GetSize(shape);
        for (size_t si = 0; si < size; ++si) {
            const Index<nDim> i = GetVectorIndex(si, shape);
            const auto n1 = GetFirstNeighbors(i, shape);
            for (const auto &j : n1) {
                const auto n2 = GetFirstNeighbors(j, shape);
                EXPECT_NE(std::find(n2.begin(), n2.end(), i), n2.end());
            }
        }
    }
}

TEST(Tensor, TestGetFirstNeighbors) {
    TestGetFirstNeighbors<2>();
    TestGetFirstNeighbors<3>();
    TestGetFirstNeighbors<4>();
}

template <size_t nDim>
void TestTileTensor(const Index<nDim> &tile_shape, const Index<nDim> &n_tiles,
                    const std::vector<uint64_t> &expected_values) {
    Tensor<nDim, uint64_t> tile(tile_shape, 0);
    for (uint64_t i = 0; i < tile.size(); ++i) {
        tile[i] = i;
    }
    const Tensor<nDim, uint64_t> tiled = TileTensor(tile, n_tiles);
    for (size_t i = 0; i < expected_values.size(); ++i) {
        EXPECT_EQ(tiled[i], expected_values[i]);
    }
    EXPECT_EQ(expected_values.size(), tiled.size());
}

TEST(Tensor, TileTensor1) {
    // clang-format off
    TestTileTensor<2>({2, 3}, {3, 2}, {
             0, 1, 0, 1, 0, 1,
             2, 3, 2, 3, 2, 3,
             4, 5, 4, 5, 4, 5,
             0, 1, 0, 1, 0, 1,
             2, 3, 2, 3, 2, 3,
             4, 5, 4, 5, 4, 5});
    // clang-format on
}

TEST(Tensor, TileTensor2) {
    // clang-format off
    TestTileTensor<2>({2, 3}, {2, 3}, {
             0, 1, 0, 1 ,
             2, 3, 2, 3 ,
             4, 5, 4, 5 ,
             0, 1, 0, 1 ,
             2, 3, 2, 3 ,
             4, 5, 4, 5 ,
             0, 1, 0, 1 ,
             2, 3, 2, 3 ,
             4, 5, 4, 5 });
    // clang-format on
}

TEST(Tensor, TileTensor3) {
    // clang-format off
    TestTileTensor<3>({2, 3, 2}, {3, 2, 2}, {
             0, 1, 0, 1, 0, 1,
             2, 3, 2, 3, 2, 3,
             4, 5, 4, 5, 4, 5,
             0, 1, 0, 1, 0, 1,
             2, 3, 2, 3, 2, 3,
             4, 5, 4, 5, 4, 5,

             6, 7, 6, 7, 6, 7,
             8, 9, 8, 9, 8, 9,
            10,11,10,11,10,11,
             6, 7, 6, 7, 6, 7,
             8, 9, 8, 9, 8, 9,
            10,11,10,11,10,11,

             0, 1, 0, 1, 0, 1,
             2, 3, 2, 3, 2, 3,
             4, 5, 4, 5, 4, 5,
             0, 1, 0, 1, 0, 1,
             2, 3, 2, 3, 2, 3,
             4, 5, 4, 5, 4, 5,

             6, 7, 6, 7, 6, 7,
             8, 9, 8, 9, 8, 9,
            10,11,10,11,10,11,
             6, 7, 6, 7, 6, 7,
             8, 9, 8, 9, 8, 9,
            10,11,10,11,10,11,
             });
    // clang-format on
}
} // namespace
