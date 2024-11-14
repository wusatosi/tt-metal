// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <functional>
#include <optional>
#include <ostream>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "common/constants.hpp"
#include "gtest/gtest.h"
#include "impl/event/event.hpp"
#include "impl/program/program.hpp"
#include "tests/tt_metal/tt_metal/unit_tests_common/common/common_fixture.hpp"
#include "third_party/json/json.hpp"
#include "tt_metal/common/logger.hpp"
#include "ttnn/device.hpp"
#include "ttnn/graph/graph_operation_queries.hpp"
#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"
// #include "ttnn/operations/common/l1_interface_common.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
// #include "ttnn/operations/eltwise/binary/binary.hpp"
// #include "ttnn/operations/eltwise/binary/binary_constraints.hpp"
// #include "ttnn/operations/eltwise/binary/binary_l1_interface.hpp"
#include "ttnn/compiler_interface/compiler_interface.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/unary/unary_compiler_interface.hpp"
// #include "ttnn/operations/matmul/device/matmul_types.hpp"
// #include "ttnn/operations/matmul/matmul.hpp"
// #include "ttnn/operations/matmul/matmul_l1_interface.hpp"
// #include "ttnn/operations/normalization/softmax/softmax.hpp"
// #include "ttnn/operations/normalization/softmax/softmax_l1_interface.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn {
namespace operations {
namespace binary {
namespace test {

struct OperandShapeTestParam {
    ttnn::SimpleShape shape;
    tt::tt_metal::MemoryConfig memory_config;
    tt::tt_metal::DataType data_type = tt::tt_metal::DataType::BFLOAT16;
    tt::tt_metal::Layout layout = tt::tt_metal::Layout::TILE;
    std::vector<uint32_t> expected_cbs_per_core;
    std::vector<std::tuple<uint32_t, uint32_t>> expected_internal_tensors_per_core;

    size_t expected_peak_cbs_per_core;
    size_t expected_peak_l1_tensors_per_core;
    size_t expected_output_l1_tensor_per_core;
};

namespace detail {
static std::ostream &operator<<(std::ostream &os, const tt::tt_metal::TensorMemoryLayout &tensor_memory_layout) {
    switch (tensor_memory_layout) {
        case TensorMemoryLayout::INTERLEAVED: os << "I"; break;
        case TensorMemoryLayout::WIDTH_SHARDED: os << "WS"; break;
        case TensorMemoryLayout::HEIGHT_SHARDED: os << "HS"; break;
        case TensorMemoryLayout::BLOCK_SHARDED: os << "BS"; break;
        default: os << "U"; break;
    }
    return os;
}
static std::ostream &operator<<(std::ostream &os, const tt::tt_metal::BufferType &buffer_type) {
    switch (buffer_type) {
        case BufferType::DRAM: os << "DRAM"; break;
        case BufferType::L1: os << "L1"; break;
        default: os << "U"; break;
    }
    return os;
}
}  // namespace detail

class EltwiseUnaryOpInterfaceTestFixture : public TTNNFixtureWithDevice,
                                           public testing::WithParamInterface<OperandShapeTestParam> {};

TEST_P(EltwiseUnaryOpInterfaceTestFixture, Unary) {
    auto input = GetParam();

    std::cout << "OP = relu(" << input.shape << ")" << std::endl;

    // Run the test
    {
        const auto &l1_input = std::make_tuple(input.shape, input.data_type, input.layout, input.memory_config);
        const auto &l1_output = std::make_tuple(input.shape, input.data_type, input.layout, input.memory_config);

        auto constraint = compiler_interface::unary::unary_op_constraints<ttnn::relu>(l1_input, l1_output);

        bool constraint_valid = std::get<0>(constraint);
        size_t peak_cbs_per_core = std::get<1>(constraint);
        size_t peak_l1_tensors_per_core = std::get<2>(constraint);
        size_t output_l1_tensor_per_core = std::get<3>(constraint);

        EXPECT_EQ(constraint_valid, true);
        EXPECT_EQ(peak_cbs_per_core, input.expected_peak_cbs_per_core);
        EXPECT_EQ(peak_l1_tensors_per_core, input.expected_peak_l1_tensors_per_core);
        EXPECT_EQ(output_l1_tensor_per_core, input.expected_output_l1_tensor_per_core);
    }
}

INSTANTIATE_TEST_SUITE_P(
    CompilerInterface,                   // Prefix for the instantiated test suite
    EltwiseUnaryOpInterfaceTestFixture,  // Test suite name
    ::testing::Values(
        OperandShapeTestParam{
            .shape = ttnn::SimpleShape(tt::tt_metal::Array4D{3, 1, 32 * 32, 32 * 32}),
            .memory_config =
                {.memory_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                 .buffer_type = tt::tt_metal::BufferType::L1,
                 .shard_spec =
                     tt::tt_metal::ShardSpec{
                         CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{3, 3}}}},
                         {6 * 32, 32 * 32},
                         ShardOrientation::COL_MAJOR}},
            .expected_peak_cbs_per_core = 0,
            .expected_peak_l1_tensors_per_core = 2 * (3 * 32 * 32 * 32 * 32) / 16,
            .expected_output_l1_tensor_per_core = 2 * (3 * 32 * 32 * 32 * 32) / 16,
        },
        OperandShapeTestParam{
            .shape = ttnn::SimpleShape(tt::tt_metal::Array4D{4, 2, 5 * 32, 7 * 32}),
            .memory_config = ttnn::L1_MEMORY_CONFIG,
            .expected_peak_cbs_per_core = 2 * 4096,
            .expected_peak_l1_tensors_per_core = 10240,
            .expected_output_l1_tensor_per_core = 10240,
        }),
    [](const testing::TestParamInfo<OperandShapeTestParam> &info) {
        std::stringstream ss;
        {
            using detail::operator<<;
            ss << info.param.memory_config.buffer_type << "_" << info.param.memory_config.memory_layout << "_";
        }
        for (size_t i = 0; i < info.param.shape.rank(); i++) {
            if (i != 0) {
                ss << "x";
            }
            ss << std::to_string(info.param.shape[i]);
        }
        return ss.str();
    });

}  // namespace test
}  // namespace binary
}  // namespace operations
}  // namespace ttnn
