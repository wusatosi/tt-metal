// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <exception>
#include <functional>
#include <optional>

#include "third_party/json/json.hpp"
#include "tt_metal/common/logger.hpp"
#include "ttnn/compiler_interface/compiler_interface.hpp"
#include "ttnn/compiler_interface/singleton_device_context.hpp"
#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"
#include "ttnn/operations/matmul/matmul.hpp"

namespace ttnn::compiler_interface {

std::tuple<bool, size_t, size_t, size_t> matumul_op_constraints(
    const OperandParams& input_a,
    const OperandParams& input_b,
    const OperandParams& output,
    bool transpose_a = false,
    bool transpose_b = false,
    const std::optional<ttnn::operations::matmul::MatmulProgramConfig>& program_config = std::nullopt) {
    // get_op_trace is a lambda that prepares input and output tensors, capturing graph trace of the op without
    // inputs allocation.
    auto get_op_trace = [](const OperandParams& input_a,
                           const OperandParams& input_b,
                           const OperandParams& output,
                           bool transpose_a,
                           bool transpose_b,
                           const std::optional<ttnn::operations::matmul::MatmulProgramConfig>& program_config) {
        nlohmann::json op_trace;

        // outer graph capture is used to avoid capturing and dispatching of dummy input tensor(s) creation
        {
            ttnn::graph::GraphProcessor::begin_graph_capture(ttnn::graph::GraphProcessor::RunMode::NO_DISPATCH);
            const auto input_tensor_a = create_device_tensor(
                std::get<ttnn::SimpleShape>(input_a),
                std::get<tt::tt_metal::DataType>(input_a),
                std::get<tt::tt_metal::Layout>(input_a),
                SingletonDeviceContext::get_instance().get_device(),
                std::get<tt::tt_metal::MemoryConfig>(input_a));

            const auto input_tensor_b = create_device_tensor(
                std::get<ttnn::SimpleShape>(input_b),
                std::get<tt::tt_metal::DataType>(input_b),
                std::get<tt::tt_metal::Layout>(input_b),
                SingletonDeviceContext::get_instance().get_device(),
                std::get<tt::tt_metal::MemoryConfig>(input_b));

            // output tensor is created in the inner graph capture to capture its allocation
            {
                ttnn::graph::GraphProcessor::begin_graph_capture(ttnn::graph::GraphProcessor::RunMode::NO_DISPATCH);
                auto output_tensor = ttnn::matmul(
                    input_tensor_a,
                    input_tensor_b,
                    transpose_a,
                    transpose_b,
                    std::get<tt::tt_metal::MemoryConfig>(output),
                    std::get<tt::tt_metal::DataType>(output),
                    program_config);
                // close inner graph capture, before output buffer is deallocated
                op_trace = ttnn::graph::GraphProcessor::end_graph_capture();
            }
            // close outer graph capture
            ttnn::graph::GraphProcessor::end_graph_capture();
        }

        // TODO(mbezulj) remove this debug print
        std::cout << graph::to_graphviz(op_trace) << std::endl;

        return op_trace;
    };

    try {
        auto op_trace = get_op_trace(input_a, input_b, output, transpose_a, transpose_b, program_config);
        return extract_data_from_trace(op_trace);
    } catch (const std::exception& e) {
        tt::log_debug(tt::LogOp, "compiler_interface - error: {}", e.what());
    }

    return std::make_tuple(false, 0, 0, 0);
}

}  // namespace ttnn::compiler_interface
