// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>

#include "ttnn/graph/graph_trace_utils.hpp"
#include "ttnn/operations/trace.hpp"

namespace ttnn::graph {

struct RuntimeQueryResponse {
    ExecutionStatus status = ExecutionStatus::Error;
    uint64_t runtime = 0;
    std::optional<std::string> error_message;
};

static constexpr int NUM_TRACE_EXECUTIONS = 10;

/**
 * @brief Extracts a trace of the operation(s) and returns the trace ID.
 *
 * This function guarantees that the capture will be stopped and released if running the op(s)
 * throws an exception.
 *
 * @tparam Op The type of the operation or a callable op chain that will be invoked to capture the trace operations.
 * @tparam Args The types of the arguments that will be passed to the operation or op chain.
 * @param op The operation or op chain that will be traced.
 * @param device A pointer to the Device object. Must be opened with trace region size set to a sufficiently high
 * amount.
 * @param args The arguments that will be passed to the operation or callable op chain.
 * @return ID for captured trace.
 */
template <typename Op, typename... Args>
auto capture_op_trace(Op op, IDevice* device, Args&&... args) {
    device->enable_program_cache();
    {  // warm up the program cache - required for trace capture
        std::apply(op, transformed_args);
    }

    auto trace_id = ttnn::operations::trace::begin_trace_capture(device, ttnn::DefaultQueueId);
    try {
        std::apply(op, transformed_args);
    } catch (const std::exception& e) {
        // Ensure trace capture is stopped and released before returning to avoid a memory leak
        ttnn::operations::trace::end_trace_capture(device, trace_id, ttnn::DefaultQueueId);
        ttnn::operations::trace::release_trace(device, trace_id);
        throw e;
    }
    ttnn::operations::trace::end_trace_capture(device, trace_id, ttnn::DefaultQueueId);

    return trace_id;
}

/**
 * @brief Executes a trace, releases the trace, and returns the runtime in nanoseconds.
 *
 * This function guarantees release_trace will be called even if executing the trace throws an exception.
 *
 * @tparam TraceID The type of the trace id returned by trace capture APIs.
 * @param trace_id ID of the captured trace.
 * @param device A pointer to the Device object
 * @return Trace runtime in nanoseconds.
 */
template <typename TraceID>
uint64_t execute_time_and_release_trace(TraceID trace_id, IDevice* device) {
    try {
        device->synchronize();
        uint64_t duration = 0;
        for (int i = 0; i < NUM_TRACE_EXECUTIONS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            ttnn::operations::trace::execute_trace(device, trace_id, ttnn::DefaultQueueId, /* blocking = */ true);
            auto end = std::chrono::high_resolution_clock::now();
            duration += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        }

        ttnn::operations::trace::release_trace(device, trace_id);
        return duration / NUM_TRACE_EXECUTIONS;

    } catch (const std::exception& e) {
        // Ensure captured trace is released before returning to avoid a memory leak
        ttnn::operations::trace::release_trace(device, trace_id);
        throw e;
    }
}

// These arguments *must* match both the args provided on the mlir side and the order of arguments in the invoke()
// method for the op
RuntimeQueryResponse predict_reshard_runtime(
    const ttnn::Tensor& input_tensor, const ttnn::MemoryConfig& memory_config) {
    if (!input_tensor.is_sharded()) {
        return return RuntimeQueryResponse{
            ExecutionStatus::Error, 0, "Reshard Runtime Model: Input tensor is not sharded"};
    }
    if (!memory_config.is_sharded()) {
        return RuntimeQueryResponse{ExecutionStatus::Error, 0, "Reshard Runtime Model: No output shard spec provided"};
    }

    uint64_t num_tiles = input_tensor.volume() / (ttnn::TILE_SIZE * ttnn::TILE_SIZE);

    auto model = load_mlpack_model(input_tensor.memory_config.memory_layout, memory_config.memory_layout);

    auto input_shard_spec = input_tensor.shard_spec.value();
    auto output_shard_spec = memory_config.shard_spec.value();

    // get input and output grid (x,y) from shard spec

    uint64_t runtime = model(input_grid_x, input_grid_y, output_grid_x, output_grid_y, num_tiles);
    return RuntimeQueryResponse{ExecutionStatus::Success, runtime, ""};
}

RuntimeQueryResponse predict_matmul_runtime(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const bool transpose_a = false,
    const bool transpose_b = false,
    const std::optional<const MemoryConfig>& memory_config = std::nullopt,
    const std::optional<const DataType> dtype = std::nullopt) {
    return RuntimeQueryResponse{ExecutionStatus::Error, 0, "Matmul runtime model not yet implemented"};
}

// For ops that have offline runtime models, dispatch to the specific model. Otherwise, return an error
// so the caller uses trace capture
template <typename Op, typename... Args>
auto get_runtime_from_model(Op op, Args&&... args) {
    if constexpr (std::is_same_v<Op, ttnn::reshard>) {
        return predict_reshard_runtime(std::forward<Args>(args));
    } else if constexpr (std::is_same_v<Op, ttnn::matmul>) {
        return predict_matmul_runtime(std::forward<Args>(args));
    } else {
        return RuntimeQueryResponse{ExecutionStatus::Error, 0, "Runtime model not yet implemented"};
    }
}

/**
 * @brief Extracts a trace of the graph operations and returns the trace execution runtime.
 *
 * This function runs trace capture by invoking the provided operation with the given arguments,
 * then excutes the trace and returns the runtime of the trace in nanoseconds.
 *
 * @tparam Op The type of the operation or a callable op chain that will be invoked to capture the trace operations.
 * @tparam Args The types of the arguments that will be passed to the operation or op chain.
 * @param op The operation or op chain that will be traced and have its runtime measured.
 * @param device A pointer to the Device object. Must be opened with trace region size set to a sufficiently high
 * amount.
 * @param args The arguments that will be passed to the operation or callable op chain.
 * @return RuntimeQueryResponse containing the execution status and the runtime, in nanoseconds.
 *         - On success: ExecutionStatus::Success and runtime in nanoseconds.
 *         - On failure: ExecutionStatus::Error, zeroed runtime, and an error message.
 */
template <typename Op, typename... Args>
auto query_op_runtime(Op op, IDevice* device, Args&&... args) {
    // helper lambda to transform TensorSpec to DeviceTensor
    auto transform_arg = [device](auto&& arg) {
        if constexpr (std::is_same_v<std::decay_t<decltype(arg)>, TensorSpec>) {
            return create_device_tensor(arg, device);
        } else {
            return std::forward<decltype(arg)>(arg);
        }
    };
    auto transformed_args = transform_arg(std::forward<Args>(args))...;

    auto response = get_runtime_from_model(Op, transformed_args...);
    if (response.status == ExecutionStatus::Success) {
        return response;
    }

    // If an offline model is not available, or there was an error, fall back on trace capture
    try {
        auto trace_id = capture_op_trace(op, device, transformed_args...);
        auto runtime = execute_time_and_release_trace(trace_id, device);
        return RuntimeQueryResponse{ExecutionStatus::Success, runtime};

    } catch (const std::exception& e) {
        tt::log_debug(tt::LogOp, "op_runtime - error: {}", e.what());
        return RuntimeQueryResponse{ExecutionStatus::Error, 0, e.what()};
    }
}

}  // namespace ttnn::graph
