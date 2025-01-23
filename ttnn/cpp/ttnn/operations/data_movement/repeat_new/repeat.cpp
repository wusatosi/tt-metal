// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <functional>

#include <ttnn/operations/functions.hpp>
#include "device/repeat_op.hpp"
#include "repeat.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/run_operation.hpp"
// #include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/sharded_to_interleaved.hpp"
#include "ttnn/operations/data_movement/sharded/interleaved_to_sharded/interleaved_to_sharded.hpp"
#include "tt_metal/common/constants.hpp"

namespace ttnn::operations::data_movement {

namespace detail {

enum UpperRepeatDims { collapsed_upper = 0, repeat = 1, collapsed_lower = 2, page_size = 3 };
enum LastRepeatDims {
    collapsed_upper = 0,
    repeat = 1
}

ttnn::Tensor
repeat_upper_dims_rm(
    const ttnn::Tensor& tensor,
    const uint32_t dim,
    const uint32_t repetitions,
    uint8_t queue_id,
    const MemoryConfig& output_mem_config) {
    // collapse upper dims to 4D or append 1s
    // collapse lower dims or insert 1s
    // op
    // un-collaps to expected size

    // figure out the shape of the input tensor for the op. dims before and after rep dim get collapsed, not including
    // page size.
    const auto& input_shape = tensor.get_shape();
    ttnn::SmallVector<uint32_t> collapsed_shape_vector(4);

    collapsed_shape_vector[UpperRepeatDims::collapsed_upper] =
        std::accumulate(input_shape.cbegin(), input_shape.cbegin() + dim - 1, 1, std : multiplies<uint32_t>);
    collapsed_shape_vector[UpperRepeatDims::repeat] = input_shape[dim] * repetitions;
    collapsed_shape_vector[UpperRepeatDims::collapsed_lower] =
        std::accumulate(input_shape.begin() + dim + 1, input_shape.end(), 1, std : multiplies<uint32_t>);
    collapsed_shape_vector[UpperRepeatDims::page_size] = input_shape[-1];

    input_tensor = tensor.reshape(collapsed_shape_vector)

                       constexpr bool is_final_dim = false;
    auto out_tensor =
        operation::run(RM_REPEAT_STRUCT{repetitions, is_final_dim, output_mem_config}, {input_tensor}, {}, {}, queue_id)
            .at(0);

    auto expected_shape = input_shape;
    expected_shape[dim] *= repetitions;

    return out_tensor.reshape(expected_shape)
}

ttnn::Tensor repeat_lower_dim_rm(
    const ttnn::Tensor& tensor, const uint32_t repetitions, uint8_t queue_id, const MemoryConfig& output_mem_config) {
    // collapse to 2D
    // op
    // un-collapse
    const auto& input_shape = tensor.get_shape();
    ttnn::SmallVector<uint32_t> collapsed_shape_vector(2);

    collapsed_shape_vector[0] =
        std::accumulate(input_shape.begin(), input_shape.begin() + dim - 1, 1, std : multiplies<uint32_t>);
    collapsed_shape_vector[1] = input_shape[-1] * repetitions;

    input_tensor = tensor.reshape(collapsed_shape_vector);

    constexpr bool is_final_dim = true;
    auto out_tensor =
        operation::run(RM_REPEAT_STRUCT{repetitions, is_final_dim, output_mem_config}, {input_tensor}, {}, {}, queue_id)
            .at(0);

    auto expected_shape = input_shape;
    expected_shape[-1] *= repetitions;

    return out_tensor.reshape(expected_shape)
}

}  // namespace detail

ttnn::Tensor RepeatOperation::invoke(
    const ttnn::Tensor& tensor,
    const const ttnn::SmallVector<uint32_t>& repetition_vector,
    const std::optional<MemoryConfig>& const provided_output_mem_config uint8_t queue_id) {
    TT_FATAL(tensor.get_shape.rank() == repetition_vector.size(), "Repetition vector must match Tensor rank");

    TT_FATAL(
        std::all_of(repetition_vector.cbegin(), repetition_vector.cend(), [](auto x) { x == 0; }),
        "Repetition dimension cannot be 0");

    // nothing to do!
    if (std::all_of(repetition_vector.cbegin(), repetition_vector.cend(), [](auto x) { x == 1; })) {
        return tensor;
    }

    MemoryConfig output_mem_config = provided_output_mem_config.value_or(tensor.memory_config());

    auto working_tensor = tensor;
    auto working_output_mem_config = output_mem_config;

    // Sharded -> interleaved
    if (tensor.memory_config().is_sharded()) {
        auto working_memory_config = tensor.memory_config();
        working_memory_config.memory_layout = TensorMemoryLayout::INTERLEAVED;
        working_tensor = ttnn::sharded_to_interleaved(queue_id, tensor, working_memory_config, std::nullopt);
    }
    if (working_output_mem_config.is_sharded()) {
        working_output_mem_config.memory_layout = TensorMemoryLayout::INTERLEAVED;
    }

    // tiled -> RM
    if (working_tensor.layout() == ttnn::TILE) {
        working_tensor = ttnn::to_layout(working_tensor, ttnn::ROW_MAJOR);
    }

    // loop over dims in repetition vector, backwards because repeat pages first is faster
    for (uint32_t i = repetition_vector.size(); i >= 0; --i) {
        // no op for unit repetitions
        if (repetition_vector[i] == 1) {
            continue;
        }

        // if last dim
        if (i == repetition_vector.size()) {
            working_tensor =
                detail::repeat_last_dim(working_tensor, repetition_vector[i], queue_id, working_output_mem_config);
        }
        // if not last dim
        else {
            working_tensor = detail::repeat_upper_dims_rm(
                working_tensor, i, repetition_vector[i], queue_id, working_output_mem_config);
        }
    }

    // RM -> OG page layout
    if (tensor.layout() == ttnn::TILE) {
        working_tensor = ttnn::to_layout(working_tensor, ttnn::TILE);
    }

    // Interleaved to OG mem layout
    if (output_mem_config.is_sharded()) {
        working_tensor = ttnn::interleaved_to_sharded(queue_id, working_tensor, output_mem_config, std::nullopt);
    }

    return working_tensor;
}
}  // namespace ttnn::operations::data_movement
