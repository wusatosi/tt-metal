// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deinterleave_device_operation.hpp"

namespace ttnn::operations::experimental::deinterleave {
void DeinterleaveOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;

    TT_FATAL(input.get_dtype() == DataType::BFLOAT16, "Deinterleave: input must be BFLOAT16");
    TT_FATAL(input.get_layout() == Layout::ROW_MAJOR, "Deinterleave: input must be ROW_MAJOR");
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Deinterleave: input must be on device");
    TT_FATAL(input.buffer() != nullptr, "Deinterleave: input must be allocated in buffer on device");
    TT_FATAL(
        input.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED,
        "Deinterleave: input must be HEIGHT_SHARDED");
    TT_FATAL(input.memory_config().shard_spec.has_value(), "Deinterleave: input must have shard_spec");
    TT_FATAL(
        input.memory_config().shard_spec.value().orientation == ShardOrientation::ROW_MAJOR,
        "Deinterleave: input must have ROW_MAJOR orientation");

    auto per_core_height = input.memory_config().shard_spec.value().shape[0] / operation_attributes.input_width;
    TT_FATAL(
        per_core_height >= 2 * operation_attributes.stride_hw[0],
        "Deinterleave: per_core_height {} must be larger than {}",
        per_core_height,
        2 * operation_attributes.stride_hw[0]);
    TT_FATAL(
        per_core_height % (2 * operation_attributes.stride_hw[0]) == 0,
        "Deinterleave: per_core_height {} must be div by {}",
        per_core_height,
        2 * operation_attributes.stride_hw[0]);
    TT_FATAL(
        per_core_height * operation_attributes.input_width == input.memory_config().shard_spec.value().shape[0],
        "Deinterleave: per_core_height {} * input_width {} must be equal to input shard_spec shape {}",
        per_core_height,
        operation_attributes.input_width,
        input.memory_config().shard_spec.value().shape[0]);
}

DeinterleaveOperation::program_factory_t DeinterleaveOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return ProgramFactory{};
}

void DeinterleaveOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

void DeinterleaveOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

DeinterleaveOperation::spec_return_value_t DeinterleaveOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    return TensorSpec(
        input.get_logical_shape(),
        tt::tt_metal::TensorLayout(
            input.get_dtype(), tt::tt_metal::PageConfig(input.get_layout()), input.memory_config()));
};

DeinterleaveOperation::tensor_return_value_t DeinterleaveOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(spec, tensor_args.input.device());
}

std::tuple<DeinterleaveOperation::operation_attributes_t, DeinterleaveOperation::tensor_args_t>
DeinterleaveOperation::invoke(
    const Tensor& input,
    const uint32_t input_height,
    const uint32_t input_width,
    const std::array<uint32_t, 2> stride_hw,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return {
        operation_attributes_t{
            input_height,
            input_width,
            stride_hw,
            init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4),
        },
        tensor_args_t{input},
    };
}
}  // namespace ttnn::operations::experimental::deinterleave
