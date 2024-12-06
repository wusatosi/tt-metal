// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "batch_norm_device_operation.hpp"

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::normalization {
void BatchNormOperation::validate_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& batch_mean = tensor_args.batch_mean;

    auto& output = tensor_args.output;

    check_tensor(input, "batch_norm", "input");

    check_tensor(output, "batch_norm", "output");

    // input (N, C, H, W)
    auto C = input.get_shape().value[1];
    // output (N, C, H, W)
    if (output.has_value()) {
        auto check_C = output.value().get_shape().value[1];
        TT_FATAL(C == check_C, "output_shape[1] must be the same as input's channel size.");
    }

    // mean (1, C, 1, 1)
    TT_FATAL(
        batch_mean.get_shape().value.without_padding()[1] == C,
        "batch_mean_shape[1] must be the same as input's channel size.");
}

BatchNormOperation::program_factory_t BatchNormOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return BatchNormFactory();
}

void BatchNormOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // We don't support sharding for now
    const auto& input = tensor_args.input;
    const auto& batch_mean = tensor_args.batch_mean;
    const auto& output = tensor_args.output;

    TT_FATAL(input.get_layout() == Layout::TILE, "Input tensor must be must be tilized");
    TT_FATAL(
        input.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Input tensor must be interleaved");
    TT_FATAL(
        operation_attributes.memory_config.memory_layout == TensorMemoryLayout::INTERLEAVED,
        "Output tensor to eltwise binary must be interleaved");

    TT_FATAL(batch_mean.get_layout() == Layout::TILE, "batch_mean tensor must be tilized");
    TT_FATAL(
        batch_mean.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
        "batch_mean tensor must be interleaved");

    validate_tensors(operation_attributes, tensor_args);
};

void BatchNormOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_tensors(operation_attributes, tensor_args);
};

DataType BatchNormOperation::operation_attributes_t::get_dtype() const {
    return this->dtype.value_or(this->input_dtype);
}

BatchNormOperation::spec_return_value_t BatchNormOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::constants;
    // mean, rstd (1, C, 1, 1)
    const auto output_shape = tensor_args.input.get_logical_shape();
    return TensorSpec(
        output_shape,
        TensorLayout(operation_attributes.get_dtype(), PageConfig(Layout::TILE), operation_attributes.memory_config));
}

BatchNormOperation::tensor_return_value_t BatchNormOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& output_tensor = tensor_args.output;
    if (output_tensor.has_value()) {
        return output_tensor.value();
    }

    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());

    // const auto output_shapes = compute_output_specs(operation_attributes, tensor_args);
    // auto dtype = tensor_args.input.get_dtype();
    // Layout layout{Layout::TILE};
    // auto device = tensor_args.input.device();
    // std::vector<std::optional<Tensor>> result;
    // result.reserve(1);
    // // output
    // if (tensor_args.output.has_value()) {
    //     result.push_back(tensor_args.output.value());
    // } else {
    //     result.push_back(
    //         // create_device_tensor(output_shapes, dtype, layout, device, operation_attributes.memory_config)
    //         create_device_tensor(output_shapes, tensor_args.input.device())
    //     );
    // }
    // return std::move(result);
}

std::tuple<BatchNormOperation::operation_attributes_t, BatchNormOperation::tensor_args_t> BatchNormOperation::invoke(
    const Tensor& input,
    const Tensor& batch_mean,
    std::optional<Tensor> output,
    const std::optional<MemoryConfig>& memory_config) {
    operation_attributes_t operation_attributes{memory_config.value_or(input.memory_config())};
    tensor_args_t tensor_args{input, batch_mean, std::move(output)};
    return {operation_attributes, tensor_args};
}
}  // namespace ttnn::operations::normalization
