// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "example_device_operation.hpp"

namespace ttnn::operations::examples {

NocInlineDeviceOperation::program_factory_t NocInlineDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return SingleCore{};
}

void NocInlineDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

void NocInlineDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

NocInlineDeviceOperation::spec_return_value_t NocInlineDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    return TensorSpec(
        input_tensor.get_logical_shape(),
        tt::tt_metal::TensorLayout(
            input_tensor.get_dtype(), tt::tt_metal::PageConfig(input_tensor.get_layout()), MemoryConfig{}));
}

NocInlineDeviceOperation::tensor_return_value_t NocInlineDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input_tensor.device());
}

std::tuple<NocInlineDeviceOperation::operation_attributes_t, NocInlineDeviceOperation::tensor_args_t>
NocInlineDeviceOperation::invoke(const Tensor& input_tensor) {
    return {operation_attributes_t{}, tensor_args_t{input_tensor}};
}

}  // namespace ttnn::operations::examples
