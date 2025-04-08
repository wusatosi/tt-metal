// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "my_new_op_operation.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

MyNewOpDeviceOperation::program_factory_t MyNewOpDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return MyDeviceProgramFactory{};
}

void MyNewOpDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Validate the operation attributes and tensor arguments here
}

void MyNewOpDeviceOperation::validate(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Validate the operation attributes and tensor arguments here
}

void MyNewOpDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Validate the operation attributes and tensor arguments here
}

MyNewOpDeviceOperation::spec_return_value_t MyNewOpDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    return TensorSpec(input_tensor_a.get_tensor_spec());
}

MyNewOpDeviceOperation::tensor_return_value_t MyNewOpDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input_tensor_a.device());
}

std::tuple<MyNewOpDeviceOperation::operation_attributes_t, MyNewOpDeviceOperation::tensor_args_t>
MyNewOpDeviceOperation::invoke(const Tensor& input_tensor_a, const Tensor& input_tensor_b, const int input_scalar) {
    return {
        operation_attributes_t{input_scalar},
        tensor_args_t{.input_tensor_a = input_tensor_a, .input_tensor_b = input_tensor_b}};
}

}  // namespace ttnn::operations::data_movement
