// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_bug_report_device_operation.hpp"

#include <cstdint>

#include <tt-metalium/base_types.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::moreh::moreh_bug_report {
MorehBugReportOperation::program_factory_t MorehBugReportOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return ProgramFactory{};
}

void validate_tensors(
    const MorehBugReportOperation::operation_attributes_t& operation_attributes,
    const MorehBugReportOperation::tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
}

void MorehBugReportOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_tensors(operation_attributes, tensor_args);
};

void MorehBugReportOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_tensors(operation_attributes, tensor_args);
};

MorehBugReportOperation::spec_return_value_t MorehBugReportOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return TensorSpec(
        tensor_args.input.get_logical_shape(),
        TensorLayout(
            tensor_args.input.get_dtype(),
            PageConfig(tensor_args.input.get_layout()),
            tensor_args.input.memory_config()));
};

MorehBugReportOperation::tensor_return_value_t MorehBugReportOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

std::tuple<MorehBugReportOperation::operation_attributes_t, MorehBugReportOperation::tensor_args_t>
MorehBugReportOperation::invoke(const Tensor& input) {
    return {{}, {input}};
}
}  // namespace ttnn::operations::moreh::moreh_bug_report
