// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <ttnn/decorators.hpp>
#include "muladd_op.hpp"
#include "ttnn/operations/muladd/muladd.hpp"
#include "muladd_program_factory.hpp"
#include "ttnn/tensor/enum_types.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/types.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::muladd {

void MulAdd::validate(const std::vector<Tensor>& input_tensors) const {
    using namespace tt::constants;
    TT_FATAL(input_tensors.size() == 4, "Number of input tensors should be 4 not {}", input_tensors.size());
}

std::vector<ttnn::TensorSpec> MulAdd::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    // return {TensorSpec(output_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE),
    // DRAM_MEMORY_CONFIG))};
    return {TensorSpec(
        input_tensors.at(0).get_logical_shape(), TensorLayout(dtype, PageConfig(Layout::TILE), memory_config))};
}

operation::ProgramWithCallbacks MulAdd::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    const auto& input_tensor_c = input_tensors.at(2);
    const auto& input_tensor_d = input_tensors.at(3);

    auto& output_tensor = output_tensors.at(0);

    // return single_core_muladd(input_tensor_a, input_tensor_b, input_tensor_c, input_tensor_d, output_tensor,
    // math_fidelity);
    return multi_core_muladd(
        input_tensor_a, input_tensor_b, input_tensor_c, input_tensor_d, output_tensor, math_fidelity);
}

}  // namespace ttnn::operations::muladd
