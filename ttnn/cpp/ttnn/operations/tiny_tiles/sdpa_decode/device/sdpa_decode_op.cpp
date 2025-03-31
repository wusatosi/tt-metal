// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_decode_op.hpp"

#include "sdpa_decode_program_factory.hpp"
#include "ttnn/run_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::tiny_tiles {

void ScaledDotProductAttentionDecode::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_FATAL(input_tensors.size() == 3, "Must have 3 input tensors and mask");

    for (auto& input_tensor : input_tensors) {
        TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to SDPA need to be on device!");
        TT_FATAL(input_tensor.buffer() != nullptr, "Operands to SDPA need to be allocated in buffers on device!");
        TT_FATAL((input_tensor.get_layout() == Layout::TILE), "Inputs to SDPA must be tilized");
        TT_FATAL(
            input_tensor.get_dtype() == DataType::BFLOAT16 || input_tensor.get_dtype() == DataType::BFLOAT8_B ||
                input_tensor.get_dtype() == DataType::BFLOAT4_B,
            "Unsupported data type {}.",
            input_tensor.get_dtype());
    }

    // Input 0 must be sharded by height or DRAM interleaved. All other inputs must be in DRAM.
    const auto Q_memcfg = input_tensors.at(0).memory_config();
    TT_FATAL(Q_memcfg.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED, "Error");
}

std::vector<TensorSpec> ScaledDotProductAttentionDecode::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    auto& input = input_tensors.at(0);
    return {TensorSpec(
        input.get_logical_shape(), TensorLayout(input.get_dtype(), PageConfig(Layout::TILE), input.memory_config()))};
}

operation::ProgramWithCallbacks ScaledDotProductAttentionDecode::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    auto& input_tensor_q = input_tensors.at(0);
    auto& input_tensor_k = input_tensors.at(1);
    auto& input_tensor_v = input_tensors.at(2);

    return detail::sdpa_decode_multi_core(input_tensor_q, input_tensor_k, input_tensor_v);
}

}  // namespace ttnn::operations::tiny_tiles
