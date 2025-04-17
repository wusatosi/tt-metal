// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/hal.hpp>

#include "my_new_op_operation.hpp"
#include "my_new_op_program_factory.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

void MyNewOpDeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& input2_tensor = input_tensors.at(1);
    std::cout << "*************** MyNewOpDeviceOperation::validate *******************" << std::endl;
    std::cout << "input_tensor.dtype() = " << input_tensor.dtype() << std::endl;
    std::cout << "input2_tensor.dtype() = " << input2_tensor.dtype() << std::endl;
    TT_FATAL(input_tensor.dtype() == input2_tensor.dtype(), "Error");
    TT_FATAL(input_tensor.dtype() == DataType::BFLOAT16, "Error");
    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE && input2_tensor.storage_type() == StorageType::DEVICE,
        "Operands to shard need to be on device!");
    // TT_FATAL(
    // TT_FATAL(input_tensor.buffer() != nullptr, "Operands to shard need to be allocated in buffers on device!");

    // TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Error");
    // //TT_FATAL(this->output_mem_config.is_sharded(), "Error");
    // // TT_FATAL(this->output_mem_config.buffer_type == BufferType::L1, "Error");
    // if (input_tensor.get_layout() == Layout::ROW_MAJOR) {
    //     TT_FATAL(
    //         (*this->output_mem_config.shard_spec).shape[1] * input_tensor.element_size() % hal::get_l1_alignment() ==
    //         0, "Shard page size must currently have L1 aligned page size");
    // }
    // if (input_tensor.get_dtype() != this->output_dtype) {
    TT_FATAL(input_tensor.get_layout() == Layout::TILE, "Error");
    // }
}

std::vector<ttnn::TensorSpec> MyNewOpDeviceOperation::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& input2_tensor = input_tensors.at(1);
    return {TensorSpec(input_tensor.get_tensor_spec())};
}

operation::ProgramWithCallbacks MyNewOpDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& input2_tensor = input_tensors.at(1);
    auto& output_tensor = output_tensors.at(0);
    return detail::my_new_op_multi_core(input_tensor, input2_tensor, output_tensor, this->scalar_multiplier);
}

}  // namespace ttnn::operations::data_movement
