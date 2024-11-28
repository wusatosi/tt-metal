// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/reshape_view/device/reshape_ti_op.hpp"
#include "tt_metal/host_api.hpp"

#include <cstdint>

namespace ttnn {

void TILE_RESHAPE_STRUCT::validate(const std::vector<Tensor>& input_tensors) const {
    //Validate the input tensor
    const Tensor& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to reshape need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.get_layout() == Layout::TILE, "This function is for RM->RM");
    TT_FATAL(this->output_mem_config.memory_layout == input_tensor_a.memory_config().memory_layout, "Output tensor must have the same memory layout as input tensor");
}

std::vector<SimpleShape> TILE_RESHAPE_STRUCT::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    return {output_shape.logical_shape()};
}

std::vector<Tensor> TILE_RESHAPE_STRUCT::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    //Create the output tensor
    auto mem_config = this->output_mem_config;
    const auto& input_tensor_a = input_tensors.at(0);
    if (input_tensor_a.memory_config().is_sharded()) {
        auto shard_spec = input_tensor_a.shard_spec().value();
        shard_spec.shape[0] = output_shape[0]*output_shape[1];
        mem_config.shard_spec = shard_spec;
    }
    return {create_device_tensor(output_shape, input_tensor_a.get_dtype(), input_tensor_a.get_layout(), input_tensor_a.device(), mem_config, input_tensor_a.tile())};
}

operation::ProgramWithCallbacks TILE_RESHAPE_STRUCT::create_program( const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const
{
    return ttnn::operations::data_movement::tile_reshape::tile_reshape_preparer(input_tensors.at(0), output_tensors.at(0),this->pad_value);
}
} // namespace ttnn
