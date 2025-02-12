// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_unicast_device_operation.hpp"
#include "fabric_unicast_program_factory.hpp"

namespace ttnn::operations::ccl {

void FabricUnicast::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    // no validation
}

std::vector<TensorSpec> FabricUnicast::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto output_shape = input_tensors.at(0).get_logical_shape();

    auto input_spec = TensorSpec(
        output_shape, TensorLayout(input_tensor.get_dtype(), PageConfig(input_tensor.layout()), output_mem_config));
    return {input_spec};
}

std::vector<Tensor> FabricUnicast::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    auto output_specs = compute_output_specs(input_tensors);
    const auto& input_tensor = input_tensors.at(0);
    return {
        create_device_tensor(output_specs[0], this->mesh_device->get_device(this->dest_device_id)),
    };
}

operation::ProgramWithCallbacks FabricUnicast::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return detail::fabric_unicast_interleaved(input_tensor, output_tensors.at(0), this->mesh_device->id());
}

}  // namespace ttnn::operations::ccl
