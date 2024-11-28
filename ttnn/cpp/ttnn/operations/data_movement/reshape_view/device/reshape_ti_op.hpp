// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape_common.hpp"
namespace ttnn {

struct TILE_RESHAPE_STRUCT {
    const ttnn::Shape output_shape;
    MemoryConfig output_mem_config;
    uint32_t pad_value;


    //Required functions to all tensor op functions
    void update_structure (const Tensor& input_tensor);
    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<SimpleShape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
};



}// namespace ttnn

namespace ttnn::operations::data_movement::tile_reshape{

operation::ProgramWithCallbacks tile_reshape_preparer(const Tensor& input, const Tensor& output, uint32_t pad_value);
}
