// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv_distribute_pybind.hpp"
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpp/pybind11/decorators.hpp"
#include "conv_distribute.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

namespace detail {

template <typename data_movement_conv_distribute_operation_t>
void bind_conv_distribute(
    pybind11::module& module, const data_movement_conv_distribute_operation_t& operation, const char* doc) {
    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const data_movement_conv_distribute_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::MemoryConfig& distributed_mem_config,
               int block_size,
               int num_blocks_per_core,
               int num_cores_with_extra_block,
               QueueId queue_id) -> ttnn::Tensor {
                return self(
                    queue_id,
                    input_tensor,
                    distributed_mem_config,
                    block_size,
                    num_blocks_per_core,
                    num_cores_with_extra_block);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("distributed_mem_config"),
            py::arg("block_size"),
            py::arg("num_blocks_per_core"),
            py::arg("num_cores_with_extra_block"),
            py::kw_only(),
            py::arg("queue_id") = 0,
        });
}

}  // namespace detail

void py_bind_conv_distribute(pybind11::module& module) {
    detail::bind_conv_distribute(
        module,
        ttnn::conv_distribute,
        R"doc(
            conv_distribute(input_tensor: ttnn.Tensor, distributed_mem_config: ttnn.ShardSpec, block_size: int, num_blocks_per_core: int, num_cores_with_extra_block: int) -> ttnn.Tensor
            Input and output tensors must be height sharded.
            Performs conv distribute operation operation, using the input tensor, given cores, and given sizes of shards.
            The resulting tensor will have shards of uneven sizes according to the parameters provided.
        )doc");
}

}  // namespace ttnn::operations::data_movement
