// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpp/pybind11/decorators.hpp"
#include "my_new_op.hpp"
#include "my_new_op_pybind.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/core_coord.hpp>

namespace ttnn::operations::data_movement {

// TODO: Add more descriptions to the arguments
void py_bind_my_new_op(pybind11::module& module) {
    bind_registered_operation(
        module,
        ttnn::my_new_op,
        R"doc(my_new_op(input_tensor: ttnn.Tensor, grid: ttnn.CoreGrid,  int, shard_shape: List[int[2]], shard_scheme: ttl.tensor.TensorMemoryLayout, shard_orientation: ttl.tensor.ShardOrientation, sharded_memory_config: MemoryConfig *, output_dtype: Optional[ttnn.dtype] = None) -> ttnn.Tensor

        Converts a tensor from interleaved to sharded memory layout

        Args:
            * :attr:`input_tensor` (ttnn.Tensor): input tensor
            * :attr:`grid` (ttnn.CoreGrid): Grid of sharded tensor
            * :attr:`shard_shape` (List(int[2])): Sharding shape.
            * :attr:`shard_scheme` (ttl.tensor.TensorMemoryLayout): Sharding scheme(height, width or block).
            * :attr:`shard_orientation` (ttl.tensor.ShardOrientation): Shard orientation (ROW or COL major).
            * :attr:`sharded_memory_config` (MemoryConfig): Instead of shard_shape, shard_scheme and orientation you can provide a single MemoryConfig representing the sharded tensor.

        Keyword Args:
            * :attr:`output_dtype` (Optional[ttnn.DataType]): Output data type, defaults to same as input.
            * :attr:`queue_id`: command queue id

        Example 1 (using grid, shape, scheme, orienttion):

            >>> sharded_tensor = ttnn.sharded_to_interleaved(tensor, ttnn.CoreGrid(3,3), [32,32], ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.ShardOrientation.ROW_MAJOR)


        Example 2 (using sharded memory config):
            >>> sharded_memory_config_dict = dict(
                core_grid=ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1)
                        ),
                    }
                ),
                strategy=ttnn.ShardStrategy.BLOCK,
            ),
            >>> shard_memory_config = ttnn.create_sharded_memory_config(input_shape, **input_sharded_memory_config_args)
            >>> sharded_tensor = ttnn.sharded_to_interleaved(tensor, shard_memory_config)

        )doc",
        ttnn::pybind_arguments_t{py::arg("input_tensor_a"), py::arg("input_tensor_b"), py::arg("input_scalar")});
}

}  // namespace ttnn::operations::data_movement
