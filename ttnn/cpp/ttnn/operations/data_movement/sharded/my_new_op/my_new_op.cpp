// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/common/queue_id.hpp"
#include "ttnn/run_operation.hpp"
#include "device/my_new_op_operation.hpp"
#include "my_new_op.hpp"
#include <tt-metalium/work_split.hpp>

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

ttnn::Tensor MyNewOpOperation::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& input2_tensor,
    const float scalar_multiplier) {  //,
                                      // const MemoryConfig& sharded_memory_config,
                                      // const std::optional<DataType>& data_type_arg) {
    std::cout << "*************** MyNewOpOperation::invoke *******************" << std::endl;
    std::cout << "scalar = " << scalar_multiplier << std::endl;
    return operation::run(
               MyNewOpDeviceOperation{//    .output_mem_config = sharded_memory_config,
                                      //    .output_dtype = data_type_arg.value_or(input_tensor.get_dtype()),
                                      .scalar_multiplier = scalar_multiplier},
               {input_tensor, input2_tensor})
        .at(0);
}

}  // namespace ttnn::operations::data_movement
