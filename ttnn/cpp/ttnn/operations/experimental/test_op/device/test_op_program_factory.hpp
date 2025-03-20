// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "test_op_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::test_op::program {

struct TestOpProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle test_op_reader_kernel_id;
        tt::tt_metal::KernelHandle test_op_compute_kernel_id;
        tt::tt_metal::KernelHandle test_op_writer_kernel_id;
        uint32_t num_cores = 0;
        uint32_t num_cores_y = 0;
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::test_op::program
