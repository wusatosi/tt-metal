// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_decode_program_factory.hpp"

#include <optional>

#include <tt-metalium/buffer.hpp>
#include "sdpa_decode_op.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/logger.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operation.hpp"

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::tiny_tiles::detail {

// implementation of softmax with optional scale/mask (see the header for input_tensor more detailed description)
operation::ProgramWithCallbacks sdpa_decode_multi_core(
    const Tensor& input_tensor_q, const Tensor& input_tensor_k, const Tensor& input_tensor_v) {
    Program program = CreateProgram();

    std::vector<uint32_t> reader_compile_time_args_common = {};

    std::vector<uint32_t> writer_compile_time_args_common = {};

    std::vector<uint32_t> compute_compile_time_args_common = {};

    // // Compute
    // auto compute_kernels_id = CreateKernel(
    //     program,
    //     "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp",
    //     core_grid,
    //     tt_metal::ComputeConfig{
    //         .math_fidelity = math_fidelity,
    //         .fp32_dest_acc_en = fp32_dest_acc_en,
    //         .math_approx_mode = math_approx_mode,
    //         .compile_args = compute_compile_time_args_common,
    //         .defines = defines});

    // // Reader
    // auto reader_kernels_id = CreateKernel(
    //     program,
    //     "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/reader_decode_all.cpp",
    //     core_grid,
    //     tt_metal::ReaderDataMovementConfig(reader_compile_time_args_common, defines));

    // // Writer
    // auto writer_kernels_id = CreateKernel(
    //     program,
    //     "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/writer_decode_all.cpp",
    //     core_grid,
    //     tt_metal::WriterDataMovementConfig(writer_compile_time_args_common, defines));

    // uint32_t q_addr = q_buffer->address();
    // uint32_t k_addr = k_buffer->address();
    // uint32_t v_addr = v_buffer->address();

    // Set rt args
    // for (uint32_t i = 0; i < num_active_cores; ++i) {
    //     CoreCoord core = core_group[i];
    // }

    // auto override_runtime_arguments_callback =
    //     [num_active_cores,
    //      core_group,
    //      reader_kernels_id,
    //      writer_kernels_id,
    //      compute_kernels_id,
    //      num_output_cores,
    //      is_output_sharded,
    //      cb_out4_id,
    //      B](
    //         const void* operation,
    //         Program& program,
    //         const std::vector<Tensor>& input_tensors,
    //         const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    //         const std::vector<Tensor>& output_tensors) {
    //         auto q_buffer = input_tensors.at(0).buffer();
    //         auto k_buffer = input_tensors.at(1).buffer();
    //         auto v_buffer = input_tensors.at(2).buffer();

    //         auto out0_buffer = output_tensors.at(0).buffer();
    //         uint32_t q_addr = q_buffer->address();
    //         uint32_t k_addr = k_buffer->address();
    //         uint32_t v_addr = v_buffer->address();
    //         uint32_t out_addr = out0_buffer->address();

    //         auto& reader_args_by_core = GetRuntimeArgs(program, reader_kernels_id);
    //         auto& writer_args_by_core = GetRuntimeArgs(program, writer_kernels_id);
    //         auto& compute_args_by_core = GetRuntimeArgs(program, compute_kernels_id);

    //         // Set rt args
    //         for (uint32_t i = 0; i < num_active_cores; ++i) {
    //             CoreCoord core = core_group[i];

    //             auto& reader_args = reader_args_by_core[core.x][core.y];
    //             auto& writer_args = writer_args_by_core[core.x][core.y];
    //             auto& compute_args = compute_args_by_core[core.x][core.y];

    //             // reader runtime args
    //             // uint32_t arg_idx = 0;
    //             // reader_args[arg_idx++] = q_addr;
    //             // reader_args[arg_idx++] = k_addr;
    //             // reader_args[arg_idx++] = v_addr;

    //             // writer runtime args
    //             // arg_idx = 0;
    //             // writer_args[arg_idx++] = out_addr;

    //             // compute runtime args
    //             arg_idx = 0;
    //         }

    //         if (is_output_sharded) {
    //             UpdateDynamicCircularBufferAddress(program, cb_out4_id, *out0_buffer);
    //         }
    //     };

    // return {.program = std::move(program), .override_runtime_arguments_callback =
    // override_runtime_arguments_callback};
    return {.program = std::move(program), .override_runtime_arguments_callback = {}};
}

}  // namespace ttnn::operations::tiny_tiles::detail
