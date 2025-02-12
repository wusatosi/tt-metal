// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_log.h>

namespace ttnn::operations::reduction::detail {

operation::ProgramWithCallbacks fabric_unicast_single_core_interleaved(
    const Tensor& input_tensor, Tensor& output_tensor, const uint32_t dest_mesh_id) {
    using namespace tt::constants;
    tt::tt_metal::Program program{};

    const auto input_buffer = input_tensor.buffer();
    const auto output_buffer = output_tensor.buffer();

    const bool input_is_dram = input_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    const bool output_is_dram = output_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    const uint32_t dst_dev_id = output_tensor.get_device().id;

    const auto input_shape = input_tensor.get_logical_shape();
    const uint32_t num_pages = std::accumulate(input_shape.cbegin(), input_shape.cend() - 1, std::multiplies<uint32_t>);
    const uint32_t page_size_bytes = input_shape[-1] * sizeof(convert_to_data_type(input_tensor.dtype()));

    const CoreRange core({0, 0}, {0, 0});

    const std::vector<uint32_t> reader_compile_time_args = {input_is_dram, output_is_dram};
    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/reader_unary_interleaved_fabric_unicast.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    const std::vector<uint32_t> reader_runtime_args = {
        input_buffer->address(), output_buffer->address(), dest_mesh_id, dst_dev_id, page_size_bytes, number_of_pages};

    SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_runtime_args);

    return {std::move(program)};

}  // namespace ttnn::operations::ccl::detail
