// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/conv/conv2dmatmul/device/conv2dmatmul_program_factory.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/math.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include <tt-metalium/tt_align.hpp>
namespace ttnn::operations::conv::conv2dmatmul::detail {

using namespace tt::constants;
using namespace tt::tt_metal;
tt::tt_metal::operation::ProgramWithCallbacks conv2dmatmul_tile_reader_writer(
    const Tensor& input_tensor,
    const Tensor& output,
    const uint32_t in_channels,
    const uint32_t out_channels,
    const uint32_t batch_size,
    const uint32_t input_height,
    const uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride) {
    // This should allocate a DRAM buffer on the device
    auto device = input_tensor.device();
    auto program = tt::tt_metal::CreateProgram();

    Buffer* src0_buffer = input_tensor.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");
    std::cout << "inside program factory " << std::endl;
    CoreRange core({0, 0}, {0, 0});

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);

    ttnn::Shape output_padded_shape = output.padded_shape();
    ttnn::Shape input_padded_shape = input_tensor.padded_shape();

    tt::log_info("cb_data_format: {}", cb_data_format);
    tt::log_info("single_tile_size: {}", single_tile_size);
    tt::log_info("input_tensor_shape: {}", input_padded_shape);
    tt::log_info("output_tensor_shape: {}", output_padded_shape);
    auto stick_nbytes = input_padded_shape[3] * 2;
    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 1;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * stick_nbytes, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, stick_nbytes);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t src1_cb_index = 1;  // For output buffer
    uint32_t num_pad_tiles = 1;
    tt::tt_metal::CircularBufferConfig cb_src1_config =
        tt::tt_metal::CircularBufferConfig(num_pad_tiles * stick_nbytes, {{src1_cb_index, cb_data_format}})
            .set_page_size(src1_cb_index, stick_nbytes);
    auto cb_src1 = tt::tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    uint32_t src2_cb_index = 2;  // For output buffer
    tt::tt_metal::CircularBufferConfig cb_src2_config =
        tt::tt_metal::CircularBufferConfig(num_pad_tiles * stick_nbytes, {{src2_cb_index, cb_data_format}})
            .set_page_size(src2_cb_index, stick_nbytes);
    auto cb_src2 = tt::tt_metal::CreateCircularBuffer(program, core, cb_src2_config);

    uint32_t num_unpadded_tiles = input_tensor.volume() / TILE_HW;

    const std::array reader_kernel_args = {
        src0_buffer->address(),
        dst_buffer->address(),
        batch_size,
        input_width,
        input_height,
        kernel_size[0],
        kernel_size[1],
        stick_nbytes,
        std::uint32_t{0},
    };
    const std::array writer_kernel_args = {
        dst_buffer->address(),
        input_width,
        input_height,
        stick_nbytes,
        std::uint32_t{0},
    };

    const std::array compute_kernel_args = {
        std::uint32_t{0},
        num_unpadded_tiles,
    };
    // Reader compile-time args
    // Data is 32 byte aligned
    bool src0_is_dram = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    std::vector<uint32_t> reader_compile_time_args = {// interleaved accessor args
                                                      (std::uint32_t)src0_is_dram};
    std::vector<uint32_t> writer_compile_time_args = {// interleaved accessor args
                                                      (std::uint32_t)src0_cb_index,
                                                      (std::uint32_t)src1_cb_index,
                                                      (std::uint32_t)dst_is_dram};
    // Tilized reader
    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/conv/conv2dmatmul/device/kernels/dataflow/"
        "reader_conv2dmatmul_interleaved_start_id.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/conv/conv2dmatmul/device/kernels/dataflow/"
        "writer_conv2dmatmul_pad_dims_interleaved.cpp",
        core,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> compute_compile_time_args = {};
    bool fp32_dest_acc_en = cb_data_format == tt::DataFormat::Float32;
    // auto compute_kernel_id = tt::tt_metal::CreateKernel(
    //     program,
    //     "ttnn/cpp/ttnn/operations/conv/conv2dmatmul/device/kernels/compute/compute_conv2dmatmul_single_core.cpp",
    //     core,
    //     tt::tt_metal::ComputeConfig{
    //         .fp32_dest_acc_en = fp32_dest_acc_en,
    //         .compile_args = compute_compile_time_args,
    //     });

    tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_kernel_args);

    tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_kernel_args);

    // tt::tt_metal::SetRuntimeArgs(program, compute_kernel_id, core, compute_kernel_args);

    // auto override_runtime_args_callback = [unary_reader_kernel_id, unary_writer_kernel_id, compute_kernel_id](
    auto override_runtime_args_callback = [unary_reader_kernel_id, unary_writer_kernel_id](
                                              const void* operation,
                                              Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>& optional_tensors,
                                              const std::vector<Tensor>& output_tensors) {
        auto src_dram_buffer = input_tensors.at(0).buffer();

        auto dst_dram_buffer = output_tensors.at(0).buffer();

        CoreCoord core = {0, 0};

        {
            auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_reader_kernel_id, core);
            runtime_args[0] = src_dram_buffer->address();
        }

        {
            auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_writer_kernel_id, core);
            runtime_args[0] = dst_dram_buffer->address();
        }
    };

    return {std::move(program), {}};
}

}  // namespace ttnn::operations::conv::conv2dmatmul::detail
