// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/conv/conv2dmatmul/device/conv2dmatmul_program_factory.hpp"
#include "hostdevcommon/kernel_structs.h"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/math.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
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
    auto device = input_tensor.device();
    auto program = tt::tt_metal::CreateProgram();

    auto compute_grid_size = device->compute_with_storage_grid_size();
    Buffer* src0_buffer = input_tensor.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);

    ttnn::Shape output_padded_shape = output.padded_shape();
    ttnn::Shape input_padded_shape = input_tensor.padded_shape();

    tt::log_info("cb_data_format: {}", cb_data_format);
    tt::log_info("single_tile_size: {}", single_tile_size);
    tt::log_info("input_tensor_shape: {}", input_padded_shape);
    tt::log_info("output_tensor_shape: {}", output_padded_shape);
    auto stick_nbytes =
        output_padded_shape[3] * tt::datum_size(tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype()));
    uint32_t ntiles_per_row = tt::div_up(input_padded_shape[-1], TILE_WIDTH);
    uint32_t ntiles = input_tensor.volume() / TILE_HW;
    uint32_t num_blocks = std::ceil(static_cast<float>(ntiles) / ntiles_per_row);

    auto grid_size = device->compute_with_storage_grid_size();
    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        ttnn::split_blocks_for_tilize(grid_size, num_blocks);
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_tiles = ntiles_per_row;
    uint32_t double_buffer = 2;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(
            num_input_tiles * single_tile_size * double_buffer, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t src1_cb_index = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig cb_src1_config =
        tt::tt_metal::CircularBufferConfig(
            num_input_tiles * single_tile_size * double_buffer, {{src1_cb_index, cb_data_format}})
            .set_page_size(src1_cb_index, single_tile_size);
    auto cb_src1 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    bool src0_is_dram = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    std::vector<uint32_t> reader_compile_time_args = {
        ntiles_per_row,
        src0_cb_index,
    };
    std::vector<uint32_t> writer_compile_time_args = {
        batch_size,
        input_height,
        input_width,
        stride[0],
        stride[1],
        stick_nbytes,
        ntiles_per_row,
        datum_size(cb_data_format),
        src1_cb_index,
    };

    // Tilized reader
    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/conv/conv2dmatmul/device/kernels/dataflow/"
        "reader_conv2dmatmul_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/conv/conv2dmatmul/device/kernels/dataflow/"
        "writer_conv2dmatmul_pad_dims_interleaved.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> compute_compile_time_args = {
        nblocks_per_core,
        ntiles_per_row,
        src0_cb_index,
        src1_cb_index,
    };

    std::vector<uint32_t> compute_compile_time_args_cliff = {
        nblocks_per_core_cliff,
        ntiles_per_row,
        src0_cb_index,
        src1_cb_index,
    };

    bool fp32_dest_acc_en = cb_data_format == tt::DataFormat::Float32;
    tt::tt_metal::KernelHandle compute_kernel_id, compute_kernel_id_cliff;
    compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize.cpp",
        core_range,
        tt::tt_metal::ComputeConfig{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .compile_args = compute_compile_time_args,
        });

    if (core_range_cliff.ranges().size() > 0) {
        compute_kernel_id_cliff = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize.cpp",
            core_range_cliff,
            tt::tt_metal::ComputeConfig{
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .compile_args = compute_compile_time_args_cliff,
            });
    }

    uint32_t ncores_full = ncores;
    auto full_cores = all_cores;
    if (nblocks_per_core_cliff > 0 && nblocks_per_core_cliff < nblocks_per_core) {
        ncores_full -= 1;
        full_cores = core_range;
    }
    uint32_t tile_start_id = 0;
    auto ncores_x = grid_size.x;
    auto ncores_y = std::ceil(static_cast<float>(ncores) / ncores_x);
    auto cores = grid_to_cores(ncores_x * ncores_y, ncores_x, ncores_y, true);
    std::vector<CoreCoord> cores_with_rtargs;
    for (auto i = 0; i < cores.size(); i++) {
        CoreCoord core = cores[i];
        if (!full_cores.contains(core)) {
            continue;
        }
        std::vector<uint32_t> reader_runtime_args = {
            src0_buffer->address(),
            tile_start_id,
            nblocks_per_core,
        };
        std::vector<uint32_t> writer_runtime_args = {
            dst_buffer->address(),
            tile_start_id,
            nblocks_per_core,
        };
        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_runtime_args);
        tile_start_id += nblocks_per_core * ntiles_per_row;
        cores_with_rtargs.push_back(core);
    }
    if (ncores_full < ncores) {
        CoreCoord core = CoreCoord{ncores_full % ncores_x, ncores_full / ncores_x};
        std::vector<uint32_t> reader_runtime_args = {
            src0_buffer->address(),
            tile_start_id,
            nblocks_per_core_cliff,
        };
        std::vector<uint32_t> writer_runtime_args = {
            dst_buffer->address(),
            tile_start_id,
            nblocks_per_core_cliff,
        };
        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_runtime_args);
        cores_with_rtargs.push_back(core);
    }

    auto override_runtime_args_callback =
        [unary_reader_kernel_id, unary_writer_kernel_id, compute_kernel_id, compute_kernel_id_cliff, cores_with_rtargs](
            // auto override_runtime_args_callback = [unary_reader_kernel_id, unary_writer_kernel_id](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_tensors,
            const std::vector<Tensor>& output_tensors) {
            auto src_dram_buffer = input_tensors.at(0).buffer();

            auto dst_dram_buffer = output_tensors.at(0).buffer();
            for (auto i = 0; i < cores_with_rtargs.size(); i++) {
                CoreCoord core = cores_with_rtargs[i];
                {
                    auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_reader_kernel_id, core);
                    runtime_args[0] = src_dram_buffer->address();
                }

                {
                    auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_writer_kernel_id, core);
                    runtime_args[0] = dst_dram_buffer->address();
                }
            }
        };

    return {std::move(program), {}};
}

}  // namespace ttnn::operations::conv::conv2dmatmul::detail
