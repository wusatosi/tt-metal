// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <bits/stdint-uintn.h>
#include <math.h>
#include <cstdint>

#include "deinterleave_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/math.hpp"

namespace ttnn::operations::experimental::deinterleave {

DeinterleaveOperation::ProgramFactory::cached_program_t DeinterleaveOperation::ProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt::constants;
    using namespace tt::tt_metal::detail;
    using namespace tt::tt_metal;
    using namespace tt;

    Program program;

    const auto& input = tensor_args.input;

    auto compute_unit_size = [&](const auto& tensor, const auto& data_format) {
        return tensor.get_logical_shape()[-1] * tensor.element_size();
    };

    uint32_t num_units = output.volume() / output.get_logical_shape()[-1];

    auto worker_grid = input.memory_config().shard_spec.value().grid;
    auto num_units_per_core = input.memory_config().shard_spec.value().shape[0];

    uint32_t src_cb_id = CBIndex::c_0;
    auto input_data_format = datatype_to_dataformat_converter(input.get_dtype());
    uint32_t input_unit_size = compute_unit_size(input, input_data_format);
    uint32_t aligned_input_unit_size = round_up_to_mul32(input_unit_size);
    uint32_t src_total_size = input.get_logical_shape()[0] * aligned_input_unit_size;

    tt::tt_metal::CircularBufferConfig src_cb_config =
        tt::tt_metal::CircularBufferConfig(src_total_size, {{src_cb_id, input_data_format}})
            .set_page_size(src_cb_id, aligned_input_unit_size)
            .set_globally_allocated_address(*input.buffer());
    auto src_cb = tt::tt_metal::CreateCircularBuffer(program, worker_grid, src_cb_config);

    uint32_t dst_cb_id = CBIndex::c_1;
    auto output_data_format = datatype_to_dataformat_converter(output.get_dtype());
    uint32_t output_unit_size = compute_unit_size(output, output_data_format);
    uint32_t aligned_output_unit_size = round_up_to_mul32(output_unit_size);
    uint32_t dst_total_size = output.get_logical_shape()[0] * aligned_output_unit_size;

    tt::tt_metal::CircularBufferConfig dst_cb_config =
        tt::tt_metal::CircularBufferConfig(dst_total_size, {{dst_cb_id, output_data_format}})
            .set_page_size(dst_cb_id, aligned_output_unit_size)
            .set_globally_allocated_address(*output.buffer());
    auto dst_cb = tt::tt_metal::CreateCircularBuffer(program, worker_grid, dst_cb_config);

    std::vector<uint32_t> reader_compile_time_args, writer_compile_time_args;

    TT_FATAL(input_unit_size == output_unit_size, "Deinterleave: input and output unit size must be equal");

    auto per_core_width = operation_attributes.input_width;
    auto per_core_height = input.memory_config().shard_spec.value().shape[0] / operation_attributes.input_width;
    log_info(
        tt::LogOp,
        "DeinterleaveOperation::ProgramFactory::create; stride_hw: {}; per core height {} per_core_width {}",
        operation_attributes.stride_hw,
        per_core_height,
        per_core_width);
    auto stick_size_bytes = aligned_input_unit_size;
    reader_compile_time_args = {
        (uint32_t)src_cb_id,
        (uint32_t)dst_cb_id,
        (uint32_t)per_core_width,
        (uint32_t)per_core_height,
        (uint32_t)stick_size_bytes,
        (uint32_t)operation_attributes.stride_hw[0],
        (uint32_t)operation_attributes.stride_hw[1],
        (uint32_t)1  // first half
    };

    writer_compile_time_args = {
        (uint32_t)src_cb_id,
        (uint32_t)dst_cb_id,
        (uint32_t)per_core_width,
        (uint32_t)per_core_height,
        (uint32_t)stick_size_bytes,
        (uint32_t)operation_attributes.stride_hw[0],
        (uint32_t)operation_attributes.stride_hw[1],
        (uint32_t)0  // second half
    };

    auto read_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deinterleave/device/kernels/deinterleave_kernel_rm.cpp",
        worker_grid,
        ReaderDataMovementConfig(reader_compile_time_args, {}));

    auto write_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deinterleave/device/kernels/deinterleave_kernel_rm.cpp",
        worker_grid,
        WriterDataMovementConfig(writer_compile_time_args, {}));

    CoreCoord device_grid = input.device()->logical_grid_size();
    uint32_t num_of_shards = worker_grid.num_cores();
    auto cores = corerange_to_cores(worker_grid, std::nullopt, true);

    uint32_t out_batches = operation_attributes.stride_hw[0] * operation_attributes.stride_hw[1];
    // assuming a single core reads only one stick type from src from the interleaved data, thus fail if cores_in_batch
    // > num_of_shards
    TT_FATAL(
        out_batches <= num_of_shards, "Deinterleave: out_batches {} > num_of_shards {}", out_batches, num_of_shards);
    tt::log_warning(tt::LogOp, "Output buffer address {:#x}", output.buffer()->address());
    tt::log_warning(tt::LogOp, "Input buffer address {:#x}", input.buffer()->address());
    for (const auto& core : cores) {
        auto my_id = core.x + core.y * device_grid.x;

        // number of output batches,for ABAB;CDCD => 4 batches AAAA;BBBB;CCCD;DDDD
        // also turns out this is the number of src core one reads from
        uint32_t num_src_cores = out_batches;
        // number of cores processing one output batch
        uint32_t cores_in_batch = num_of_shards / out_batches;
        // batch this core is processing [0-3]
        uint32_t dst_batch = my_id / cores_in_batch;

        // id of this core in batch
        uint32_t id_in_batch = my_id % cores_in_batch;
        uint32_t start_id = id_in_batch * num_src_cores;
        uint32_t start_y = start_id / device_grid.x;
        uint32_t start_x = start_id % device_grid.x;
        uint32_t end_x = start_x + (num_src_cores % device_grid.x == 0 ? device_grid.x : num_src_cores % device_grid.x);
        uint32_t end_y = start_y + (num_src_cores % device_grid.x == 0 ? num_src_cores / device_grid.x
                                                                       : 1 + num_src_cores / device_grid.x);

        // core should proccess data from start_xy to end_xy, but we dont want every core reading from the same source
        // to start from the same point but offset and process the data in a round robin fashion. cores that read same
        // srcs have same id_in_batch, but different dst_batch
        uint32_t offset_x = (start_x + dst_batch) % device_grid.x;
        uint32_t offset_y = start_y + (start_x + dst_batch) / device_grid.x;

        // src offset are not affected by offset change, because we always read all data from one source and ordering
        // here is not important.
        uint32_t src_width_stride = operation_attributes.stride_hw[1] * stick_size_bytes;
        uint32_t src_height_offset_to_next =
            (operation_attributes.stride_hw[0] - 1) * per_core_width * stick_size_bytes;

        uint32_t src_datum_width_offset = (dst_batch % operation_attributes.stride_hw[0]) * stick_size_bytes;
        uint32_t src_datum_height_offset =
            (dst_batch / operation_attributes.stride_hw[1]) * per_core_width * stick_size_bytes;

        uint32_t src_offset_dm0 = src_datum_width_offset + src_datum_height_offset;
        uint32_t src_offset_dm1 = (per_core_height / 2 * per_core_width * aligned_input_unit_size) +
                                  src_datum_width_offset + src_datum_height_offset;

        // stride to move for one src_core output in the dst buffer
        uint32_t dst_height = per_core_height / operation_attributes.stride_hw[0];
        uint32_t dst_width = per_core_width / operation_attributes.stride_hw[1];

        uint32_t dst_stride = dst_height * dst_width * stick_size_bytes;
        // dst offset now needs some adjustments
        uint32_t dst_offset_dm0 = dst_batch * dst_stride;
        uint32_t dst_offset_dm1 = dst_batch * dst_stride + dst_stride / 2;

        log_warning(
            tt::LogOp,
            "DeinterleaveOperation::ProgramFactory::create; core: {} myid {}, start {}-{}, end {}-{}, dst_batch {}, "
            "id_in_batch {} offset {}-{}",
            core,
            my_id,
            start_y,
            start_x,
            end_y,
            end_x,
            dst_batch,
            id_in_batch,
            offset_y,
            offset_x);
        TT_FATAL(end_x > 0, "Deinterleave: end_x {} == 0", end_x);
        TT_FATAL(end_y > 0, "Deinterleave: end_y {} == 0", end_y);

        TT_FATAL(
            end_x <= device_grid.x,
            "Deinterleave: unsupported configuration. {} end_x {} cannot be larger than device_grid.x {}",
            core,
            end_x,
            device_grid.x);

        TT_FATAL(
            end_y <= device_grid.y,
            "Deinterleave: unsupported configuration. {} end_y {} cannot be larger than device_grid.y {}",
            core,
            end_y,
            device_grid.y);

        log_warning(
            tt::LogOp,
            "src_width_stride {}, src_height_offset_to_next {}",
            src_width_stride,
            src_height_offset_to_next);
        log_warning(
            tt::LogOp,
            "dst_batch {}, src_offset_dm0 {}, src_offset_dm1 {}, dst_stride {}, dst_offset_dm0 {}, dst_offset_dm1 {}",
            dst_batch,
            src_offset_dm0,
            src_offset_dm1,
            dst_stride,
            dst_offset_dm0,
            dst_offset_dm1);
        SetRuntimeArgs(
            program,
            read_kernel_id,
            core,
            {
                (uint32_t)start_x,
                (uint32_t)end_x,
                (uint32_t)start_y,
                (uint32_t)end_y,
                (uint32_t)src_width_stride,
                (uint32_t)src_height_offset_to_next,
                (uint32_t)src_offset_dm0,
                (uint32_t)dst_stride,
                (uint32_t)dst_offset_dm0,
                (uint32_t)offset_x,
                (uint32_t)offset_y,
                (uint32_t)num_src_cores,

            });
        SetRuntimeArgs(
            program,
            write_kernel_id,
            core,
            {
                (uint32_t)start_x,
                (uint32_t)end_x,
                (uint32_t)start_y,
                (uint32_t)end_y,
                (uint32_t)src_width_stride,
                (uint32_t)src_height_offset_to_next,
                (uint32_t)src_offset_dm1,
                (uint32_t)dst_stride,
                (uint32_t)dst_offset_dm1,
                (uint32_t)offset_x,
                (uint32_t)offset_y,
                (uint32_t)num_src_cores,
            });
    }
    // uint32_t start_id = 0;
    // uint32_t num_cores_group_1 = core_group_1.num_cores();
    // auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y);

    return {std::move(program), {read_kernel_id, write_kernel_id, worker_grid}};
}

void DeinterleaveOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto& program = cached_program.program;
    const auto& read_kernel_id = cached_program.shared_variables.read_kernel_id;
    const auto& write_kernel_id = cached_program.shared_variables.write_kernel_id;

    auto input_buffer_address = tensor_args.input.buffer()->address();
    auto output_buffer_address = output.buffer()->address();

    TT_FATAL(false, "to resolve overriding runtime args");
    // std::vector<std::vector<uint32_t>>& reader_args = GetRuntimeArgs(program, read_kernel_id);
    // reader_args[0] = input_buffer_address;
    // std::vector<std::vector<uint32_t>>& writer_args = GetRuntimeArgs(program, write_kernel_id);
    // writer_args[0] = output_buffer_address;
}
}  // namespace ttnn::operations::experimental::deinterleave
