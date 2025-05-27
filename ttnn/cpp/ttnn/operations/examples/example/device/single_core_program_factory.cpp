// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "example_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <iostream>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>

// #define USE_WRITER 1

namespace ttnn::operations::examples {
ExampleDeviceOperation::SingleCore::cached_program_t ExampleDeviceOperation::SingleCore::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& output_tensor = tensor_args.output_tensor;
    tt::tt_metal::Program program{};
    tt::tt_metal::IDevice* device = input_tensor.device();

    auto in_buffer = input_tensor.buffer();
    auto out_buffer = output_tensor.buffer();

    // shard spec variables
    auto shard_spec = input_tensor.shard_spec().value();
    auto all_cores = shard_spec.grid;
    uint32_t ncores = shard_spec.num_cores();
    auto out_shard_spec = output_tensor.shard_spec().value();

    std::cout << "dtype : " << input_tensor.get_dtype() << std::endl;
    tt::DataFormat in_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    uint32_t in_tile_size = tt::tt_metal::detail::TileSize(in_df);
    tt::DataFormat out_df = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.get_dtype());
    uint32_t out_tile_size = tt::tt_metal::detail::TileSize(out_df);

    TT_FATAL(in_tile_size == out_tile_size, "Input and output tile size should be same");

    uint32_t num_tile_per_core = 0;

    if (input_tensor.get_dtype() == DataType::BFLOAT8_B) {
        uint32_t ntiles_along_width = std::ceil(shard_spec.shape[1] / (float)tt::constants::TILE_WIDTH);
        uint32_t ntiles_along_height = std::ceil(shard_spec.shape[0] / (float)tt::constants::TILE_HEIGHT);
        num_tile_per_core = ntiles_along_width * ntiles_along_height;
    } else {
        TT_FATAL(
            (shard_spec.shape[1] * datum_size(in_df)) % hal::get_l1_alignment() == 0,
            "Shard width should be multiple of {} to satisfy L1 alignment",
            hal::get_l1_alignment());
        size_t shard_height = shard_spec.shape[0];
        size_t shard_width = shard_spec.shape[1];
        size_t shard_size_in_bytes = shard_height * shard_width * datum_size(in_df);
        TT_FATAL(shard_size_in_bytes % in_tile_size == 0, "Shard Size must be multiple of in_tile_size");
        num_tile_per_core = (shard_size_in_bytes + in_tile_size - 1) / in_tile_size;  // ceil value
    }

    uint32_t buffering_factor = 1;  // data is already fully buffered in the CBs since its sharded
    uint32_t aligned_input_tile_nbytes =
        round_up_to_mul32(in_tile_size);  // will have issue if the page is not multiple of 32
    uint32_t in_cb_pagesize = aligned_input_tile_nbytes;
    uint32_t in_cb_npages = num_tile_per_core * buffering_factor;

    // ----------------- CB -----------------
    // allocate cb interface
    uint32_t in_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig cb_in_config =
        tt::tt_metal::CircularBufferConfig(in_cb_pagesize * in_cb_npages, {{in_cb_index, in_df}})
            .set_page_size(in_cb_index, in_cb_pagesize)
            .set_globally_allocated_address(*in_buffer);  // set allocated address
    auto cb_in = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_in_config);

    // out buffer wrapped by CB
    std::cout << "Address of out buffer : " << out_buffer->address() << std::endl;
    uint32_t out_cb_index = tt::CBIndex::c_1;
#ifdef USE_WRITER
    tt::tt_metal::CircularBufferConfig cb_out_config =
        tt::tt_metal::CircularBufferConfig(in_cb_pagesize * in_cb_npages, {{out_cb_index, out_df}})
            .set_page_size(out_cb_index, in_cb_pagesize);
#else
    tt::tt_metal::CircularBufferConfig cb_out_config =
        tt::tt_metal::CircularBufferConfig(in_cb_pagesize * in_cb_npages, {{out_cb_index, out_df}})
            .set_page_size(out_cb_index, in_cb_pagesize)
            .set_globally_allocated_address(*out_buffer);  // set allocated address
#endif
    auto cb_out = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_out_config);

    bool src_is_dram = in_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    TT_FATAL(src_is_dram == 0, "Input buffer should be in L1");
    std::vector<uint32_t> reader_compile_time_args = {
        in_cb_index,
    };

    bool dst_is_dram = out_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    TT_FATAL(dst_is_dram == 0, "Output buffer should be in L1");

    // ----------------- Kernels -----------------
    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/examples/example/device/kernels/dataflow/reader_unary_sharded.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, {}));

    std::vector<uint32_t> compute_kernel_args_group_1 = {1, num_tile_per_core};

    bool math_approx_mode = false;
    auto compute_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/examples/example/device/kernels/compute/compute_unary_sharded.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args_group_1});

    tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, all_cores, {num_tile_per_core});

    // test for writer kernel
#ifdef USE_WRITER
    uint32_t dst_cb_index = tt::CBIndex::c_2;
    tt::tt_metal::CircularBufferConfig cb_dst_config =
        tt::tt_metal::CircularBufferConfig(in_cb_pagesize * in_cb_npages, {{dst_cb_index, out_df}})
            .set_page_size(dst_cb_index, in_cb_pagesize)
            .set_globally_allocated_address(*out_buffer);  // set allocated address
    auto cb_dst = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_dst_config);

    std::vector<uint32_t> writer_compile_time_args = {
        dst_cb_index,
    };

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/examples/example/device/kernels/dataflow/writer_unary_sharded.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig());

    tt::tt_metal::SetRuntimeArgs(
        program, unary_writer_kernel_id, all_cores, {num_tile_per_core, out_buffer->address()});
#endif

    return {std::move(program), {.cb_src0 = cb_in, .out_cb = cb_out}};
}

void ExampleDeviceOperation::SingleCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    const auto& cb_src0 = cached_program.shared_variables.cb_src0;
    const auto& out_cb = cached_program.shared_variables.out_cb;

    auto in_buffer = tensor_args.input_tensor.buffer();
    auto out_buffer = tensor_args.output_tensor.buffer();
    tt::tt_metal::UpdateDynamicCircularBufferAddress(program, cb_src0, *in_buffer);
    tt::tt_metal::UpdateDynamicCircularBufferAddress(program, out_cb, *out_buffer);
}

}  // namespace ttnn::operations::examples
