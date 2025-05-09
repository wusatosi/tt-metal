// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cpp/ttnn/operations/normalization/layernorm_distributed/device/layernorm_pre_all_gather_op.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/math.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/circular_buffer.hpp>
#include <optional>
#include <variant>

using uint32_t = std::uint32_t;
using namespace tt::constants;

namespace ttnn::operations::normalization {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
inline bool is_dram(const Tensor& input_tensor) {
    return input_tensor.memory_config().buffer_type() == BufferType::DRAM;
}
inline bool is_dram(const std::optional<const Tensor>& input_tensor) {
    return input_tensor.has_value() ? is_dram(input_tensor.value()) : true;
}
inline bool is_dram(const Buffer* b) { return b->buffer_type() == BufferType::DRAM; }

inline uint16_t bfloat16(float float_num) {
    uint32_t uint32_data;
    TT_ASSERT(sizeof float_num == sizeof uint32_data);

    uint32_data = *reinterpret_cast<uint32_t*>(&float_num);
    // just move upper 16 to lower 16 (truncate)
    uint32_data = (uint32_data >> 16);

    // store lower 16 as 16-bit uint
    return (uint16_t)uint32_data;
}
inline uint32_t pack_two_bfloat16_into_uint32(std::pair<uint16_t, uint16_t> two_bfloats) {
    // first -> lower 16
    // second -> upper 16
    return (uint32_t)two_bfloats.first | ((uint32_t)two_bfloats.second << 16);
}
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

operation::ProgramWithCallbacks layernorm_pre_allgather_multi_core(
    const Tensor& a,
    Tensor& output,
    LayerNormDistributedType norm_type,
    DeviceComputeKernelConfig compute_kernel_config) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    const bool is_rmsnorm = norm_type == LayerNormDistributedType::RMSNORM;
    const auto shape = a.get_padded_shape();
    const uint32_t W = shape[-1], H = shape[-2];
    const uint32_t HW = H * W;
    const uint32_t NC = a.volume() / HW;

    // Kernels are configured to support BFLOAT8_B, but bad pcc so we need mixed precision support in compute
    const auto& a_dtype = a.get_dtype();

    const uint32_t Wt = W / TILE_WIDTH;
    const uint32_t Ht = H / TILE_HEIGHT;
    const uint32_t tile_cols_per_device = is_rmsnorm ? 1 : 2;

    uint32_t num_tile_rows = NC * Ht;

    ////////////////////////////////////////////////////////////////////////////
    //                       Device Setup
    //////////////////////////////////////////////////////////////////////////
    IDevice* device = a.device();

    ////////////////////////////////////////////////////////////////////////////
    //                Circular Buffer Data Format Setup
    //////////////////////////////////////////////////////////////////////////
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    uint32_t block_size = 1;  // find_max_divisor(Wt, 8);
    uint32_t writer_block_size = 1;

    tt::DataFormat in_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    tt::DataFormat out_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    uint32_t in_single_tile_size = tt::tt_metal::detail::TileSize(in_data_format);
    uint32_t out_single_tile_size = tt::tt_metal::detail::TileSize(out_data_format);
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);
    uint32_t bfloat16_tile_size = tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b);

    tt::log_info("in_data_format: {}", in_data_format);
    tt::log_info("out_data_format: {}", out_data_format);

    tt::DataFormat inb_data_format = tt::DataFormat::Invalid;
    uint32_t inb_single_tile_size = 0;

    auto a_addr = a.buffer()->address();
    auto dst_addr = output.buffer()->address();

    uint32_t num_tiles = a.volume() / TILE_HW;

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    /*
    in0_cb: a
    in1_cb: 1 (reduction scalar)

    output CB is packed such that the first tile is for x**2 stats, second tile is for x stats
    in RMSNorm, only first tile has valid data.

    intermed0_cb: xˆ2
    out0_cb: [sum(xˆ2), sum(x)]  # For layernorm
    out0_cb: [sum(xˆ2)]  # RMSNorm

    */
    const uint32_t double_buffer_constant = 2;
    const uint32_t in0_tiles = Wt * double_buffer_constant;
    const uint32_t in1_tiles = 1;  // reduce scalar

    const uint32_t intermed0_tiles = Wt * double_buffer_constant;  // xˆ2
    uint32_t out0_tiles = 1;
    if (!is_rmsnorm) {
        out0_tiles = 2;
    }

    TT_ASSERT(
        W <= TILE_WIDTH * in0_tiles &&
        "W exceeds the maximum supported size of tile buffer (kernel limitation right now).");
    TT_ASSERT(
        in0_tiles % block_size == 0 &&
        "Size of buffer must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(
        intermed0_tiles % block_size == 0 &&
        "Size of buffer must be divisible by the size of block used by the reader and compute kernel.");

    auto grid_size = device->compute_with_storage_grid_size();
    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_tile_rows_per_core_group_1,
         num_tile_rows_per_core_group_2] = tt::tt_metal::split_work_to_cores(grid_size, num_tile_rows, true);

    tt::log_info("num_cores: {}", num_cores);
    tt::log_info("grid_size: {}", grid_size);
    tt::log_info("core_group_1: {}", core_group_1.str());
    tt::log_info("num_tile_rows_per_core_group_1: {}", num_tile_rows_per_core_group_1);
    tt::log_info("core_group_2: {}", core_group_2.str());
    tt::log_info("num_tile_rows_per_core_group_2: {}", num_tile_rows_per_core_group_2);

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    Program program = CreateProgram();

    std::vector<uint32_t> reader_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t)is_dram(a),
        (std::uint32_t)block_size,
    };

    std::vector<uint32_t> writer_compile_time_args = {// interleaved accessor args
                                                      (std::uint32_t)is_dram(output),
                                                      (std::uint32_t)writer_block_size};

    bool tile_dtype_is_bfloat16 = a.get_dtype() == tt::tt_metal::DataType::BFLOAT16;
    std::map<string, string> compute_defines;

    if (is_rmsnorm) {
        compute_defines["RMSNORM"] = "1";
    }

    auto reader_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/"
        "reader_unary_interleaved_ln_rm_gb_pre_allgather.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    auto writer_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/"
        "writer_unary_interleaved_start_id_blocked.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> compute_args = {Wt, block_size};

    auto compute_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/"
        "layernorm_pre_allgather.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_args,
            .defines = compute_defines});

    // Create circular buffers
    // c_in0 -> a
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(in0_tiles * in_single_tile_size, {{tt::CBIndex::c_0, in_data_format}})
            .set_page_size(tt::CBIndex::c_0, in_single_tile_size);
    CreateCircularBuffer(program, all_cores, cb_src0_config);
    // c_in1 -> reduce scalar
    CircularBufferConfig cb_reduce_config =
        CircularBufferConfig(in1_tiles * bfloat16_tile_size, {{tt::CBIndex::c_1, cb_data_format}})
            .set_page_size(tt::CBIndex::c_1, bfloat16_tile_size);
    CreateCircularBuffer(program, all_cores, cb_reduce_config);

    // LN and RMS shared intermediates //
    // c_intermed0 -> xˆ2
    CircularBufferConfig cb_intermed0_config =
        CircularBufferConfig(intermed0_tiles * single_tile_size, {{tt::CBIndex::c_6, cb_data_format}})
            .set_page_size(tt::CBIndex::c_6, single_tile_size);
    CreateCircularBuffer(program, all_cores, cb_intermed0_config);

    CircularBufferConfig cb_out0_config =
        CircularBufferConfig(out0_tiles * out_single_tile_size, {{tt::CBIndex::c_14, out_data_format}})
            .set_page_size(tt::CBIndex::c_14, out_single_tile_size);
    CreateCircularBuffer(program, all_cores, cb_out0_config);

    // Log all circular buffers with program.circular_buffers_on_corerange(all_cores), which returns
    // std::vector<std::shared_ptr<CircularBuffer>>
    for (const auto& cb : program.circular_buffers_on_corerange(*all_cores.ranges().begin())) {
        for (const auto index : cb->buffer_indices()) {
            tt::log_info("cb_id {}", index);
            tt::log_info("page_size: {}", cb->page_size(index));
            tt::log_info("num_pages: {}", cb->num_pages(index));
            tt::log_info("data_format: {}", cb->data_format(index));
        }
    }

    uint32_t curr_row = 0;
    float winv = 1.0f;
    auto bfloat_winv_value = bfloat16(winv);
    uint32_t packed_winv_value = pack_two_bfloat16_into_uint32({bfloat_winv_value, bfloat_winv_value});
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};

        uint32_t num_tile_rows_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        uint32_t in_tile_offset = curr_row * Wt;
        uint32_t out_tile_offset = curr_row * out0_tiles;

        SetRuntimeArgs(
            program, reader_kernels_id, core, {a_addr, num_tile_rows_per_core, Wt, in_tile_offset, packed_winv_value});
        SetRuntimeArgs(program, compute_kernels_id, core, {num_tile_rows_per_core});
        SetRuntimeArgs(
            program, writer_kernels_id, core, {dst_addr, num_tile_rows_per_core * out0_tiles, out_tile_offset});
        curr_row += num_tile_rows_per_core;
    }

    auto override_runtime_arguments_callback =
        [reader_kernel_id = reader_kernels_id, writer_kernel_id = writer_kernels_id, num_cores, grid_size](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto& input_tensor = input_tensors.at(0);

            const auto input_addr = input_tensor.buffer()->address();

            const auto& output_tensor = output_tensors.at(0);
            const auto output_addr = output_tensor.buffer()->address();

            auto& reader_runtime_args_by_core = GetRuntimeArgs(program, reader_kernel_id);
            auto& writer_runtime_args_by_core = GetRuntimeArgs(program, writer_kernel_id);

            for (uint32_t i = 0; i < num_cores; ++i) {
                const CoreCoord core = {i % grid_size.x, i / grid_size.x};

                {
                    auto& reader_args = reader_runtime_args_by_core.at(core.x).at(core.y);

                    reader_args[0] = input_addr;
                }

                {
                    auto& writer_args = writer_runtime_args_by_core.at(core.x).at(core.y);
                    writer_args[0] = output_addr;
                }
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

operation::ProgramWithCallbacks layernorm_pre_allgather_multi_core_sharded(
    const Tensor& a,
    Tensor& output,
    LayerNormDistributedType norm_type,
    DeviceComputeKernelConfig compute_kernel_config) {
    tt::log_info("starting: {}", 0);
    using namespace CMAKE_UNIQUE_NAMESPACE;

    const bool is_rmsnorm = norm_type == LayerNormDistributedType::RMSNORM;
    const auto shape = a.get_padded_shape();
    const uint32_t W = shape[-1], H = shape[-2];
    const uint32_t HW = H * W;
    const uint32_t NC = a.volume() / HW;

    // Kernels are configured to support BFLOAT8_B, but bad pcc so we need mixed precision support in compute
    const auto& a_dtype = a.get_dtype();

    const uint32_t Wt = W / TILE_WIDTH;
    const uint32_t Ht = H / TILE_HEIGHT;
    const uint32_t tile_cols_per_device = is_rmsnorm ? 1 : 2;

    uint32_t num_tile_rows = NC * Ht;

    // Get the grid size to determine how many cores we have available
    auto grid_size = a.device()->compute_with_storage_grid_size();

    // Calculate sharding factors based on grid dimensions and input shape
    // For sequences, we want to use grid.y dimension
    // For embedding dimension, we want to use grid.x dimension

    // First determine cores per dimension - maximize usage while ensuring divisibility
    uint32_t cores_per_sequence = std::min<uint32_t>(grid_size.y, Ht);  // Don't use more cores than rows
    uint32_t cores_per_dim = std::min<uint32_t>(grid_size.x - 1, Wt);   // Reserve right-most column for merge cores

    // Ensure the dimensions are divisible properly
    while (Ht % cores_per_sequence != 0 && cores_per_sequence > 1) {
        cores_per_sequence--;
    }

    while (Wt % cores_per_dim != 0 && cores_per_dim > 1) {
        cores_per_dim--;
    }

    // Calculate dimension shard factor and tiles per shard
    uint32_t dim_shard_factor = cores_per_dim;
    uint32_t tiles_per_dim_shard = Wt / dim_shard_factor;

    // Calculate total cores used
    uint32_t total_cores_used = cores_per_sequence * cores_per_dim + cores_per_sequence;  // Regular cores + merge cores

    // Position merge cores in the column after the regular grid
    uint32_t merge_core_x = cores_per_dim;  // Position merge cores right after the regular grid

    // Create core ranges for different core groups
    CoreRange regular_cores_range({0, 0}, {cores_per_dim - 1, cores_per_sequence - 1});

    // Create a list of merge cores (one per sequence)
    std::vector<CoreCoord> merge_cores;
    std::vector<CoreRange> merge_core_ranges;

    for (uint32_t seq_idx = 0; seq_idx < cores_per_sequence; seq_idx++) {
        // Place merge cores in the column after the regular grid
        CoreCoord merge_core = {merge_core_x, seq_idx};
        merge_cores.push_back(merge_core);
        merge_core_ranges.push_back(CoreRange(merge_core, merge_core));
    }

    // Create a CoreRangeSet for merge cores
    CoreRangeSet merge_core_range(merge_core_ranges);

    tt::log_info("is_rmsnorm: {}", is_rmsnorm);
    tt::log_info("W: {}", W);
    tt::log_info("H: {}", H);
    tt::log_info("Wt: {}", Wt);
    tt::log_info("Ht: {}", Ht);
    tt::log_info("Grid size: {}x{}", grid_size.x, grid_size.y);
    tt::log_info("Cores per sequence: {}", cores_per_sequence);
    tt::log_info("Cores per dimension: {}", cores_per_dim);
    tt::log_info("Total cores used: {}", total_cores_used);
    tt::log_info("Dimension shard factor: {}", dim_shard_factor);
    tt::log_info("Tiles per dimension shard: {}", tiles_per_dim_shard);
    tt::log_info("Merge cores column: {}", merge_core_x);

    ////////////////////////////////////////////////////////////////////////////
    //                       Device Setup
    //////////////////////////////////////////////////////////////////////////
    IDevice* device = a.device();

    ////////////////////////////////////////////////////////////////////////////
    //                Circular Buffer Data Format Setup
    //////////////////////////////////////////////////////////////////////////
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    uint32_t block_size = 1;  // find_max_divisor(Wt, 8);

    tt::DataFormat in_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    tt::DataFormat out_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    uint32_t in_single_tile_size = tt::tt_metal::detail::TileSize(in_data_format);
    uint32_t out_single_tile_size = tt::tt_metal::detail::TileSize(out_data_format);
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);
    uint32_t bfloat16_tile_size = tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b);

    tt::log_info("in_data_format: {}", in_data_format);
    tt::log_info("out_data_format: {}", out_data_format);

    auto a_addr = a.buffer()->address();
    auto dst_addr = output.buffer()->address();

    uint32_t num_tiles = a.volume() / TILE_HW;

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    /*
    in0_cb: a
    in1_cb: 1 (reduction scalar)

    output CB is packed such that the first tile is for x**2 stats, second tile is for x stats
    in RMSNorm, only first tile has valid data.

    intermed0_cb: xˆ2
    shared_data_cb: Shared data buffer for dimension shards
    merge_data_cb: Data for merge computation
    out0_cb: [sum(xˆ2), sum(x)]  # For layernorm
    out0_cb: [sum(xˆ2)]  # RMSNorm

    */
    const uint32_t double_buffer_constant = 2;
    const uint32_t in0_tiles = tiles_per_dim_shard * double_buffer_constant;  // Only need tiles for one dimension shard
    const uint32_t in1_tiles = 1;                                             // reduce scalar

    const uint32_t intermed0_tiles = tiles_per_dim_shard * double_buffer_constant;  // xˆ2
    uint32_t results_per_dim = is_rmsnorm ? 1 : 2;
    const uint32_t shared_data_tiles = dim_shard_factor * results_per_dim;  // Shared space for dimension results
    const uint32_t merge_data_tiles = shared_data_tiles;                    // Same size as shared data
    uint32_t out0_tiles = results_per_dim;

    TT_ASSERT(
        tiles_per_dim_shard <= in0_tiles &&
        "W shard exceeds the maximum supported size of tile buffer (kernel limitation right now).");
    TT_ASSERT(
        in0_tiles % block_size == 0 &&
        "Size of buffer must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(
        intermed0_tiles % block_size == 0 &&
        "Size of buffer must be divisible by the size of block used by the reader and compute kernel.");

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    Program program = CreateProgram();

    // Create semaphores - one for each sequence group to signal completion of local processing
    std::vector<uint32_t> semaphore_ids(cores_per_sequence);
    // Create additional semaphores for merge to output communication
    std::vector<uint32_t> merge_output_semaphore_ids(cores_per_sequence);

    for (uint32_t seq_idx = 0; seq_idx < cores_per_sequence; seq_idx++) {
        // Create semaphores visible to both regular and merge cores
        CoreRangeSet all_cores = CoreRangeSet({CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1})});
        semaphore_ids[seq_idx] = CreateSemaphore(program, all_cores, INVALID);
        merge_output_semaphore_ids[seq_idx] = CreateSemaphore(program, all_cores, INVALID);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Reader kernels
    ////////////////////////////////////////////////////////////////////////////

    // Reader compile time args for local computation
    std::vector<uint32_t> reader_local_compile_time_args = {
        (std::uint32_t)is_dram(a),
        (std::uint32_t)block_size,
        (std::uint32_t)dim_shard_factor,
        (std::uint32_t)cores_per_dim};

    // Create reader config and set NOC
    tt::tt_metal::ReaderDataMovementConfig reader_local_config(reader_local_compile_time_args);
    reader_local_config.noc = NOC::NOC_0;  // Explicitly use NOC0

    auto reader_local_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/"
        "reader_local_layernorm_pre_allgather.cpp",
        regular_cores_range,  // ONLY on regular cores
        reader_local_config);

    // Reader compile time args for merge computation
    std::vector<uint32_t> reader_final_compile_time_args = {
        (std::uint32_t)semaphore_ids[0],  // Base semaphore ID
        (std::uint32_t)is_rmsnorm,
        (std::uint32_t)dim_shard_factor,
        (std::uint32_t)0,                  // noc_start_x
        (std::uint32_t)0,                  // noc_start_y
        (std::uint32_t)cores_per_dim - 1,  // noc_end_x (only regular cores)
        (std::uint32_t)grid_size.y - 1,    // noc_end_y
        (std::uint32_t)cores_per_dim       // num_dests
    };

    // Create reader config and set NOC
    tt::tt_metal::ReaderDataMovementConfig reader_final_config(reader_final_compile_time_args);
    reader_final_config.noc = NOC::NOC_0;  // Use NOC0 for reader_final

    auto reader_final_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/"
        "reader_final_layernorm_pre_allgather.cpp",
        merge_core_range,  // ONLY on merge cores
        reader_final_config);

    ////////////////////////////////////////////////////////////////////////////
    // Writer kernels
    ////////////////////////////////////////////////////////////////////////////

    // Writer compile time args for local computation
    std::vector<uint32_t> writer_local_compile_time_args = {
        (std::uint32_t)semaphore_ids[0],  // Base semaphore ID
        (std::uint32_t)is_rmsnorm,
        (std::uint32_t)dim_shard_factor,
        (std::uint32_t)cores_per_dim};

    // Create writer config and set NOC
    tt::tt_metal::WriterDataMovementConfig writer_local_config(writer_local_compile_time_args);
    writer_local_config.noc = NOC::NOC_0;  // Use NOC0 for writer_local - merge cores won't run this

    auto writer_local_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/"
        "writer_local_layernorm_pre_allgather.cpp",
        regular_cores_range,  // ONLY on regular cores
        writer_local_config);

    // Writer compile time args for final output
    std::vector<uint32_t> writer_final_compile_time_args = {(std::uint32_t)is_dram(output), (std::uint32_t)is_rmsnorm};

    // Create writer config and set NOC
    tt::tt_metal::WriterDataMovementConfig writer_final_config(writer_final_compile_time_args);
    writer_final_config.noc = NOC::NOC_0;  // Use NOC0 for writer_final

    auto writer_final_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/"
        "writer_final_layernorm_pre_allgather.cpp",
        merge_core_range,  // ONLY on merge cores
        writer_final_config);

    ////////////////////////////////////////////////////////////////////////////
    // Compute kernels
    ////////////////////////////////////////////////////////////////////////////

    // Local compute kernel args
    std::vector<uint32_t> compute_local_args = {tiles_per_dim_shard, block_size};

    std::map<string, string> compute_defines;
    if (is_rmsnorm) {
        compute_defines["RMSNORM"] = "1";
    }

    auto compute_local_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/"
        "layernorm_pre_allgather_local_sharded.cpp",
        regular_cores_range,  // ONLY on regular cores
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_local_args,
            .defines = compute_defines});

    // Merge compute kernel args
    std::vector<uint32_t> compute_merge_args = {dim_shard_factor, results_per_dim};

    auto compute_merge_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/"
        "layernorm_pre_allgather_merge_sharded.cpp",
        merge_core_range,  // ONLY on merge cores
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_merge_args,
            .defines = compute_defines});

    ////////////////////////////////////////////////////////////////////////////
    // Create circular buffers
    ////////////////////////////////////////////////////////////////////////////

    // Create a CoreRangeSet containing all used cores for buffers needed by both core groups
    CoreRangeSet all_used_cores = CoreRangeSet({CoreRange({0, 0}, {merge_core_x, cores_per_sequence - 1})});

    // c_in0 -> a (input data) - needed by regular cores only
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(in0_tiles * in_single_tile_size, {{tt::CBIndex::c_0, in_data_format}})
            .set_page_size(tt::CBIndex::c_0, in_single_tile_size);
    CreateCircularBuffer(program, regular_cores_range, cb_src0_config);

    // c_in1 -> reduce scalar - needed by regular cores only
    CircularBufferConfig cb_reduce_config =
        CircularBufferConfig(in1_tiles * bfloat16_tile_size, {{tt::CBIndex::c_1, cb_data_format}})
            .set_page_size(tt::CBIndex::c_1, bfloat16_tile_size);
    CreateCircularBuffer(program, regular_cores_range, cb_reduce_config);

    // c_intermed0 -> xˆ2 (intermediate computation) - needed by regular cores only
    CircularBufferConfig cb_intermed0_config =
        CircularBufferConfig(intermed0_tiles * single_tile_size, {{tt::CBIndex::c_6, cb_data_format}})
            .set_page_size(tt::CBIndex::c_6, single_tile_size);
    CreateCircularBuffer(program, regular_cores_range, cb_intermed0_config);

    // Local compute partial results - needed by regular cores only
    CircularBufferConfig cb_partial_config =
        CircularBufferConfig(results_per_dim * single_tile_size, {{tt::CBIndex::c_7, cb_data_format}})
            .set_page_size(tt::CBIndex::c_7, single_tile_size);
    CreateCircularBuffer(program, regular_cores_range, cb_partial_config);

    // Buffer for merge computation input - needed by merge cores only
    // We need separate space for each sequence's merge data
    uint32_t total_merge_tiles = cores_per_sequence * merge_data_tiles;
    CircularBufferConfig cb_merge_data_config =
        CircularBufferConfig(total_merge_tiles * single_tile_size, {{tt::CBIndex::c_9, cb_data_format}})
            .set_page_size(tt::CBIndex::c_9, single_tile_size);
    CreateCircularBuffer(program, merge_core_range, cb_merge_data_config);

    // Output buffer - needed by both regular cores (for intermediate results) and merge cores (for final output)
    CircularBufferConfig cb_out0_config =
        CircularBufferConfig(out0_tiles * out_single_tile_size, {{tt::CBIndex::c_14, out_data_format}})
            .set_page_size(tt::CBIndex::c_14, out_single_tile_size);
    CreateCircularBuffer(program, all_used_cores, cb_out0_config);

    // Log all circular buffers
    auto cbs = program.circular_buffers();
    for (const auto& cb : cbs) {
        for (const auto& index : cb->buffer_indices()) {
            tt::log_info("cb_id {}", index);
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    // Setup runtime arguments
    ////////////////////////////////////////////////////////////////////////////

    float winv = 1.0f;
    auto bfloat_winv_value = bfloat16(winv);
    uint32_t packed_winv_value = pack_two_bfloat16_into_uint32({bfloat_winv_value, bfloat_winv_value});

    // Set up runtime arguments for each core
    for (uint32_t seq_idx = 0; seq_idx < cores_per_sequence; seq_idx++) {
        // Get the semaphore ID for this sequence
        uint32_t seq_semaphore_id = semaphore_ids[seq_idx];
        uint32_t merge_output_semaphore_id = merge_output_semaphore_ids[seq_idx];

        // Get the merge core for this sequence
        CoreCoord merge_core = merge_cores[seq_idx];

        // Calculate tiles per sequence
        uint32_t tiles_per_seq = (Ht + cores_per_sequence - 1) / cores_per_sequence;
        uint32_t seq_start_tile = seq_idx * tiles_per_seq;
        uint32_t seq_end_tile = std::min(seq_start_tile + tiles_per_seq, Ht);
        uint32_t seq_tiles = seq_end_tile - seq_start_tile;

        if (seq_tiles == 0) {
            continue;  // Skip if no tiles assigned to this sequence
        }

        // Compute tiles processed for this sequence
        uint32_t tiles_processed = seq_tiles * NC;

        // Prepare normalization factor (1/N)
        float inv_N = 1.0f / static_cast<float>(W * H);
        auto bfloat_inv_N = bfloat16(inv_N);
        uint32_t packed_inv_N = pack_two_bfloat16_into_uint32({bfloat_inv_N, bfloat_inv_N});

        // Setup reader_final with the normalization factor
        SetRuntimeArgs(
            program,
            reader_final_kernel_id,
            merge_core,
            {
                packed_inv_N  // Pass normalization factor to the reader, which will setup the reduce buffer
            });

        // Merge compute arguments - no longer needs the normalization factor
        SetRuntimeArgs(program, compute_merge_kernel_id, merge_core, {});

        // Final writer arguments - write directly to DRAM from merge core
        uint32_t output_tiles = tiles_processed * tile_cols_per_device;
        uint32_t output_offset = seq_idx * output_tiles * out_single_tile_size;

        SetRuntimeArgs(
            program,
            writer_final_kernel_id,
            merge_core,
            {
                dst_addr + output_offset,  // Output address with offset
                output_tiles,
                results_per_dim,           // Tiles per batch
                merge_output_semaphore_id  // Use the semaphore for sync
            });

        // Process all regular cores for this sequence
        for (uint32_t dim_idx = 0; dim_idx < cores_per_dim; dim_idx++) {
            CoreCoord core = {dim_idx, seq_idx};

            // Calculate tile assignments
            uint32_t dim_start_tile = dim_idx * tiles_per_dim_shard;
            uint32_t seq_tile_offset = seq_start_tile * Wt;
            uint32_t start_tile = seq_tile_offset + dim_start_tile;

            tt::log_info("Regular core ({},{}) assigned seq_idx={}, dim_idx={}", core.x, core.y, seq_idx, dim_idx);
            tt::log_info(
                "  Processing seq tiles [{}, {}), dim tiles [{}, {})",
                seq_start_tile,
                seq_end_tile,
                dim_start_tile,
                dim_start_tile + tiles_per_dim_shard);

            // Local reader arguments
            SetRuntimeArgs(
                program,
                reader_local_kernel_id,
                core,
                {a_addr, tiles_processed, tiles_per_dim_shard, start_tile, packed_winv_value});

            // Local compute arguments
            SetRuntimeArgs(program, compute_local_kernel_id, core, {tiles_processed});

            // Calculate merge buffer offset for this sequence
            uint32_t seq_merge_buffer_offset = seq_idx * merge_data_tiles * single_tile_size;

            // Local writer arguments - all cores write to merge core
            SetRuntimeArgs(
                program, writer_local_kernel_id, core, {merge_core.x, merge_core.y, dim_idx, seq_merge_buffer_offset});
        }
    }

    // Return program with callback for runtime argument override
    auto override_runtime_arguments_callback =
        [reader_local_kernel_id,
         reader_final_kernel_id,
         writer_final_kernel_id,
         cores_per_sequence,
         cores_per_dim,
         merge_core_x](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto& input_tensor = input_tensors.at(0);
            const auto input_addr = input_tensor.buffer()->address();

            const auto& output_tensor = output_tensors.at(0);
            const auto output_addr = output_tensor.buffer()->address();

            // Update regular cores' read addresses
            for (uint32_t seq_idx = 0; seq_idx < cores_per_sequence; seq_idx++) {
                for (uint32_t dim_idx = 0; dim_idx < cores_per_dim; dim_idx++) {
                    CoreCoord core(dim_idx, seq_idx);

                    // Update reader args for regular cores
                    try {
                        auto& reader_args = GetRuntimeArgs(program, reader_local_kernel_id, core);
                        reader_args[0] = input_addr;  // Update input address
                    } catch (const std::exception&) {
                        // Skip if args not found for this core
                    }
                }

                // Update merge core write addresses
                CoreCoord merge_core(merge_core_x, seq_idx);
                try {
                    auto& writer_args = GetRuntimeArgs(program, writer_final_kernel_id, merge_core);
                    uint32_t tile_cols_per_device = writer_args.size() > 2 ? writer_args[2] : 1;
                    uint32_t tiles_processed = writer_args[1];
                    uint32_t output_offset =
                        seq_idx * tiles_processed * tile_cols_per_device *
                        tt::tt_metal::detail::TileSize(
                            tt::tt_metal::datatype_to_dataformat_converter(output_tensor.get_dtype()));

                    // Original output address plus sequence-specific offset
                    writer_args[0] = output_addr + output_offset;
                } catch (const std::exception&) {
                    // Skip if args not found for this core
                }
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::normalization
