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

namespace {                         // Anonymous namespace for helper functions
namespace CMAKE_UNIQUE_NAMESPACE {  // To avoid ODR violations if this file is included multiple times
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
    uint32_data = (uint32_data >> 16);
    return (uint16_t)uint32_data;
}
inline uint32_t pack_two_bfloat16_into_uint32(std::pair<uint16_t, uint16_t> two_bfloats) {
    return (uint32_t)two_bfloats.first | ((uint32_t)two_bfloats.second << 16);
}
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

// Original non-sharded version (interleaved input)
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
    const auto& a_dtype = a.get_dtype();
    const uint32_t Wt = W / TILE_WIDTH;
    const uint32_t Ht = H / TILE_HEIGHT;
    const uint32_t tile_cols_per_device = is_rmsnorm ? 1 : 2;
    uint32_t num_tile_rows = NC * Ht;

    IDevice* device = a.device();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);
    uint32_t block_size = 1;
    uint32_t writer_block_size = 1;
    tt::DataFormat in_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    tt::DataFormat out_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    uint32_t in_single_tile_size = tt::tt_metal::detail::TileSize(in_data_format);
    uint32_t out_single_tile_size = tt::tt_metal::detail::TileSize(out_data_format);
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);
    uint32_t bfloat16_tile_size = tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b);
    auto a_addr = a.buffer()->address();
    auto dst_addr = output.buffer()->address();
    uint32_t num_tiles = a.volume() / TILE_HW;

    const uint32_t double_buffer_constant = 2;
    const uint32_t in0_tiles = Wt * double_buffer_constant;
    const uint32_t in1_tiles = 1;
    const uint32_t intermed0_tiles = Wt * double_buffer_constant;
    uint32_t out0_tiles = is_rmsnorm ? 1 : 2;

    auto grid_size = device->compute_with_storage_grid_size();
    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_tile_rows_per_core_group_1,
         num_tile_rows_per_core_group_2] = tt::tt_metal::split_work_to_cores(grid_size, num_tile_rows, true);

    Program program = CreateProgram();
    std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t)is_dram(a), (std::uint32_t)block_size};
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)is_dram(output), (std::uint32_t)writer_block_size};
    std::map<string, string> compute_defines_std;
    if (is_rmsnorm) {
        compute_defines_std["RMSNORM"] = "1";
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
    std::vector<uint32_t> compute_args_std = {Wt, block_size};
    auto compute_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/"
        "layernorm_pre_allgather.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_args_std,
            .defines = compute_defines_std});
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(in0_tiles * in_single_tile_size, {{tt::CBIndex::c_0, in_data_format}})
            .set_page_size(tt::CBIndex::c_0, in_single_tile_size);
    CreateCircularBuffer(program, all_cores, cb_src0_config);
    CircularBufferConfig cb_reduce_config =
        CircularBufferConfig(in1_tiles * bfloat16_tile_size, {{tt::CBIndex::c_1, cb_data_format}})
            .set_page_size(tt::CBIndex::c_1, bfloat16_tile_size);
    CreateCircularBuffer(program, all_cores, cb_reduce_config);
    CircularBufferConfig cb_intermed0_config =
        CircularBufferConfig(intermed0_tiles * single_tile_size, {{tt::CBIndex::c_6, cb_data_format}})
            .set_page_size(tt::CBIndex::c_6, single_tile_size);
    CreateCircularBuffer(program, all_cores, cb_intermed0_config);
    CircularBufferConfig cb_out0_config =
        CircularBufferConfig(out0_tiles * out_single_tile_size, {{tt::CBIndex::c_14, out_data_format}})
            .set_page_size(tt::CBIndex::c_14, out_single_tile_size);
    CreateCircularBuffer(program, all_cores, cb_out0_config);

    // Log all circular buffers with program.circular_buffers(), which returns
    // std::vector<std::shared_ptr<CircularBuffer>>
    for (const auto& cb : program.circular_buffers()) {
        for (const auto index : cb->buffer_indices()) {
            tt::log_debug("cb_id {}", index);
            tt::log_debug("page_size: {}", cb->page_size(index));
            tt::log_debug("num_pages: {}", cb->num_pages(index));
            tt::log_debug("data_format: {}", cb->data_format(index));
        }
    }

    uint32_t curr_row = 0;
    float winv = 1.0f;
    auto bfloat_winv_value = bfloat16(winv);
    uint32_t packed_winv_value = pack_two_bfloat16_into_uint32({bfloat_winv_value, bfloat_winv_value});
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};
        uint32_t num_tile_rows_per_core =
            core_group_1.contains(core) ? num_tile_rows_per_core_group_1 : num_tile_rows_per_core_group_2;
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
            const auto input_addr = input_tensors.at(0).buffer()->address();
            const auto output_addr = output_tensors.at(0).buffer()->address();
            auto& reader_runtime_args_by_core = GetRuntimeArgs(program, reader_kernel_id);
            auto& writer_runtime_args_by_core = GetRuntimeArgs(program, writer_kernel_id);
            for (uint32_t i = 0; i < num_cores; ++i) {
                const CoreCoord core = {i % grid_size.x, i / grid_size.x};
                reader_runtime_args_by_core.at(core.x).at(core.y)[0] = input_addr;
                writer_runtime_args_by_core.at(core.x).at(core.y)[0] = output_addr;
            }
        };
    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

// This is the sharded version that we are debugging
operation::ProgramWithCallbacks layernorm_pre_allgather_multi_core_sharded(
    const Tensor& a,
    Tensor& output,
    LayerNormDistributedType norm_type,
    DeviceComputeKernelConfig compute_kernel_config) {
    tt::log_info("Starting layernorm_pre_allgather_multi_core_sharded (interleaved input path)");
    using namespace CMAKE_UNIQUE_NAMESPACE;
    Program program{};

    const bool is_rmsnorm = norm_type == LayerNormDistributedType::RMSNORM;
    const auto shape = a.get_padded_shape();
    const uint32_t W = shape[-1], H = shape[-2];
    const uint32_t Wt = W / TILE_WIDTH;
    const uint32_t Ht = H / TILE_HEIGHT;
    const uint32_t num_tile_rows = (a.volume() / (H * W)) * Ht;  // Total rows of tiles (NC * Ht)

    auto* device = a.device();
    auto grid_size = device->compute_with_storage_grid_size();

    uint32_t max_cores_y = grid_size.y;
    uint32_t max_cores_x = grid_size.x > 1 ? grid_size.x - 1 : 1;
    uint32_t cores_per_sequence = std::min(max_cores_y, num_tile_rows);
    while (num_tile_rows % cores_per_sequence != 0 && cores_per_sequence > 1) {
        cores_per_sequence--;
    }
    uint32_t rows_per_core = num_tile_rows / cores_per_sequence;
    uint32_t cores_per_dim = std::min(max_cores_x, Wt);
    while (Wt % cores_per_dim != 0 && cores_per_dim > 1) {
        cores_per_dim--;
    }
    uint32_t tiles_per_dim_shard = Wt / cores_per_dim;

    CoreRange regular_cores_range({0, 0}, {cores_per_dim - 1, cores_per_sequence - 1});
    std::vector<CoreCoord> merge_cores;
    std::vector<CoreRange> merge_core_ranges_vec;  // Renamed to avoid conflict
    uint32_t merge_core_x_coord = cores_per_dim;
    for (uint32_t seq_idx = 0; seq_idx < cores_per_sequence; ++seq_idx) {
        CoreCoord merge_core = {merge_core_x_coord, seq_idx};
        merge_cores.push_back(merge_core);
        merge_core_ranges_vec.push_back(CoreRange(merge_core, merge_core));
    }
    CoreRangeSet merge_core_range(merge_core_ranges_vec);

    tt::log_info(tt::LogOp, "RMSNorm MultiCoreSharded (Interleaved Input Path):");
    tt::log_info(tt::LogOp, "  Grid size: {}x{}", grid_size.x, grid_size.y);
    tt::log_info(tt::LogOp, "  Cores per sequence: {}", cores_per_sequence);
    tt::log_info(tt::LogOp, "  Cores per dim: {}", cores_per_dim);
    tt::log_info(tt::LogOp, "  Rows per core: {}", rows_per_core);
    tt::log_info(tt::LogOp, "  Tiles per dim shard: {}", tiles_per_dim_shard);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);
    tt::DataFormat out_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t out_single_tile_size = tt::tt_metal::detail::TileSize(out_data_format);
    uint32_t results_per_dim = is_rmsnorm ? 1 : 2;
    uint32_t total_tiles_per_seq_in_merge_buffer = cores_per_dim * results_per_dim;

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_tiles_local = tiles_per_dim_shard * 2;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles_local * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    CreateCircularBuffer(program, regular_cores_range, cb_src0_config);
    uint32_t reduce_cb_index = tt::CBIndex::c_1;
    CircularBufferConfig cb_reduce_config =
        CircularBufferConfig(1 * single_tile_size, {{reduce_cb_index, cb_data_format}})
            .set_page_size(reduce_cb_index, single_tile_size);
    std::set<CoreRange> all_cores_for_reduce_set;
    all_cores_for_reduce_set.insert(regular_cores_range);
    for (const auto& range : merge_core_range.ranges()) {
        all_cores_for_reduce_set.insert(range);
    }
    CoreRangeSet all_used_cores_for_sem_and_reduce(all_cores_for_reduce_set);
    CreateCircularBuffer(program, all_used_cores_for_sem_and_reduce, cb_reduce_config);
    uint32_t x2_cb_index = tt::CBIndex::c_6;
    CircularBufferConfig cb_x2_config =
        CircularBufferConfig(num_input_tiles_local * single_tile_size, {{x2_cb_index, cb_data_format}})
            .set_page_size(x2_cb_index, single_tile_size);
    CreateCircularBuffer(program, regular_cores_range, cb_x2_config);
    uint32_t partial_cb_index = tt::CBIndex::c_7;
    CircularBufferConfig cb_partial_config =
        CircularBufferConfig(results_per_dim * single_tile_size, {{partial_cb_index, cb_data_format}})
            .set_page_size(partial_cb_index, single_tile_size);
    CreateCircularBuffer(program, regular_cores_range, cb_partial_config);
    uint32_t merge_data_cb_index = tt::CBIndex::c_9;
    CircularBufferConfig cb_merge_data_config =
        CircularBufferConfig(
            total_tiles_per_seq_in_merge_buffer * single_tile_size, {{merge_data_cb_index, cb_data_format}})
            .set_page_size(merge_data_cb_index, single_tile_size);
    auto cb_merge_data_in_handle =
        CreateCircularBuffer(program, all_used_cores_for_sem_and_reduce, cb_merge_data_config);
    uint32_t dst_cb_index = tt::CBIndex::c_14;
    CircularBufferConfig cb_dst_config =
        CircularBufferConfig(1 * out_single_tile_size, {{dst_cb_index, out_data_format}})
            .set_page_size(dst_cb_index, out_single_tile_size);
    CreateCircularBuffer(program, merge_core_range, cb_dst_config);

    std::set<CoreRange> all_cores_set_for_sem;
    all_cores_set_for_sem.insert(regular_cores_range);
    for (const auto& range : merge_core_range.ranges()) {
        all_cores_set_for_sem.insert(range);
    }
    CoreRangeSet all_used_cores_for_sem(all_cores_set_for_sem);

    auto receiver_sem_addr = tt::tt_metal::CreateSemaphore(program, all_used_cores_for_sem, 0);
    auto sender_sem_addr = tt::tt_metal::CreateSemaphore(program, all_used_cores_for_sem, 0);
    log_debug(tt::LogOp, "Host: receiver_sem L1 addr: 0x{:x} (init_val=0)", receiver_sem_addr);
    log_debug(tt::LogOp, "Host: sender_sem L1 addr: 0x{:x} (init_val=0)", sender_sem_addr);

    CoreRange writer_bbox = regular_cores_range;
    CoreCoord writer_noc_start = device->worker_core_from_logical_core(writer_bbox.start_coord);
    CoreCoord writer_noc_end = device->worker_core_from_logical_core(writer_bbox.end_coord);

    auto input_buffer = a.buffer();
    auto output_buffer = output.buffer();
    float reduce_scaler_val = 1.0f;  // / W;
    auto bfloat_scaler = bfloat16(reduce_scaler_val);
    uint32_t packed_reduce_scaler = pack_two_bfloat16_into_uint32({bfloat_scaler, bfloat_scaler});

    std::map<string, string> compute_defines;
    if (is_rmsnorm) {
        compute_defines["RMSNORM"] = "1";
    }

    std::vector<uint32_t> reader_local_compile_args = {1};  // DRAM input
    auto reader_local_kernels = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/"
        "reader_local_layernorm_pre_allgather.cpp",
        regular_cores_range,
        tt::tt_metal::ReaderDataMovementConfig(reader_local_compile_args, compute_defines));

    // Create writer_local_kernels BEFORE compute_local_kernels to match TopK pattern
    std::vector<uint32_t> writer_local_compile_args = {
        (is_rmsnorm ? 1u : 0u), cores_per_dim, cores_per_dim, receiver_sem_addr, sender_sem_addr, merge_data_cb_index};
    auto writer_local_kernels = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/"
        "writer_local_layernorm_pre_allgather.cpp",
        regular_cores_range,
        tt::tt_metal::WriterDataMovementConfig(writer_local_compile_args, compute_defines));

    constexpr uint32_t compute_local_block_size = 1;
    std::vector<uint32_t> compute_local_compile_args = {tiles_per_dim_shard, compute_local_block_size};
    auto compute_local_kernels = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/"
        "layernorm_pre_allgather_local_sharded.cpp",
        regular_cores_range,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_local_compile_args,
            .defines = compute_defines});

    std::vector<uint32_t> reader_final_compile_args = {
        (is_rmsnorm ? 1u : 0u),
        cores_per_dim,
        receiver_sem_addr,
        sender_sem_addr,
        writer_noc_start.x,
        writer_noc_start.y,
        writer_noc_end.x,
        writer_noc_end.y,
        (uint32_t)regular_cores_range.size()};
    auto reader_final_kernels = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/"
        "reader_final_layernorm_pre_allgather.cpp",
        merge_core_range,
        tt::tt_metal::ReaderDataMovementConfig(reader_final_compile_args, compute_defines));
    std::vector<uint32_t> compute_merge_compile_args = {cores_per_dim, total_tiles_per_seq_in_merge_buffer};
    auto compute_merge_kernels = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/"
        "layernorm_pre_allgather_merge_sharded.cpp",
        merge_core_range,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_merge_compile_args,
            .defines = compute_defines});
    std::vector<uint32_t> writer_final_compile_args = {1, (is_rmsnorm ? 1u : 0u)};
    auto writer_final_kernels = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/"
        "writer_final_layernorm_pre_allgather.cpp",
        merge_core_range,
        tt::tt_metal::WriterDataMovementConfig(writer_final_compile_args, compute_defines));

    for (uint32_t seq_idx = 0; seq_idx < cores_per_sequence; ++seq_idx) {
        CoreCoord current_merge_core_logical = merge_cores[seq_idx];
        CoreCoord current_merge_core_physical = device->worker_core_from_logical_core(current_merge_core_logical);
        std::vector<uint32_t> reader_final_args = {packed_reduce_scaler};
        SetRuntimeArgs(program, reader_final_kernels, current_merge_core_logical, reader_final_args);
        uint32_t seq_row_start_tile_id = seq_idx * rows_per_core * Wt;
        for (uint32_t dim_idx = 0; dim_idx < cores_per_dim; ++dim_idx) {
            CoreCoord local_core_logical = {dim_idx, seq_idx};
            uint32_t Wt_offset = dim_idx * tiles_per_dim_shard;
            std::vector<uint32_t> reader_local_args = {
                input_buffer->address(), rows_per_core, Wt, Wt_offset, tiles_per_dim_shard, seq_row_start_tile_id};
            SetRuntimeArgs(program, reader_local_kernels, local_core_logical, reader_local_args);
            std::vector<uint32_t> compute_local_args = {rows_per_core};
            SetRuntimeArgs(program, compute_local_kernels, local_core_logical, compute_local_args);
            std::vector<uint32_t> writer_local_args = {
                current_merge_core_physical.x, current_merge_core_physical.y, dim_idx};
            SetRuntimeArgs(program, writer_local_kernels, local_core_logical, writer_local_args);
        }
        std::vector<uint32_t> compute_merge_args = {};
        SetRuntimeArgs(program, compute_merge_kernels, current_merge_core_logical, compute_merge_args);
        std::vector<uint32_t> writer_final_args = {output_buffer->address(), rows_per_core};
        SetRuntimeArgs(program, writer_final_kernels, current_merge_core_logical, writer_final_args);
    }

    auto override_runtime_arguments_callback =
        [reader_local_kernels,
         writer_final_kernels,
         cores_per_sequence,
         cores_per_dim,
         rows_per_core,
         Wt,
         results_per_dim,
         merge_cores](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            auto input_buffer = input_tensors.at(0).buffer();
            auto output_buffer = output_tensors.at(0).buffer();
            for (uint32_t seq_idx = 0; seq_idx < cores_per_sequence; ++seq_idx) {
                CoreCoord current_merge_core_logical = merge_cores[seq_idx];
                uint32_t seq_row_start_tile_id = seq_idx * rows_per_core * Wt;
                for (uint32_t dim_idx = 0; dim_idx < cores_per_dim; ++dim_idx) {
                    CoreCoord local_core_logical = {dim_idx, seq_idx};
                    auto& reader_args = GetRuntimeArgs(program, reader_local_kernels, local_core_logical);
                    reader_args[0] = input_buffer->address();
                    reader_args[5] = seq_row_start_tile_id;
                }
                auto& writer_args = GetRuntimeArgs(program, writer_final_kernels, current_merge_core_logical);
                writer_args[0] = output_buffer->address();
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::normalization
