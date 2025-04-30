// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/fabric.hpp>
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_silu/device/all_gather_silu_op.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include "cpp/ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "cpp/ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"

#include "cpp/ttnn/operations/ccl/common/uops/command_lowering.hpp"

#include "cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "cpp/ttnn/operations/ccl/common/host/command_backend_runtime_args_overrider.hpp"
#include <sstream>
#include <type_traits>
#include <ranges>
#include <optional>

using namespace tt::constants;

namespace ttnn {

using namespace ccl;
struct llama_config {
    CoreRange sem_drain_core = CoreRange({3, 0}, {3, 0});
    std::array<CoreRange, 6> semaphore_mcast_ranges = {
        CoreRange({1, 0}, {2, 0}),   // 2
        CoreRange({1, 9}, {2, 9}),   // 2
        CoreRange({5, 9}, {6, 9}),   // 2
        CoreRange({1, 4}, {2, 5}),   // 4
        CoreRange({5, 0}, {6, 2}),   // 6
        CoreRange({5, 4}, {6, 7})};  // 8

    uint32_t num_semaphore_ranges = 6;
};
tt::tt_metal::operation::ProgramWithCallbacks all_gather_silu_llama_sharded(
    const Tensor& input_tensor,
    const Tensor& buffer_tensor,
    IDevice* sender_device,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    Tensor& output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology,
    const GlobalSemaphore& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    tt::tt_metal::Program program{};

    IDevice* mesh_device = input_tensor.mesh_device();
    if (!mesh_device) {
        mesh_device = input_tensor.device();
    }
    auto buffer_address = buffer_tensor.buffer()->address();
    const bool enable_async_output_tensor = false;
    const bool enable_persistent_fabric_mode = true;

    bool is_first_chip = ring_index == 0;
    bool is_last_chip = ring_index == ring_size - 1;
    log_trace(
        tt::LogOp,
        "DEBUG: device: {}, is_first_chip: {}, is_last_chip: {}",
        sender_device->id(),
        is_first_chip,
        is_last_chip);

    // Get OP Config, topology config
    std::vector<Tensor> input_tensors = {input_tensor};
    std::vector<Tensor> buffer_tensors = {buffer_tensor};
    std::vector<Tensor> output_tensors = {output_tensor};
    const auto& op_config = ttnn::ccl::CCLOpConfig(input_tensors, buffer_tensors, topology);
    auto [num_targets_forward, num_targets_backward, dynamic_alternate] =
        ccl::get_forward_backward_configuration(ring_size, ring_index, topology);
    llama_config llama_configuration;

    // Get worker cores, assuming 1 worker per link
    uint32_t num_workers_per_link = 1;
    // const auto [sender_worker_core_range, sender_worker_cores] =
    //     choose_worker_cores(num_links, num_workers_per_link, enable_persistent_fabric_mode, mesh_device,
    //     sub_device_id);
    auto sender_worker_core_range = CoreRangeSet(CoreRange({1, 0}, {num_links, 0}));
    auto sender_worker_cores = corerange_to_cores(sender_worker_core_range, num_links, true);

    // Tensor Info
    const auto input_tensor_num_pages = input_tensor.buffer()->num_pages();
    const auto input_tensor_cores = input_tensor.memory_config().shard_spec->grid;
    const auto input_tensor_shard_shape = input_tensor.memory_config().shard_spec->shape;
    const auto input_tensor_shard_num_pages = input_tensor_shard_shape[0] * input_tensor_shard_shape[1] / TILE_HW;
    const auto output_tensor_cores = buffer_tensor.memory_config().shard_spec->grid;
    const auto output_tensor_shard_shape = buffer_tensor.memory_config().shard_spec->shape;
    const auto output_tensor_shard_num_pages = output_tensor_shard_shape[0] * output_tensor_shard_shape[1] / TILE_HW;

    // Reshard info
    const auto& tile_shape = input_tensor.get_tensor_spec().tile().get_tile_shape();
    auto reshard_tensor_shard_shape = output_tensor.memory_config().shard_spec->shape;
    uint32_t reshard_out_tiles_per_core = reshard_tensor_shard_shape[1] / tile_shape[1];
    uint32_t reshard_in_tiles_per_core = output_tensor_shard_shape[1] / tile_shape[1];
    // reshard kernel
    auto reshard_worker_grid = output_tensor.shard_spec().value().grid;
    auto reshard_cores = corerange_to_cores(reshard_worker_grid, reshard_worker_grid.num_cores(), true);

    tt::log_debug(tt::LogOp, "input_tensor_num_pages: {}", input_tensor_num_pages);
    tt::log_debug(tt::LogOp, "input_tensor_cores: {}", input_tensor_cores);
    tt::log_debug(tt::LogOp, "input_tensor_shard_shape: {}", input_tensor_shard_shape);
    tt::log_debug(tt::LogOp, "input_tensor_shard_num_pages: {}", input_tensor_shard_num_pages);
    tt::log_debug(tt::LogOp, "output_tensor_cores: {}", output_tensor_cores);
    tt::log_debug(tt::LogOp, "output_tensor_shard_shape: {}", output_tensor_shard_shape);
    tt::log_debug(tt::LogOp, "output_tensor_shard_num_pages: {}", output_tensor_shard_num_pages);

    // L1 Scratch CB Creation
    const size_t packet_size_bytes = tt::tt_fabric::get_1d_fabric_config().channel_buffer_size_bytes;
    uint32_t l1_scratch_cb_page_size_bytes = op_config.get_page_size();
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t cb_num_pages =
        input_tensor_num_pages / num_links +
        1;  // We are dealing with small shapes, so assuming all pages for a worker can be fit into the CB

    uint32_t src0_cb_index = tt::CB::c_in0;
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{src0_cb_index, df}})
            .set_page_size(src0_cb_index, l1_scratch_cb_page_size_bytes);
    tt::tt_metal::CBHandle cb_src0_workers = CreateCircularBuffer(program, sender_worker_core_range, cb_src0_config);
    // Set aside a buffer we can use for storing packet headers in (particularly for atomic incs)
    const auto reserved_packet_header_CB_index = tt::CB::c_in1;
    static constexpr auto num_packet_headers_storable = 8;
    static constexpr auto packet_header_size_bytes = sizeof(tt::tt_fabric::PacketHeader);
    tt::tt_metal::CircularBufferConfig cb_reserved_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * 2,
            {{reserved_packet_header_CB_index, tt::DataFormat::RawUInt32}})
            .set_page_size(reserved_packet_header_CB_index, packet_header_size_bytes);
    auto reserved_packet_header_CB_handle =
        CreateCircularBuffer(program, sender_worker_core_range, cb_reserved_packet_header_config);

    uint32_t single_tile_size =
        tt::tt_metal::detail::TileSize(tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype()));
    uint32_t q_output_cb_index = tt::CBIndex::c_16;
    tt::tt_metal::CircularBufferConfig cb_q_output_config =
        tt::tt_metal::CircularBufferConfig(reshard_out_tiles_per_core * single_tile_size, {{q_output_cb_index, df}})
            .set_page_size(q_output_cb_index, single_tile_size)
            .set_globally_allocated_address(*output_tensor.buffer());
    auto cb_q_output = tt::tt_metal::CreateCircularBuffer(program, reshard_worker_grid, cb_q_output_config);

    // KERNEL CREATION
    // Reader
    std::vector<CoreRange> sem_cores_vector;
    sem_cores_vector.push_back(llama_configuration.sem_drain_core);
    for (auto cr : llama_configuration.semaphore_mcast_ranges) {
        sem_cores_vector.push_back(cr);
    }
    const auto& sem_cores_updated = CoreRangeSet(sem_cores_vector);
    uint32_t reshard_semaphore_id = tt::tt_metal::CreateSemaphore(program, sem_cores_updated, 0);

    auto reader_kernel_config = tt::tt_metal::ReaderDataMovementConfig{};
    reader_kernel_config.compile_args = {
        ring_index,                 // my_chip_id
        src0_cb_index,              // cb0_id
        op_config.get_page_size(),  // tensor0_page_size
    };
    log_trace(tt::LogOp, "Reader Compile Args:");
    for (const auto& arg : reader_kernel_config.compile_args) {
        log_trace(tt::LogOp, "\t{}", arg);
    }
    auto reader_worker_only_range = CoreRangeSet(CoreRange({3, 0}, {num_links, 0}));
    auto reader_worker_only_cores = corerange_to_cores(reader_worker_only_range, num_links - 2, true);

    auto worker_sender_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_silu/device/kernels/llama_all_gather_silu_reader.cpp",
        reader_worker_only_range,
        reader_kernel_config);

    // Writer
    auto writer_kernel_config = tt::tt_metal::WriterDataMovementConfig{};
    writer_kernel_config.compile_args = {
        ring_index,                       // my_chip_id
        reserved_packet_header_CB_index,  // reserved_packet_header_cb_id
        num_packet_headers_storable,      // num_packet_headers_storable
        src0_cb_index,                    // cb0_id
        num_pages_per_packet,             // packet_size_in_pages
        op_config.get_page_size(),        // tensor0_page_size
        num_targets_forward,              // num_targets_forward_direction
        num_targets_backward,             // num_targets_backward_direction
        dynamic_alternate,                // dynamic_alternate
        llama_configuration.num_semaphore_ranges,
    };
    log_trace(tt::LogOp, "Writer Compile Args:");
    for (const auto& arg : writer_kernel_config.compile_args) {
        log_trace(tt::LogOp, "\t{}", arg);
    }
    auto worker_sender_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_silu/device/kernels/llama_all_gather_silu_writer.cpp",
        sender_worker_core_range,
        writer_kernel_config);

    // Kernel Runtime Args
    CoreCoord drain_sync_core = mesh_device->worker_core_from_logical_core(
        sender_worker_cores[2]);  // the first worker of each chip is the drain sync core, which contains the output
                                  // ready semaphore
    auto input_cores_vec = corerange_to_cores(input_tensor_cores, std::nullopt, true);
    auto output_cores_vec = corerange_to_cores(output_tensor_cores, std::nullopt, true);
    auto cores_per_device = 8;
    std::vector<uint32_t> start_core_indices = {0, 7, 15, 22};
    uint32_t start_core_index_for_device = start_core_indices[ring_index];
    uint32_t end_core_index_for_device = start_core_index_for_device + cores_per_device;

    auto output_cores_this_device = std::vector<CoreCoord>(
        output_cores_vec.begin() + start_core_index_for_device, output_cores_vec.begin() + end_core_index_for_device);

    log_trace(tt::LogOp, "output_cores_this_device: {}", output_cores_this_device);
    auto reshard_kernel_config = tt::tt_metal::ReaderDataMovementConfig{};
    reshard_kernel_config.compile_args = {q_output_cb_index, single_tile_size};

    std::vector<CoreRange> reshard_worker_only_vector;
    for (auto cr : reshard_worker_grid.ranges()) {
        bool common_core = (cr.start_coord.x == 1 || cr.start_coord.x == 2) && cr.start_coord.y == 0;
        if (!common_core) {
            reshard_worker_only_vector.push_back(cr);
        }
    }
    for (auto cr : reshard_worker_only_vector) {
        printf(
            "reshard only core: %zu %zu %zu %zu\n", cr.start_coord.x, cr.start_coord.y, cr.end_coord.x, cr.end_coord.y);
    }
    auto reshard_worker_only_grid = CoreRangeSet(reshard_worker_only_vector);
    auto reshard_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_silu/device/kernels/llama_all_gather_silu_reshard.cpp",
        reshard_worker_only_grid,
        reshard_kernel_config);

    auto reader_worker_common_range = CoreRangeSet(CoreRange({1, 0}, {2, 0}));
    auto reader_worker_common_cores = corerange_to_cores(reader_worker_common_range, 2, true);
    for (auto cr : reader_worker_common_range.ranges()) {
        printf(
            "reader common core: %zu %zu %zu %zu\n",
            cr.start_coord.x,
            cr.start_coord.y,
            cr.end_coord.x,
            cr.end_coord.y);
    }
    auto reader_common_kernel_config = tt::tt_metal::ReaderDataMovementConfig{};
    reader_common_kernel_config.compile_args = {
        ring_index,
        src0_cb_index,
        op_config.get_page_size(),
        q_output_cb_index,
    };
    auto reader_common_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_silu/device/kernels/"
        "llama_all_gather_silu_common_reader.cpp",
        reader_worker_common_range,
        reader_common_kernel_config);

    auto sem_mcast_ranges = CoreRangeSet(llama_configuration.semaphore_mcast_ranges);
    std::vector<uint32_t> mcast_start_x;
    std::vector<uint32_t> mcast_start_y;
    std::vector<uint32_t> mcast_end_x;
    std::vector<uint32_t> mcast_end_y;

    for (const auto& range : sem_mcast_ranges.ranges()) {
        auto start_core = mesh_device->worker_core_from_logical_core(range.start_coord);
        auto end_core = mesh_device->worker_core_from_logical_core(range.end_coord);
        if (writer_kernel_config.noc == tt::tt_metal::NOC::NOC_1) {
            std::swap(start_core, end_core);
        }
        mcast_start_x.push_back(start_core.x);
        mcast_start_y.push_back(start_core.y);
        mcast_end_x.push_back(end_core.x);
        mcast_end_y.push_back(end_core.y);
    }
    std::unordered_map<uint32_t, std::vector<uint32_t>> reader_common_dict;
    uint32_t start_value = 0;
    for (uint32_t link = 0; link < num_links; link++) {
        CoreCoord core = sender_worker_cores[link];
        uint32_t start_two = 0;
        if (num_links == 3) {
            if (ring_index == 0 || ring_index == 2) {
                start_two = link == 1 ? 1 : 0;
            } else {
                start_two = link == 1 ? 0 : 1;
            }
        }

        else if (num_links == 4) {
            if (ring_index == 1 || ring_index == 3) {
                start_two = link == 0;
            }
        }

        // construct input and output core x and y
        uint32_t base_pages_per_worker = input_tensor_num_pages / num_links;
        uint32_t remainder = input_tensor_num_pages % num_links;
        uint32_t input_tile_id_start = link * base_pages_per_worker + std::min(link, remainder);
        uint32_t input_tile_id_end = (link + 1) * base_pages_per_worker + std::min(link + 1, remainder);
        bool ends_6_tiles = ring_index % 2 == 0;
        bool starts_6_tiles = ring_index % 2 == 1;
        uint32_t worker_6_tiles = ends_6_tiles ? 3 : 0;
        if (num_links == 4 && ends_6_tiles) {
            input_tile_id_start = link * (base_pages_per_worker + 1);
            input_tile_id_end =
                link == worker_6_tiles ? input_tile_id_start + 6 : (link + 1) * (base_pages_per_worker + 1);
        } else if (num_links == 4 && starts_6_tiles) {
            input_tile_id_start = link == worker_6_tiles ? 0 : (link - 1) * (base_pages_per_worker + 1) + 6;
            input_tile_id_end = link == worker_6_tiles ? input_tile_id_start + 6 : input_tile_id_start + 8;
        }

        uint32_t worker_num_tiles_to_read = input_tile_id_end - input_tile_id_start;
        uint32_t input_first_core_tile_start_offset = input_tile_id_start % input_tensor_shard_num_pages;
        uint32_t output_first_core_tile_start_offset =
            (input_tensor_num_pages * ring_index + input_tile_id_start) % output_tensor_shard_num_pages;

        std::vector<uint32_t> input_tensor_cores_x;
        std::vector<uint32_t> input_tensor_cores_y;
        std::vector<uint32_t> output_tensor_cores_x;
        std::vector<uint32_t> output_tensor_cores_y;
        for (uint32_t i = input_tile_id_start / input_tensor_shard_num_pages;
             i < (input_tile_id_end + input_tensor_shard_num_pages - 1) / input_tensor_shard_num_pages;
             i++) {
            auto this_core = mesh_device->worker_core_from_logical_core(input_cores_vec[i]);
            input_tensor_cores_x.push_back(this_core.x);
            input_tensor_cores_y.push_back(this_core.y);
        }
        uint32_t incr = start_two ? 3 : 2;
        uint32_t num_cores_per_link = 3;
        if (num_links == 4) {
            incr = 2;
            num_cores_per_link = 2;
        }
        for (uint32_t i = start_value; i < start_value + num_cores_per_link; i++) {
            auto this_core = mesh_device->worker_core_from_logical_core(output_cores_this_device[i]);
            output_tensor_cores_x.push_back(this_core.x);
            output_tensor_cores_y.push_back(this_core.y);
        }
        start_value += incr;

        tt::log_debug(tt::LogOp, "input_tile_id_start: {}", input_tile_id_start);
        tt::log_debug(tt::LogOp, "input_tile_id_end: {}", input_tile_id_end);
        tt::log_debug(tt::LogOp, "worker_num_tiles_to_read: {}", worker_num_tiles_to_read);
        tt::log_debug(tt::LogOp, "input_first_core_tile_start_offset: {}", input_first_core_tile_start_offset);
        tt::log_debug(tt::LogOp, "output_first_core_tile_start_offset: {}", output_first_core_tile_start_offset);
        tt::log_debug(tt::LogOp, "input_tensor_cores_x: {}", input_tensor_cores_x);
        tt::log_debug(tt::LogOp, "input_tensor_cores_y: {}", input_tensor_cores_y);
        tt::log_debug(tt::LogOp, "output_tensor_cores_x: {}", output_tensor_cores_x);
        tt::log_debug(tt::LogOp, "output_tensor_cores_y: {}", output_tensor_cores_y);

        // if (link == 0) {
        //     // drain sync core is the first worker core
        //     drain_sync_core = mesh_device->worker_core_from_logical_core(core);
        // }
        //  Set reader runtime args
        std::vector<uint32_t> reader_common_rt_args = {
            input_tensor.buffer()->address(), buffer_tensor.buffer()->address()};
        std::vector<uint32_t> reader_rt_args = {
            input_tensor.buffer()->address(),    // tensor_address0
            input_tensor_shard_num_pages,        // num_tiles_per_core
            worker_num_tiles_to_read,            // num_tiles_to_read
            input_first_core_tile_start_offset,  // first_core_tile_start_offset
            input_tensor_cores_x.size(),         // num_cores
        };
        reader_rt_args.insert(reader_rt_args.end(), input_tensor_cores_x.begin(), input_tensor_cores_x.end());
        reader_rt_args.insert(reader_rt_args.end(), input_tensor_cores_y.begin(), input_tensor_cores_y.end());
        log_trace(tt::LogOp, "Reader Runtime Args:");
        for (const auto& arg : reader_rt_args) {
            log_trace(tt::LogOp, "\t{}", arg);
        }
        if ((core.x == 1 || core.x == 2) && core.y == 0) {
            for (uint32_t rt_idx = 1; rt_idx < reader_rt_args.size(); rt_idx++) {
                reader_common_rt_args.push_back(reader_rt_args[rt_idx]);
            }
            reader_common_dict[core.x] = reader_common_rt_args;
        } else {
            tt::tt_metal::SetRuntimeArgs(program, worker_sender_reader_kernel_id, {core}, reader_rt_args);
        }

        // Set writer runtime args
        bool wait_output_semaphore = (link == 2) && !enable_async_output_tensor;
        bool reset_global_semaphore = (link == 2) && !enable_async_output_tensor;
        uint32_t out_ready_sem_wait_value = (dynamic_alternate ? (ring_size + 1) : ring_size) * num_links;
        std::vector<uint32_t> writer_rt_args = {
            buffer_address,                       // tensor_address0
            semaphore.address(),                  // out_ready_sem_bank_addr (absolute address)
            output_tensor_shard_num_pages,        // num_tiles_per_core
            worker_num_tiles_to_read,             // num_tiles_to_read
            output_first_core_tile_start_offset,  // first_core_tile_start_offset
            output_tensor_cores_x.size(),         // num_cores
            wait_output_semaphore,                // wait_output_semaphore
            reset_global_semaphore,               // reset_global_semaphore
            drain_sync_core.x,                    // out_ready_sem_noc0_x
            drain_sync_core.y,                    // out_ready_sem_noc0_y
            out_ready_sem_wait_value,             // out_ready_sem_wait_value
            reshard_semaphore_id,
        };
        writer_rt_args.insert(writer_rt_args.end(), output_tensor_cores_x.begin(), output_tensor_cores_x.end());
        writer_rt_args.insert(writer_rt_args.end(), output_tensor_cores_y.begin(), output_tensor_cores_y.end());
        log_trace(tt::LogOp, "Writer Runtime Args:");
        for (const auto& arg : writer_rt_args) {
            log_trace(tt::LogOp, "\t{}", arg);
        }
        writer_rt_args.insert(writer_rt_args.end(), mcast_start_x.begin(), mcast_start_x.end());
        writer_rt_args.insert(writer_rt_args.end(), mcast_start_y.begin(), mcast_start_y.end());
        writer_rt_args.insert(writer_rt_args.end(), mcast_end_x.begin(), mcast_end_x.end());
        writer_rt_args.insert(writer_rt_args.end(), mcast_end_y.begin(), mcast_end_y.end());

        writer_rt_args.push_back(forward_device.has_value());
        if (forward_device.has_value()) {
            tt::tt_fabric::append_fabric_connection_rt_args(
                sender_device->id(), forward_device.value()->id(), link, program, {core}, writer_rt_args);
        }
        writer_rt_args.push_back(backward_device.has_value());
        if (backward_device.has_value()) {
            tt::tt_fabric::append_fabric_connection_rt_args(
                sender_device->id(), backward_device.value()->id(), link, program, {core}, writer_rt_args);
        }
        writer_rt_args.push_back(start_two);
        tt::tt_metal::SetRuntimeArgs(program, worker_sender_writer_kernel_id, {core}, writer_rt_args);
    }

    uint32_t core_idx = 0;
    uint32_t remaining_in_core = reshard_in_tiles_per_core;
    for (uint32_t i = 0; i < reshard_worker_grid.num_cores(); i++) {
        auto reshard_core = reshard_cores[i];
        uint32_t tiles_so_far = 0;
        std::vector<uint32_t> reshard_cores_x;
        std::vector<uint32_t> reshard_cores_y;
        std::vector<uint32_t> tiles_from_cur_core;
        while (tiles_so_far < reshard_out_tiles_per_core) {
            auto this_reshard_core = mesh_device->worker_core_from_logical_core(output_cores_vec[core_idx]);
            reshard_cores_x.push_back(this_reshard_core.x);
            reshard_cores_y.push_back(this_reshard_core.y);
            uint32_t incr_value = std::min(remaining_in_core, reshard_out_tiles_per_core - tiles_so_far);
            tiles_so_far += incr_value;
            tiles_from_cur_core.push_back(incr_value);
            remaining_in_core -= incr_value;
            if (remaining_in_core == 0) {
                core_idx++;
                remaining_in_core = reshard_in_tiles_per_core;
            }
        }
        std::vector<uint32_t> reshard_rt_args = {
            buffer_address,  // input address
            i,
            reshard_cores_x.size(),
            reshard_semaphore_id};
        reshard_rt_args.insert(reshard_rt_args.end(), reshard_cores_x.begin(), reshard_cores_x.end());
        reshard_rt_args.insert(reshard_rt_args.end(), reshard_cores_y.begin(), reshard_cores_y.end());
        reshard_rt_args.insert(reshard_rt_args.end(), tiles_from_cur_core.begin(), tiles_from_cur_core.end());

        if ((reshard_core.x == 1 || reshard_core.x == 2) && (reshard_core.y == 0)) {
            for (uint32_t rt_idx = 1; rt_idx < reshard_rt_args.size(); rt_idx++) {
                reader_common_dict[reshard_core.x].push_back(reshard_rt_args[rt_idx]);
            }
            tt::tt_metal::SetRuntimeArgs(
                program, reader_common_kernel_id, reshard_core, reader_common_dict[reshard_core.x]);
        } else {
            tt::tt_metal::SetRuntimeArgs(program, reshard_kernel_id, reshard_core, reshard_rt_args);
        }
    }
    auto override_runtime_arguments_callback =
        [worker_sender_reader_kernel_id,
         worker_sender_writer_kernel_id,
         reshard_kernel_id,
         reader_common_kernel_id,
         semaphore,
         sender_worker_cores,
         reshard_cores,
         cb_q_output](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto& input = input_tensors[0];
            const auto& buffer_tensor = input_tensors[1];
            const auto& output = output_tensors[0];
            auto dst_buffer_query = output.buffer();

            UpdateDynamicCircularBufferAddress(program, cb_q_output, *dst_buffer_query);

            auto semaphore = static_cast<const ttnn::AllGatherSilu*>(operation)->semaphore;

            log_trace(tt::LogOp, "DEBUG: semaphore: {}", semaphore.address());

            // update senders
            auto& worker_reader_sender_runtime_args_by_core = GetRuntimeArgs(program, worker_sender_reader_kernel_id);
            auto& worker_writer_sender_runtime_args_by_core = GetRuntimeArgs(program, worker_sender_writer_kernel_id);
            auto& reshard_runtime_args_by_core = GetRuntimeArgs(program, reshard_kernel_id);
            auto& reader_common_runtime_args_by_core = GetRuntimeArgs(program, reader_common_kernel_id);
            for (const auto& core : sender_worker_cores) {
                // reader
                bool common_core = ((core.x == 1 || core.x == 2) && core.y == 0);
                if (!common_core) {
                    auto& worker_reader_sender_runtime_args = worker_reader_sender_runtime_args_by_core[core.x][core.y];
                    worker_reader_sender_runtime_args[0] = input.buffer()->address();
                }

                // writer
                auto& worker_writer_sender_runtime_args = worker_writer_sender_runtime_args_by_core[core.x][core.y];
                worker_writer_sender_runtime_args[0] = buffer_tensor.buffer()->address();
                worker_writer_sender_runtime_args[1] = semaphore.address();
            }

            for (const auto& r_core : reshard_cores) {
                bool common_core = ((r_core.x == 1 || r_core.x == 2) && r_core.y == 0);
                if (!common_core) {
                    auto& reshard_runtime_args = reshard_runtime_args_by_core[r_core.x][r_core.y];
                    reshard_runtime_args[0] = buffer_tensor.buffer()->address();
                } else {
                    auto& reader_common_runtime_args = reader_common_runtime_args_by_core[r_core.x][r_core.y];
                    reader_common_runtime_args[0] = input.buffer()->address();
                    reader_common_runtime_args[1] = buffer_tensor.buffer()->address();
                }
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn
