// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/fabric.hpp>
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_legacy/device/all_gather_legacy_op.hpp"
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

tt::tt_metal::operation::ProgramWithCallbacks all_gather_legacy(
    const Tensor& input_tensor,
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
    auto mesh_device = input_tensor.mesh_device();
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
    std::vector<Tensor> output_tensors = {output_tensor};
    const auto& op_config = ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, topology);
    auto [num_targets_forward, num_targets_backward, dynamic_alternate] =
        ccl::get_forward_backward_configuration(ring_size, ring_index, topology);

    // Get worker cores, assuming 1 worker per link
    auto sender_core_range = CoreRangeSet(CoreRange({0, 0}, {0, 0}));
    auto receiver_core_range = CoreRangeSet(CoreRange({1, 0}, {1, 0}));
    auto semaphore_core_range = CoreRangeSet(CoreRange({0, 0}, {1, 0}));
    auto sender_cores = corerange_to_cores(sender_core_range, 1, true);
    auto receiver_cores = corerange_to_cores(receiver_core_range, 1, true);
    uint32_t num_workers_per_link = 1;
    const auto [sender_worker_core_range, sender_worker_cores] =
        choose_worker_cores(num_links, num_workers_per_link, enable_persistent_fabric_mode, mesh_device, sub_device_id);

    // L1 Scratch CB Creation
    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    uint32_t l1_scratch_cb_page_size_bytes = op_config.get_page_size();
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t cb_num_pages = 3 * num_pages_per_packet;  // tripple buffering
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

    uint32_t receiver_cb_index = tt::CBIndex::c_16;
    tt::tt_metal::CircularBufferConfig cb_receiver_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{receiver_cb_index, df}})
            .set_page_size(receiver_cb_index, l1_scratch_cb_page_size_bytes);
    tt::tt_metal::CBHandle cb_receiver_workers = CreateCircularBuffer(program, receiver_core_range, cb_src0_config);

    uint32_t sync_semaphore = tt::tt_metal::CreateSemaphore(program, semaphore_core_range, 0);
    // Tensor Info
    const auto input_tensor_layout = input_tensor.buffer()->buffer_layout();
    const auto input_tensor_buffer_type = input_tensor.buffer()->buffer_type();
    const auto input_tensor_page_layout = input_tensor.layout();
    const auto input_tensor_num_pages = input_tensor.buffer()->num_pages();
    const auto output_tensor_layout = output_tensor.buffer()->buffer_layout();
    const auto output_tensor_buffer_type = output_tensor.buffer()->buffer_type();
    const auto output_tensor_page_layout = output_tensor.layout();

    // KERNEL CREATION
    // Sender Reader
    auto sender_reader_kernel_config = tt::tt_metal::ReaderDataMovementConfig{};
    sender_reader_kernel_config.compile_args = {
        static_cast<uint32_t>(input_tensor_buffer_type),  // buffer0_type
        src0_cb_index,                                    // cb0_id
        num_pages_per_packet,                             // packet_size_in_pages
        op_config.get_page_size(),                        // tensor0_page_size
        ring_size,                                        // num_devices
        input_tensor_num_pages,                           // num_tiles_per_device
        ring_index,                                       // ring index
    };
    log_trace(tt::LogOp, "Reader Compile Args:");
    for (const auto& arg : sender_reader_kernel_config.compile_args) {
        log_trace(tt::LogOp, "\t{}", arg);
    }
    auto worker_sender_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_legacy/device/kernels/sender_reader.cpp",
        sender_core_range,
        sender_reader_kernel_config);

    // Sender Writer
    auto sender_writer_kernel_config = tt::tt_metal::WriterDataMovementConfig{};
    sender_writer_kernel_config.compile_args = {
        src0_cb_index,  // cb0_id
        receiver_cb_index,
        reserved_packet_header_CB_index,  // reserved_packet_header_cb_id
        op_config.get_page_size(),        // tensor0_page_size
        num_pages_per_packet,             // packet_size_in_pages
        ring_size,                        // num_devices
        input_tensor_num_pages,           // num_tiles_per_device
        ring_index,                       // ring index
        // static_cast<uint32_t>(output_tensor_buffer_type),  // buffer0_type

    };
    for (const auto& arg : sender_writer_kernel_config.compile_args) {
        log_trace(tt::LogOp, "\t{}", arg);
    }
    auto worker_sender_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_legacy/device/kernels/sender_writer.cpp",
        sender_core_range,
        sender_writer_kernel_config);

    // Receiver
    auto receiver_kernel_config = tt::tt_metal::ReaderDataMovementConfig{};
    receiver_kernel_config.compile_args = {
        static_cast<uint32_t>(output_tensor_buffer_type),  // buffer0_type
        receiver_cb_index,                                 // cb0_id
        reserved_packet_header_CB_index,                   // reserved_packet_header_cb_id
        op_config.get_page_size(),                         // tensor0_page_size
        num_pages_per_packet,                              // packet_size_in_pages
        ring_size,                                         // num_devices
        input_tensor_num_pages,                            // num_tiles_per_device
        ring_index,                                        // ring index

    };
    for (const auto& arg : receiver_kernel_config.compile_args) {
        log_trace(tt::LogOp, "\t{}", arg);
    }
    auto worker_receiver_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_legacy/device/kernels/receiver.cpp",
        receiver_core_range,
        receiver_kernel_config);

    std::vector<uint32_t> sender_reader_rt_args = {
        input_tensor.buffer()->address(),   // tensor_address0
        output_tensor.buffer()->address(),  // tensor_address1
        sync_semaphore,                     // sync_semaphore
    };
    log_trace(tt::LogOp, "Reader Runtime Args:");
    for (const auto& arg : sender_reader_rt_args) {
        log_trace(tt::LogOp, "\t{}", arg);
    }
    tt::tt_metal::SetRuntimeArgs(program, worker_sender_reader_kernel_id, {sender_cores[0]}, sender_reader_rt_args);

    // Set writer runtime args
    auto receiver_core_noc = mesh_device->worker_core_from_logical_core(receiver_cores[0]);
    auto sender_core_noc = mesh_device->worker_core_from_logical_core(sender_cores[0]);
    std::vector<uint32_t> sender_writer_rt_args = {
        semaphore.address(),  // out_ready_sem_bank_addr (absolute address)
        receiver_core_noc.x,  // out_ready_sem_noc0_x
        receiver_core_noc.y,  // out_ready_sem_noc0_y
    };
    log_trace(tt::LogOp, "Writer Runtime Args:");
    for (const auto& arg : sender_writer_rt_args) {
        log_trace(tt::LogOp, "\t{}", arg);
    }
    sender_writer_rt_args.push_back(forward_device.has_value());
    if (forward_device.has_value()) {
        tt::tt_fabric::append_fabric_connection_rt_args(
            sender_device->id(), forward_device.value()->id(), 0, program, sender_cores[0], sender_writer_rt_args);
    }
    sender_writer_rt_args.push_back(backward_device.has_value());
    if (backward_device.has_value()) {
        tt::tt_fabric::append_fabric_connection_rt_args(
            sender_device->id(), backward_device.value()->id(), 0, program, sender_cores[0], sender_writer_rt_args);
    }

    tt::tt_metal::SetRuntimeArgs(program, worker_sender_writer_kernel_id, {sender_cores[0]}, sender_writer_rt_args);

    // Set writer runtime args
    std::vector<uint32_t> receiver_rt_args = {
        output_tensor.buffer()->address(),  // tensor_address0
        semaphore.address(),                // out_ready_sem_bank_addr (absolute address)
        sync_semaphore,
        sender_core_noc.x,  // out_ready_sem_noc0_x
        sender_core_noc.y,  // out_ready_sem_noc0_y
    };
    log_trace(tt::LogOp, "Writer Runtime Args:");
    for (const auto& arg : receiver_rt_args) {
        log_trace(tt::LogOp, "\t{}", arg);
    }

    tt::tt_metal::SetRuntimeArgs(program, worker_receiver_kernel_id, {receiver_cores[0]}, receiver_rt_args);

    auto override_runtime_arguments_callback =
        [worker_sender_reader_kernel_id,
         worker_sender_writer_kernel_id,
         worker_receiver_kernel_id,
         semaphore,
         sender_cores,
         receiver_cores](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto& input = input_tensors[0];
            const auto& output = output_tensors[0];

            auto semaphore = static_cast<const ttnn::AllGatherLegacy*>(operation)->semaphore;

            log_trace(tt::LogOp, "DEBUG: semaphore: {}", semaphore.address());

            // update senders
            auto& worker_reader_sender_runtime_args_by_core = GetRuntimeArgs(program, worker_sender_reader_kernel_id);
            auto& worker_writer_sender_runtime_args_by_core = GetRuntimeArgs(program, worker_sender_writer_kernel_id);
            auto& worker_receiver_runtime_args_by_core = GetRuntimeArgs(program, worker_receiver_kernel_id);

            for (const auto& core : sender_cores) {
                // reader
                auto& worker_reader_sender_runtime_args = worker_reader_sender_runtime_args_by_core[core.x][core.y];
                worker_reader_sender_runtime_args[0] = input.buffer()->address();
                worker_reader_sender_runtime_args[1] = output.buffer()->address();
                // writer
                auto& worker_writer_sender_runtime_args = worker_writer_sender_runtime_args_by_core[core.x][core.y];
                worker_writer_sender_runtime_args[0] = semaphore.address();
            }
            for (const auto& core : receiver_cores) {
                auto& worker_receiver_runtime_args = worker_receiver_runtime_args_by_core[core.x][core.y];
                worker_receiver_runtime_args[0] = output.buffer()->address();
                worker_receiver_runtime_args[1] = semaphore.address();
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn
