// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_command_queue.hpp"
#include "mesh_workload_utils.hpp"

namespace tt::tt_metal::distributed {

MeshCommandQueue::MeshCommandQueue(std::shared_ptr<MeshDevice> mesh_device, uint32_t id) {
    this->mesh_device_ = mesh_device;
    this->id_ = id;

    this->config_buffer_mgr_ = tt::tt_metal::WorkerConfigBufferMgr();
    for (uint32_t index = 0; index < tt::tt_metal::hal.get_programmable_core_type_count(); index++) {
        this->config_buffer_mgr_.init_add_buffer(
            tt::tt_metal::hal.get_dev_addr(
                tt::tt_metal::hal.get_programmable_core_type(index), tt::tt_metal::HalL1MemAddrType::KERNEL_CONFIG),
            tt::tt_metal::hal.get_dev_size(
                tt::tt_metal::hal.get_programmable_core_type(index), tt::tt_metal::HalL1MemAddrType::KERNEL_CONFIG));
    }
    // Subtract 1 from the number of entries, so the watcher can read information (e.g. fired asserts) from the
    // previous launch message.
    this->config_buffer_mgr_.init_add_buffer(0, launch_msg_buffer_num_entries - 1);
}

void MeshCommandQueue::enqueue_mesh_workload(MeshWorkload& mesh_workload, bool blocking) {
    std::unordered_set<SubDeviceId> sub_device_ids = mesh_workload.determine_sub_device_ids(mesh_device_);
    TT_FATAL(sub_device_ids.size() == 1, "Programs must be executed on a single sub-device");
    auto sub_device_id = *(sub_device_ids.begin());

    uint32_t num_workers = 0;
    if (mesh_workload.runs_on_noc_multicast_only_cores()) {
        num_workers += this->mesh_device_->num_worker_cores(HalProgrammableCoreType::TENSIX, sub_device_id);
    }
    if (mesh_workload.runs_on_noc_unicast_only_cores()) {
        num_workers += this->mesh_device_->num_worker_cores(HalProgrammableCoreType::ACTIVE_ETH, sub_device_id);
    }

    program_utils::ProgramDispatchMetadata dispatch_metadata;
    program_utils::reserve_space_in_kernel_config_buffer(
        this->config_buffer_mgr_,
        mesh_workload.get_program_config_sizes(),
        mesh_workload.kernel_binary_always_stored_in_ringbuffer(),
        mesh_workload.program_binary_status,
        num_workers,
        this->expected_num_workers_completed_,
        dispatch_metadata);

    const tt::stl::Span<ConfigBufferEntry> kernel_config_addrs{
        dispatch_metadata.kernel_config_addrs.data(), dispatch_metadata.kernel_config_addrs.size() - 1};

    std::unordered_set<uint32_t> chip_ids_in_workload = {};

    for (auto& program_on_grid : mesh_workload.get_programs()) {
        auto& device_range = program_on_grid.first;
        auto& program_cmd_seq = mesh_workload.get_dispatch_cmds_for_program(program_on_grid.second);

        program_utils::update_program_dispatch_commands(
            program_on_grid.second,
            program_cmd_seq,
            kernel_config_addrs,
            this->worker_launch_message_buffer_state_.get_mcast_wptr(),
            this->worker_launch_message_buffer_state_.get_unicast_wptr(),
            this->expected_num_workers_completed_,
            this->mesh_device_->virtual_program_dispatch_core(this->id_),
            this->mesh_device_->dispatch_core_type(),
            sub_device_id,
            dispatch_metadata,
            mesh_workload.program_binary_status,
            this->mesh_device_->num_worker_cores(HalProgrammableCoreType::ACTIVE_ETH, sub_device_id));

        for (std::size_t logical_x = device_range.start_coord.x; logical_x < device_range.end_coord.x; logical_x++) {
            for (std::size_t logical_y = device_range.start_coord.y; logical_y < device_range.end_coord.y;
                 logical_y++) {
                experimental::write_program_commands(
                    this->mesh_device_->get_device(logical_y, logical_x)->command_queue(this->id_),
                    program_cmd_seq,
                    num_workers,
                    sub_device_id,
                    dispatch_metadata.stall_first,
                    dispatch_metadata.stall_before_program,
                    false);
                chip_ids_in_workload.insert(this->mesh_device_->get_device(logical_y, logical_x)->id());
            }
        }
    }

    for (auto& device : this->mesh_device_->get_devices()) {
        if (chip_ids_in_workload.find(device->id()) == chip_ids_in_workload.end()) {
            experimental::write_go_signal(
                device->command_queue(this->id_),
                this->expected_num_workers_completed_,
                this->mesh_device_->virtual_program_dispatch_core(this->id_),
                mesh_workload.runs_on_noc_multicast_only_cores(),
                mesh_workload.runs_on_noc_unicast_only_cores(),
                this->mesh_device_->num_worker_cores(HalProgrammableCoreType::ACTIVE_ETH, sub_device_id));
        }
    }
    if (mesh_workload.runs_on_noc_multicast_only_cores()) {
        this->worker_launch_message_buffer_state_.inc_mcast_wptr(1);
    }
    if (mesh_workload.runs_on_noc_unicast_only_cores()) {
        this->worker_launch_message_buffer_state_.inc_unicast_wptr(1);
    }
    this->expected_num_workers_completed_ += num_workers;

    mesh_workload.program_binary_status = ProgramBinaryStatus::Committed;

    if (blocking) {
        this->finish();
    }
    mesh_workload.set_last_used_command_queue(this);
}

void MeshCommandQueue::finish() {
    for (auto device : this->mesh_device_->get_devices()) {
        Finish(device->command_queue(this->id_));
    }
}

}  // namespace tt::tt_metal::distributed
