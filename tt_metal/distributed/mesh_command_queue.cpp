// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_command_queue.hpp"
#include "mesh_workload_utils.hpp"

namespace tt::tt_metal::distributed {

MeshCommandQueue::MeshCommandQueue(MeshDevice* mesh_device, uint32_t id) {
    this->mesh_device_ = mesh_device;
    this->id_ = id;

    this->config_buffer_mgr_ = tt::tt_metal::WorkerConfigBufferMgr();
    program_dispatch::initialize_worker_config_buf_mgr(this->config_buffer_mgr_);
    this->populate_virtual_program_dispatch_core();
    this->populate_dispatch_core_type();
    auto device = this->mesh_device_->get_devices()[0];
    buf_dispatch_constants_ = buffer_utils::generate_buffer_dispatch_constants(device->sysmem_manager(), dispatch_core_type_, id_);
}

uint32_t MeshCommandQueue::num_worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) {
    if (core_type == HalProgrammableCoreType::TENSIX) {
        uint32_t num_workers = 0;
        for (auto& device : this->mesh_device_->get_devices()) {
            if (num_workers) {
                TT_FATAL(
                    num_workers == device->num_worker_cores(core_type, sub_device_id),
                    "Worker grid size must be consistent across all devices in a Mesh.");
            } else {
                num_workers = device->num_worker_cores(core_type, sub_device_id);
            }
        }
        return num_workers;
    } else {
        uint32_t min_num_worker_cores = std::numeric_limits<uint32_t>::max();
        for (auto& device : this->mesh_device_->get_devices()) {
            min_num_worker_cores = std::min(min_num_worker_cores, device->num_worker_cores(core_type, sub_device_id));
        }
        return min_num_worker_cores;
    }
}

void MeshCommandQueue::populate_virtual_program_dispatch_core() {
    int device_idx = 0;
    for (auto device : this->mesh_device_->get_devices()) {
        if (device_idx) {
            TT_FATAL(
                this->dispatch_core_ == device->virtual_program_dispatch_core(this->id_),
                "Expected Dispatch Cores to match across devices in a Mesh");
        } else {
            this->dispatch_core_ = device->virtual_program_dispatch_core(this->id_);
        }
        device_idx++;
    }
}

void MeshCommandQueue::populate_dispatch_core_type() {
    uint32_t device_idx = 0;
    for (auto device : this->mesh_device_->get_devices()) {
        if (device_idx) {
            TT_FATAL(
                this->dispatch_core_type_ == dispatch_core_manager::instance().get_dispatch_core_type(device->id()),
                "Expected the Dispatch Core Type to match across device in a Mesh");
        } else {
            this->dispatch_core_type_ = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
        }
        device_idx++;
    }
}

CoreCoord MeshCommandQueue::virtual_program_dispatch_core() const { return this->dispatch_core_; }

CoreType MeshCommandQueue::dispatch_core_type() const { return this->dispatch_core_type_; }

void MeshCommandQueue::enqueue_mesh_workload(MeshWorkload& mesh_workload, bool blocking) {
    std::unordered_set<SubDeviceId> sub_device_ids = mesh_workload.determine_sub_device_ids(mesh_device_);
    TT_FATAL(sub_device_ids.size() == 1, "Programs must be executed on a single sub-device");
    auto sub_device_id = *(sub_device_ids.begin());
    auto mesh_device_id = this->mesh_device_->get_mesh_id();
    TT_FATAL(
        mesh_workload.get_program_binary_status(mesh_device_id) != ProgramBinaryStatus::NotSent,
        "Expected program binaries to be written to the MeshDevice.");

    // Compute number of workers being used for this workload.
    uint32_t num_workers = 0;
    bool unicast_go_signals = mesh_workload.runs_on_noc_unicast_only_cores();
    bool mcast_go_signals = mesh_workload.runs_on_noc_multicast_only_cores();
    if (mcast_go_signals) {
        num_workers += this->num_worker_cores(HalProgrammableCoreType::TENSIX, sub_device_id);
    }
    if (unicast_go_signals) {
        num_workers += this->num_worker_cores(HalProgrammableCoreType::ACTIVE_ETH, sub_device_id);
    }

    program_dispatch::ProgramDispatchMetadata dispatch_metadata;
    // Reserve space in the L1 Kernel Config Ring Buffer for this workload.
    program_dispatch::reserve_space_in_kernel_config_buffer(
        this->config_buffer_mgr_,
        mesh_workload.get_program_config_sizes(),
        mesh_workload.get_program_binary_status(mesh_device_id),
        num_workers,
        this->expected_num_workers_completed_,
        dispatch_metadata);

    std::unordered_set<uint32_t> chip_ids_in_workload = {};
    // Iterate over all programs. Update dispatch commands per program to reflect
    // current device state. Write the finalized program command sequence to each
    // physical device tied to the program.
    for (const auto& device_range : mesh_workload.get_logical_device_ranges()) {
        auto& program = mesh_workload.get_program_on_device_range(device_range);
        auto& program_cmd_seq = mesh_workload.get_dispatch_cmds_for_program(program);

        program_dispatch::update_program_dispatch_commands(
            program,
            program_cmd_seq,
            this->worker_launch_message_buffer_state_.get_mcast_wptr(),
            this->worker_launch_message_buffer_state_.get_unicast_wptr(),
            this->expected_num_workers_completed_,
            this->virtual_program_dispatch_core(),
            this->dispatch_core_type(),
            sub_device_id,
            dispatch_metadata,
            mesh_workload.get_program_binary_status(mesh_device_id),
            std::pair<bool, int>(unicast_go_signals, this->num_worker_cores(HalProgrammableCoreType::ACTIVE_ETH, sub_device_id)));

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
    // Send go signals to devices not running a program to ensure consistent global state
    for (auto& device : this->mesh_device_->get_devices()) {
        if (chip_ids_in_workload.find(device->id()) == chip_ids_in_workload.end()) {
            experimental::write_go_signal(
                device->command_queue(this->id_),
                this->expected_num_workers_completed_,
                this->virtual_program_dispatch_core(),
                mcast_go_signals,
                unicast_go_signals,
                this->num_worker_cores(HalProgrammableCoreType::ACTIVE_ETH, sub_device_id));
        }
    }
    // Increment Launch Message Buffer Write Pointers
    if (mcast_go_signals) {
        this->worker_launch_message_buffer_state_.inc_mcast_wptr(1);
    }
    if (unicast_go_signals) {
        this->worker_launch_message_buffer_state_.inc_unicast_wptr(1);
    }
    // Update the expected number of workers dispatch must wait on
    this->expected_num_workers_completed_ += num_workers;
    // From the dispatcher's perspective, binaries are now committed to DRAM
    mesh_workload.set_program_binary_status(mesh_device_id, ProgramBinaryStatus::Committed);
    mesh_workload.set_last_used_command_queue_for_testing(this);

    if (blocking) {
        this->finish();
    }
}

// Need a Shard orientation equivalent here to figure out the order in which data is written to devices.
// Allows for replication/sharding of data in logical x/y to arbitary device x/y/
void MeshCommandQueue::write_sharded_buffer(MeshBuffer& buffer, const void* src) {
    auto global_buffer_shape = buffer.global_shard_spec().global_buffer_shape;
    auto global_buffer_size = buffer.global_shard_spec().global_buffer_size;

    auto shard_shape = buffer.physical_shard_shape();
    auto replicated_dims = buffer.replicated_dims();
    TT_FATAL(not std::get<0>(replicated_dims), "Replication along the x axis is not supported for buffers.");
    auto datum_size_bytes = buffer.datum_size_bytes();

    auto stride_size_bytes = datum_size_bytes * std::get<0>(global_buffer_shape);
    auto single_read_size = datum_size_bytes * std::get<0>(shard_shape);
    auto total_read_size_per_shard = single_read_size * std::get<1>(shard_shape);

    auto num_shards_x = std::get<0>(global_buffer_shape) / std::get<0>(shard_shape);
    auto num_shards_y = std::get<1>(global_buffer_shape) / std::get<1>(shard_shape);

    uint32_t num_devices_x = buffer.mesh_device()->num_cols();
    uint32_t num_devices_y = buffer.mesh_device()->num_rows();

    uint32_t device_x = 0;
    uint32_t device_y = 0;
    std::vector<uint32_t> shard_data = std::vector<uint32_t>(total_read_size_per_shard / sizeof(uint32_t), 0);
    for (std::size_t shard_y = 0; shard_y < num_shards_y; shard_y++) {
        for (std::size_t shard_x = 0; shard_x < num_shards_x; shard_x++) {
            auto read_offset = shard_x * single_read_size + shard_y * stride_size_bytes * std::get<1>(shard_shape);
            uint32_t size_to_read = total_read_size_per_shard;
            uint32_t local_offset = 0;
            while (size_to_read) {
                std::memcpy(shard_data.data() + local_offset * (single_read_size / sizeof(uint32_t)), (uint8_t*)(src) + read_offset + local_offset * stride_size_bytes, single_read_size);
                size_to_read -= single_read_size;
                local_offset++;
            }
            if (std::get<1>(replicated_dims)) {
                for (auto replicated_device_y = 0 ; replicated_device_y < num_devices_y; replicated_device_y++) {
                    std::cout << "Replicate to: " << device_x << " " << replicated_device_y << std::endl;
                }
                device_x++;
            } else {
                std::cout << "Write to: " << device_x << " " << device_y << std::endl;
                device_x = (device_x + 1) % num_devices_x;
                if (device_x == 0) device_y++;
            }
            
            // Write shard to device here.
            // If replicated, check the shard orientation and copy data across the opposite orientation.
            // for (auto device : devices to replicate)
                // write_shard();
            
            // Device index updated according to shard orientation.
            
            // if (row_major_write) {
            //     device_x = (device_x + 1) % num_devices_x
            //     if (device_x == 0) device_y++;
            // } else {
            //     device_y = (device_y + 1) % num_devices_y
            //     if (device_y == 0) device_x++;
            // }
            std::cout << "======= Shard data ========" << std::endl;
            for (int i = 0; i < shard_data.size(); i++) {
                if (i % std::get<0>(shard_shape) == 0) {
                    std::cout << std::endl;
                }
                std::cout << shard_data[i] << " ";
            }
            std::cout << std::endl;
        }
    }

}

void MeshCommandQueue::enqueue_write_to_sub_grid(MeshBuffer& buffer, const void* src, bool blocking, const LogicalDeviceRange& device_range) {
    // TODO: Add proper support for Mesh Level Sub Devices
    auto sub_device_ids = tt::stl::Span<const SubDeviceId>(mesh_device_->get_device(0)->get_sub_device_ids());
    std::array<uint32_t, dispatch_constants::DISPATCH_MESSAGE_ENTRIES> expected_num_workers_completed;
    expected_num_workers_completed[0] = expected_num_workers_completed_;

    if (buffer.global_layout() == MeshBufferLayout::REPLICATED) {
        if (is_sharded(buffer.device_local_layout().buffer_layout)) {
            for (std::size_t logical_x = device_range.start_coord.x; logical_x < device_range.end_coord.x; logical_x++) {
                for (std::size_t logical_y = device_range.start_coord.y; logical_y < device_range.end_coord.y; logical_y++) {
                    auto device_shard_view = buffer.get_shard_buffer(logical_x, logical_y);
                    buffer_utils::ShardedBufferDispatchParams dispatch_params = buffer_utils::initialize_sharded_buf_dispatch_params(*device_shard_view, id_, expected_num_workers_completed);
                    const auto cores = buffer_utils::get_cores_for_sharded_buffer(dispatch_params, *device_shard_view);
                    for (uint32_t core_id = 0; core_id < device_shard_view->num_cores(); ++core_id) {
                        buffer_utils::write_sharded_buffer_to_core(src, core_id, *device_shard_view, dispatch_params, buf_dispatch_constants_, sub_device_ids, cores);
                    }
                }
            }
        } else {
            for (std::size_t logical_x = device_range.start_coord.x; logical_x < device_range.end_coord.x; logical_x++) {
                for (std::size_t logical_y = device_range.start_coord.y; logical_y < device_range.end_coord.y; logical_y++) {
                    auto device_shard_view = buffer.get_shard_buffer(logical_x, logical_y);
                    auto dispatch_params = buffer_utils::initialize_interleaved_buf_dispatch_params(*device_shard_view, buf_dispatch_constants_, id_, expected_num_workers_completed);
                    buffer_utils::write_interleaved_buffer_to_device(src, dispatch_params, *device_shard_view, buf_dispatch_constants_, sub_device_ids);
                }
            }
        }
    } else {
        TT_FATAL(false, "Writing to a Sharded MeshBuffer is not currently supported.");
    }
    if (blocking) {
        this->finish();
    }
}

void MeshCommandQueue::enqueue_write_mesh_buffer(MeshBuffer& buffer, const void* src, bool blocking) {
    LogicalDeviceRange mesh_device_extent({0, 0}, {buffer.mesh_device()->num_cols(), buffer.mesh_device()->num_rows()});
    this->enqueue_write_to_sub_grid(buffer, src, blocking, mesh_device_extent);
}

void MeshCommandQueue::finish() {
    for (auto device : this->mesh_device_->get_devices()) {
        Finish(device->command_queue(this->id_));
    }
}

}  // namespace tt::tt_metal::distributed
