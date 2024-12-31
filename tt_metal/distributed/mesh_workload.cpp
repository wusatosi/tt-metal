// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_command_queue.hpp"
#include "mesh_workload.hpp"
#include "mesh_workload_utils.hpp"
#include "tt_metal/detail/tt_metal.hpp"

namespace tt::tt_metal::distributed {

MeshWorkload::MeshWorkload() {
    for (uint32_t i = 0; i < hal.get_programmable_core_type_count(); i++) {
        this->kernel_groups_.push_back({});
        this->kernels_.push_back({});
    }
}

void MeshWorkload::add_program(const LogicalDeviceRange& device_range, Program& program) {
    this->programs_[device_range] = std::move(program);
}

void MeshWorkload::compile(MeshCommandQueue& mesh_cq) {
    auto& mesh_device = mesh_cq.mesh_device();
    for (auto& program_on_grid : this->programs_) {
        program_on_grid.second.compile(mesh_device->get_device(0));
        program_on_grid.second.allocate_circular_buffers(mesh_device->get_device(0));
        tt::tt_metal::detail::ValidateCircularBufferRegion(program_on_grid.second, mesh_device->get_device(0));
    }
    program_utils::finalize(*this, mesh_device->get_device(0));
}

void MeshWorkload::load_binaries(MeshCommandQueue& mesh_cq) {
    // Allocate kernel binary buffers of max size across all devices, to ensure
    // we have lock step allocation.

    auto& mesh_device = mesh_cq.mesh_device();
    if (this->program_binary_status == ProgramBinaryStatus::NotSent) {
        uint32_t max_kernel_bin_buf_size = 0;
        for (auto& program_on_grid : this->programs_) {
            uint32_t curr_kernel_bin_size =
                program_on_grid.second.get_program_transfer_info().binary_data.size() * sizeof(uint32_t);
            max_kernel_bin_buf_size = std::max(max_kernel_bin_buf_size, curr_kernel_bin_size);
        }
        // Allocate a buffer for kernel binaries on each device.
        // Once MeshBuffer is available, allocate kernel bin MeshBuffer directly here
        for (auto device : mesh_device->get_devices()) {
            std::shared_ptr<Buffer> kernel_bin_buf = Buffer::create(
                device,
                max_kernel_bin_buf_size,
                HostMemDeviceCommand::PROGRAM_PAGE_SIZE,
                BufferType::DRAM,
                TensorMemoryLayout::INTERLEAVED,
                std::nullopt,
                false);
            this->kernel_bin_buffers_.insert(
                kernel_bin_buf);  // Tie the lifetime of kernel binary buffers to the MeshWorkload
        }
        // Iterate over the sub-grids and EnqueueWriteMeshBuffer to each sub-grid that runs the program
        for (auto& program_on_grid : this->programs_) {
            auto& device_range = program_on_grid.first;
            std::size_t kernel_bin_size =
                program_on_grid.second.get_program_transfer_info().binary_data.size() * sizeof(uint32_t);
            for (std::size_t logical_x = device_range.start_coord.x; logical_x < device_range.end_coord.x;
                 logical_x++) {
                for (std::size_t logical_y = device_range.start_coord.y; logical_y < device_range.end_coord.y;
                     logical_y++) {
                    Device* device = mesh_device->get_device(logical_y, logical_x);
                    // Get a view of the allocated buffer that matches the size of the kernel binary
                    // for the sub grid
                    std::shared_ptr<Buffer> buffer_view = Buffer::create(
                        device,
                        (*(this->kernel_bin_buffers_.begin()))->address(),
                        kernel_bin_size,
                        HostMemDeviceCommand::PROGRAM_PAGE_SIZE,
                        BufferType::DRAM,
                        TensorMemoryLayout::INTERLEAVED,
                        std::nullopt,
                        false);
                    EnqueueWriteBuffer(
                        device->command_queue(mesh_cq.id()),
                        buffer_view,
                        program_on_grid.second.get_program_transfer_info().binary_data.data(),
                        false);
                    program_on_grid.second.set_kernels_bin_buffer(buffer_view);
                    program_on_grid.second.set_program_binary_status(device->id(), ProgramBinaryStatus::InFlight);
                }
            }
        }
        this->program_binary_status = ProgramBinaryStatus::InFlight;
    }
}

void MeshWorkload::generate_dispatch_commands(MeshCommandQueue& mesh_cq) {
    auto& mesh_device = mesh_cq.mesh_device();
    for (auto& program_on_grid : this->programs_) {
        auto grid_start = program_on_grid.first.start_coord;
        program_on_grid.second.generate_dispatch_commands(mesh_device->get_device(grid_start.y, grid_start.x));
    }
}

bool MeshWorkload::runs_on_noc_multicast_only_cores() {
    bool ret = false;
    for (auto& program_on_grid : this->programs_) {
        ret = ret || (program_on_grid.second.runs_on_noc_multicast_only_cores());
    }
    return ret;
}

bool MeshWorkload::runs_on_noc_unicast_only_cores() {
    bool ret = false;
    for (auto& program_on_grid : this->programs_) {
        ret = ret || (program_on_grid.second.runs_on_noc_unicast_only_cores());
    }
    return ret;
}

bool MeshWorkload::kernel_binary_always_stored_in_ringbuffer() {
    bool stored_in_ring_buf = true;
    for (auto& program_on_grid : this->programs_) {
        stored_in_ring_buf &= program_on_grid.second.kernel_binary_always_stored_in_ringbuffer();
    }
    return stored_in_ring_buf;
}

std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>& MeshWorkload::get_kernels(
    uint32_t programmable_core_type_index) {
    if (not this->kernels_.at(programmable_core_type_index).size()) {
        for (auto& program_on_grid : this->programs_) {
            auto& device_range = program_on_grid.first;
            uint32_t device_range_handle = (device_range.start_coord.y << 24) | (device_range.start_coord.x << 16);
            for (auto kernel : program_on_grid.second.get_kernels(programmable_core_type_index)) {
                KernelHandle handle = (device_range_handle | kernel.first);
                this->kernels_.at(programmable_core_type_index).insert({handle, kernel.second});
            }
        }
    }
    return this->kernels_.at(programmable_core_type_index);
}

std::vector<std::shared_ptr<KernelGroup>>& MeshWorkload::get_kernel_groups(uint32_t programmable_core_type_index) {
    if (not this->kernel_groups_.at(programmable_core_type_index).size()) {
        for (auto& program_on_grid : this->programs_) {
            auto& device_range = program_on_grid.first;
            uint32_t device_range_handle = (device_range.start_coord.y << 24) | (device_range.start_coord.x << 16);
            for (auto kg : program_on_grid.second.get_kernel_groups(programmable_core_type_index)) {
                for (auto& optional_kernel_id : kg->kernel_ids) {
                    if (optional_kernel_id.has_value()) {
                        optional_kernel_id = (device_range_handle | optional_kernel_id.value());
                    }
                }
                this->kernel_groups_.at(programmable_core_type_index).push_back(kg);
            }
        }
    }
    return this->kernel_groups_.at(programmable_core_type_index);
}

std::vector<Semaphore>& MeshWorkload::semaphores() {
    if (not this->semaphores_.size()) {
        for (auto& program_on_grid : this->programs_) {
            this->semaphores_.insert(
                this->semaphores_.end(),
                program_on_grid.second.semaphores().begin(),
                program_on_grid.second.semaphores().end());
        }
    }
    return this->semaphores_;
}

std::vector<uint32_t> MeshWorkload::get_program_config_sizes() {
    std::vector<uint32_t> global_program_config_sizes;
    for (auto& program_on_grid : this->programs_) {
        if (global_program_config_sizes.size()) {
            for (int i = 0; i < global_program_config_sizes.size(); i++) {
                TT_FATAL(
                    global_program_config_sizes[i] == program_on_grid.second.get_program_config_sizes()[i],
                    "Expected config sizes to be identical across all programs in a MeshWorkload.");
            }
        } else {
            global_program_config_sizes = program_on_grid.second.get_program_config_sizes();
        }
    }
    return global_program_config_sizes;
}

ProgramBinaryStatus MeshWorkload::get_program_binary_status(std::shared_ptr<MeshDevice> mesh_device) {
    ProgramBinaryStatus bin_status;
    int idx = 0;
    for (auto& program_on_grid : this->programs_) {
        auto grid_start = program_on_grid.first.start_coord;
        Device* device = mesh_device->get_device(grid_start.y, grid_start.x);
        if (idx) {
            TT_FATAL(
                bin_status == program_on_grid.second.get_program_binary_status(device->id()),
                "Expected program binary status to be identical across all programs in the MeshWorkload.");
        } else {
            bin_status = program_on_grid.second.get_program_binary_status(device->id());
        }
        idx++;
    }
    return bin_status;
}

std::unordered_set<SubDeviceId> MeshWorkload::determine_sub_device_ids(std::shared_ptr<MeshDevice> mesh_device) {
    std::unordered_set<SubDeviceId> sub_devices_;
    for (auto& program_on_grid : this->programs_) {
        auto grid_start = program_on_grid.first.start_coord;
        Device* device = mesh_device->get_device(grid_start.y, grid_start.x);
        auto sub_devs_for_program = program_on_grid.second.determine_sub_device_ids(device);
        for (auto& sub_dev : sub_devs_for_program) {
            sub_devices_.insert(sub_dev);
        }
    }
    return sub_devices_;
}

ProgramCommandSequence& MeshWorkload::get_dispatch_cmds_for_program(Program& program) {
    return program.get_cached_program_command_sequences().begin()->second;
}

void MeshWorkload::set_last_used_command_queue(MeshCommandQueue* mesh_cq) { this->last_used_command_queue = mesh_cq; }

ProgramConfig& MeshWorkload::get_program_config(uint32_t index) {
    TT_FATAL(
        this->programs_.size() and this->is_finalized(),
        "Program Configs can only be queried if a MeshWorkload is populated and finalized.");
    return this->programs_.begin()->second.get_program_config(index);
}

uint32_t MeshWorkload::get_cb_base_addr(
    std::shared_ptr<MeshDevice> mesh_device, CoreCoord logical_core, CoreType core_type) {
    auto device = mesh_device->get_device(0);
    CoreCoord virtual_core = device->virtual_core_from_logical_core(logical_core, core_type);
    HalProgrammableCoreType programmable_core_type = device->get_programmable_core_type(virtual_core);
    uint32_t index = hal.get_programmable_core_type_index(programmable_core_type);

    uint32_t base_addr =
        this->last_used_command_queue->get_config_buffer_mgr().get_last_slot_addr(programmable_core_type);
    uint32_t relative_cb_offset = get_program_config(index).cb_offset;
    return base_addr + relative_cb_offset;
}

// void MeshWorkload::set_runtime_args(const LogicalDeviceRange& device_range, const CoreRangeSet& core_range_set,
// KernelHandle kernel_id, const std::vector<uint32_t> runtime_args) {
//     std::size_t intersection_count = 0;

//     for (auto& program_on_grid : this->programs_) {
//         auto& program_device_range = program_on_grid.first;
//         if (device_range.intersects(program_device_range)) {
//             program_to_set_rt
//         }
//     }
// }

}  // namespace tt::tt_metal::distributed
