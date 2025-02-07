// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <mesh_buffer.hpp>
#include <mesh_command_queue.hpp>
#include <mesh_workload.hpp>
#include <tt_metal.hpp>

#include "tt_metal/distributed/mesh_workload_utils.hpp"
#include <tt-metalium/distributed.hpp>

namespace tt::tt_metal::distributed {

MeshWorkload::MeshWorkload() {
    // A MeshWorkload tracks maintains its own handles to kernels across all
    // encapsulated programs
    kernel_groups_.resize(hal.get_programmable_core_type_count());
    kernels_.resize(hal.get_programmable_core_type_count());
}

void MeshWorkload::add_program(const LogicalDeviceRange& device_range, Program&& program) {
    // Add a program to a MeshWorkload and tie it a specific logical device range
    programs_[device_range] = std::move(program);
    logical_device_ranges_.push_back(device_range);
}

void MeshWorkload::compile(MeshDevice* mesh_device) {
    // Multi-Step Compile:
    // 1. Compile Kernel Binaries
    // 2. Allocate and Validate CBs
    // 3. Finalize: Compute relative offsets for all data structures in L1
    for (auto& [device_range, program] : programs_) {
        program.compile(mesh_device);
        program.allocate_circular_buffers(mesh_device);
        tt::tt_metal::detail::ValidateCircularBufferRegion(program, mesh_device);
    }
    program_dispatch::finalize_program_offsets(*this, mesh_device);
}

void MeshWorkload::load_binaries(MeshCommandQueue& mesh_cq) {
    // Load binaries for all programs to their respective devices in
    // the Mesh. Only done when the MeshWorkload is enqueued for the first
    // time.
    auto* mesh_device = mesh_cq.device();
    if (program_binary_status_.size()) {
        TT_FATAL(
            program_binary_status_.find(mesh_device->id()) != program_binary_status_.end(),
            "Reusing MeshWorkloads across MeshDevices is currently not supported.");
        TT_FATAL(
            program_binary_status_.at(mesh_device->id()) == ProgramBinaryStatus::Committed,
            "Expected Program Biinaries to be committed to DRAM.");
    } else {
        // Allocate kernel binary buffers of max size across all devices, to ensure we have lock step allocation.
        uint32_t max_kernel_bin_buf_size = 0;
        for (auto& [device_range, program] : programs_) {
            uint32_t curr_kernel_bin_size = program.get_program_transfer_info().binary_data.size() * sizeof(uint32_t);
            max_kernel_bin_buf_size = std::max(max_kernel_bin_buf_size, curr_kernel_bin_size);
        }
        // In production cases, max_kernel_bin_buf_size will always be non-zero (programs have kernels). This check is
        // primarily for test workloads, where a program may not have an attached kernel.
        if (max_kernel_bin_buf_size) {
            // Allocate a MeshBuffer for kernel binaries on each device. This buffer is replicated along the MeshDevice
            // and matches the max kernel binary size across programs.
            DeviceLocalBufferConfig device_local_kernel_bin_buf_config = {
                .page_size = HostMemDeviceCommand::PROGRAM_PAGE_SIZE,
                .buffer_type = BufferType::DRAM,
                .buffer_layout = TensorMemoryLayout::INTERLEAVED,
            };
            ReplicatedBufferConfig global_kernel_bin_buf_config = {
                .size = max_kernel_bin_buf_size,
            };
            kernel_bin_buf_ =
                MeshBuffer::create(global_kernel_bin_buf_config, device_local_kernel_bin_buf_config, mesh_device);
            // Iterate over the sub-grids and EnqueueWriteMeshBuffer to each sub-grid that runs an individual program
            for (auto& [device_range, program] : this->programs_) {
                auto& grid_start = device_range.start_coord;
                std::size_t kernel_bin_size = program.get_program_transfer_info().binary_data.size() * sizeof(uint32_t);
                global_kernel_bin_buf_config.size = kernel_bin_size;
                auto kernel_bin_buf_view = MeshBuffer::create(
                    global_kernel_bin_buf_config,
                    device_local_kernel_bin_buf_config,
                    mesh_device,
                    kernel_bin_buf_->address());

                mesh_device->mesh_command_queue().enqueue_write_shard_to_sub_grid(
                    *kernel_bin_buf_view, program.get_program_transfer_info().binary_data.data(), device_range, false);

                std::shared_ptr<Buffer> buffer_view = Buffer::create(
                    mesh_device,
                    kernel_bin_buf_->address(),
                    kernel_bin_size,
                    HostMemDeviceCommand::PROGRAM_PAGE_SIZE,
                    BufferType::DRAM,
                    TensorMemoryLayout::INTERLEAVED,
                    std::nullopt,
                    false);
                program.set_kernels_bin_buffer(buffer_view);
            }
        }
        program_binary_status_[mesh_device->id()] = ProgramBinaryStatus::InFlight;
    }
}

ProgramBinaryStatus MeshWorkload::get_program_binary_status(std::size_t mesh_id) const {
    if (program_binary_status_.find(mesh_id) != program_binary_status_.end()) {
        return program_binary_status_.at(mesh_id);
    }
    return ProgramBinaryStatus::NotSent;
}

void MeshWorkload::set_program_binary_status(std::size_t mesh_id, ProgramBinaryStatus status) {
    program_binary_status_[mesh_id] = status;
}

void MeshWorkload::generate_dispatch_commands(MeshCommandQueue& mesh_cq) {
    // Generate Dispatch Commands for each Program in the MeshWorkload.
    // These commands will be updated based on MeshDevice state when the
    // workload is enqueued.
    auto mesh_device = mesh_cq.device();
    for (auto& [device_range, program] : programs_) {
        program.generate_dispatch_commands(mesh_device);
    }
}

bool MeshWorkload::runs_on_noc_multicast_only_cores() {
    // Return true if any program in the MeshWorkload runs on cores
    // that can be multicasted to
    bool ret = false;
    for (auto& [device_range, program] : programs_) {
        ret = ret || (program.runs_on_noc_multicast_only_cores());
    }
    return ret;
}

bool MeshWorkload::runs_on_noc_unicast_only_cores() {
    // Return true if any program in the MeshWorkload runs on cores
    // that can only be unicasted to
    bool ret = false;
    for (auto& [device_range, program] : programs_) {
        ret = ret || (program.runs_on_noc_unicast_only_cores());
    }
    return ret;
}

bool MeshWorkload::kernel_binary_always_stored_in_ringbuffer() {
    // Return true if kernel binaries cannot be placed in a ring buffer for
    // any program in the MeshWorkload
    bool stored_in_ring_buf = true;
    for (auto& [device_range, program] : programs_) {
        stored_in_ring_buf &= program.kernel_binary_always_stored_in_ringbuffer();
    }
    return stored_in_ring_buf;
}

std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>& MeshWorkload::get_kernels(
    uint32_t programmable_core_type_index) {
    // Get all kernels across all programs in the MeshWorkload
    if (not kernels_.at(programmable_core_type_index).size()) {
        for (auto& [device_range, program] : programs_) {
            uint32_t device_range_handle = (device_range.start_coord.y << 24) | (device_range.start_coord.x << 16);
            for (const auto& kernel : program.get_kernels(programmable_core_type_index)) {
                KernelHandle handle = (device_range_handle | kernel.first);
                kernels_.at(programmable_core_type_index).insert({handle, kernel.second});
            }
        }
    }
    return kernels_.at(programmable_core_type_index);
}

std::vector<std::shared_ptr<KernelGroup>>& MeshWorkload::get_kernel_groups(uint32_t programmable_core_type_index) {
    // Get all kernel groups across all programs in the MeshWorkload
    if (not kernel_groups_.at(programmable_core_type_index).size()) {
        for (auto& [device_range, program] : programs_) {
            uint32_t device_range_handle = (device_range.start_coord.y << 24) | (device_range.start_coord.x << 16);
            for (auto& kg : program.get_kernel_groups(programmable_core_type_index)) {
                for (auto& optional_kernel_id : kg->kernel_ids) {
                    if (optional_kernel_id.has_value()) {
                        optional_kernel_id = (device_range_handle | optional_kernel_id.value());
                    }
                }
                kernel_groups_.at(programmable_core_type_index).push_back(kg);
            }
        }
    }
    return kernel_groups_.at(programmable_core_type_index);
}

std::vector<Semaphore>& MeshWorkload::semaphores() {
    // Get all semaphores across all programs in the MeshWorkload
    if (not semaphores_.size()) {
        for (auto& [device_range, program] : programs_) {
            semaphores_.insert(semaphores_.end(), program.semaphores().begin(), program.semaphores().end());
        }
    }
    return semaphores_;
}

std::vector<uint32_t> MeshWorkload::get_program_config_sizes() {
    // Get the config sizes for all L1 Program Data Structures
    std::vector<uint32_t> global_program_config_sizes;
    for (auto& program_on_grid : programs_) {
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

std::unordered_set<SubDeviceId> MeshWorkload::determine_sub_device_ids(MeshDevice* mesh_device) {
    // Get the sub device ids for all program across all devices in the Workload
    std::unordered_set<SubDeviceId> sub_devices_;
    for (auto& [device_range, program] : programs_) {
        auto grid_start = device_range.start_coord;
        IDevice* device = mesh_device->get_device(grid_start.y, grid_start.x);
        auto sub_devs_for_program = program.determine_sub_device_ids(mesh_device);
        for (auto& sub_dev : sub_devs_for_program) {
            sub_devices_.insert(sub_dev);
        }
    }
    return sub_devices_;
}

ProgramCommandSequence& MeshWorkload::get_dispatch_cmds_for_program(Program& program) {
    // Get the dispatch commands associated with this program
    return program.get_cached_program_command_sequences().begin()->second;
}

// The functions below are for testing purposes only
void MeshWorkload::set_last_used_command_queue_for_testing(MeshCommandQueue* mesh_cq) {
    last_used_command_queue_ = mesh_cq;
}

MeshCommandQueue* MeshWorkload::get_last_used_command_queue() const { return last_used_command_queue_; }

ProgramConfig& MeshWorkload::get_program_config(uint32_t index) {
    TT_FATAL(
        programs_.size() and is_finalized(),
        "Program Configs can only be queried if a MeshWorkload is populated and finalized.");
    return programs_.begin()->second.get_program_config(index);
}

uint32_t MeshWorkload::get_sem_base_addr(
    std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type) {
    HalProgrammableCoreType programmable_core_type =
        ::tt::tt_metal::detail::hal_programmable_core_type_from_core_type(core_type);
    uint32_t base_addr = program_dispatch::program_base_addr_on_core(*this, mesh_device.get(), programmable_core_type);
    return base_addr + get_program_config(hal.get_programmable_core_type_index(programmable_core_type)).sem_offset;
}

uint32_t MeshWorkload::get_sem_size(
    std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type) {
    uint32_t sem_size = 0;
    uint32_t program_idx = 0;
    IDevice* device = mesh_device->get_device_index(0);
    for (auto& [device_range, program] : programs_) {
        if (program_idx) {
            TT_ASSERT(sem_size == program.get_sem_size(device, logical_core, core_type));
        } else {
            sem_size = program.get_sem_size(device, logical_core, core_type);
        }
        program_idx++;
    }
    return sem_size;
}

uint32_t MeshWorkload::get_cb_base_addr(
    std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type) {
    HalProgrammableCoreType programmable_core_type =
        ::tt::tt_metal::detail::hal_programmable_core_type_from_core_type(core_type);
    uint32_t base_addr = program_dispatch::program_base_addr_on_core(*this, mesh_device.get(), programmable_core_type);
    return base_addr + get_program_config(hal.get_programmable_core_type_index(programmable_core_type)).cb_offset;
}

uint32_t MeshWorkload::get_cb_size(
    std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type) {
    uint32_t cb_size = 0;
    uint32_t program_idx = 0;
    IDevice* device = mesh_device->get_device_index(0);
    for (auto& [device_range, program] : programs_) {
        if (program_idx) {
            TT_ASSERT(cb_size == program.get_cb_size(device, logical_core, core_type));
        } else {
            cb_size = program.get_cb_size(device, logical_core, core_type);
        }
        program_idx++;
    }
    return cb_size;
}

std::atomic<uint32_t> MeshTrace::global_trace_id = 0;

uint32_t MeshTrace::next_id() { return global_trace_id++; }

std::shared_ptr<MeshTraceBuffer> MeshTrace::create_empty_mesh_trace_buffer() {
    return std::make_shared<MeshTraceBuffer>(std::make_shared<MeshTraceDescriptor>(), nullptr);
}

static constexpr uint32_t kExecBufPageMin = 1024;
static constexpr uint32_t kExecBufPageMax = 4096;

static constexpr bool kBlocking = true;
static constexpr bool kNonBlocking = false;

// Assumes pages are interleaved across all banks starting at 0
size_t interleaved_page_size(
    const uint32_t buf_size, const uint32_t num_banks, const uint32_t min_size, const uint32_t max_size) {
    // Populate power of 2 numbers within min and max as candidates
    TT_FATAL(
        min_size > 0 and min_size <= max_size,
        "min_size {} not positive and less than or equal to max_size {}.",
        min_size,
        max_size);
    std::vector<uint32_t> candidates;
    candidates.reserve(__builtin_clz(min_size) - __builtin_clz(max_size) + 1);
    for (uint32_t size = 1; size <= max_size; size <<= 1) {
        if (size >= min_size) {
            candidates.push_back(size);
        }
    }
    uint32_t min_waste = -1;
    uint32_t pick = 0;
    // Pick the largest size that minimizes waste
    for (const uint32_t size : candidates) {
        // Pad data to the next fully banked size
        uint32_t fully_banked = num_banks * size;
        uint32_t padded_size = (buf_size + fully_banked - 1) / fully_banked * fully_banked;
        uint32_t waste = padded_size - buf_size;
        if (waste <= min_waste) {
            min_waste = waste;
            pick = size;
        }
    }
    TT_FATAL(
        pick >= min_size and pick <= max_size,
        "pick {} not between min_size {} and max_size {}",
        pick,
        min_size,
        max_size);
    return pick;
}

void MeshTrace::populate_mesh_buffer(MeshCommandQueue& mesh_cq, std::shared_ptr<MeshTraceBuffer> trace_buffer) {
    auto mesh_device = mesh_cq.device();
    uint64_t unpadded_size = trace_buffer->desc->total_trace_size;
    size_t page_size = interleaved_page_size(
        unpadded_size,
        mesh_cq.device()->allocator()->get_num_banks(BufferType::DRAM),
        kExecBufPageMin,
        kExecBufPageMax);
    size_t padded_size = round_up(unpadded_size, page_size);

    const auto current_trace_buffers_size = mesh_cq.device()->get_trace_buffers_size();
    mesh_cq.device()->set_trace_buffers_size(current_trace_buffers_size + padded_size);
    auto trace_region_size = mesh_cq.device()->allocator()->get_config().trace_region_size;
    TT_FATAL(
        mesh_cq.device()->get_trace_buffers_size() <= trace_region_size,
        "Creating trace buffers of size {}B on MeshDevice {}, but only {}B is allocated for trace region.",
        mesh_cq.device()->get_trace_buffers_size(),
        mesh_cq.device()->id(),
        trace_region_size);

    DeviceLocalBufferConfig device_local_trace_buf_config = {
        .page_size = page_size,
        .buffer_type = BufferType::TRACE,
        .buffer_layout = TensorMemoryLayout::INTERLEAVED,
    };

    ReplicatedBufferConfig global_trace_buf_config = {
        .size = padded_size,
    };

    trace_buffer->mesh_buffer =
        MeshBuffer::create(global_trace_buf_config, device_local_trace_buf_config, mesh_cq.device());

    std::unordered_map<LogicalDeviceRange, uint32_t> write_offset_per_device_range = {};
    for (auto& mesh_trace_data : trace_buffer->desc->ordered_trace_data) {
        auto& device_range = mesh_trace_data.device_range;
        if (write_offset_per_device_range.find(device_range) == write_offset_per_device_range.end()) {
            write_offset_per_device_range.insert({device_range, 0});
        }
        std::vector<uint32_t> write_data = mesh_trace_data.data;
        auto unpadded_data_size = write_data.size() * sizeof(uint32_t);
        auto padded_data_size = round_up(unpadded_size, page_size);
        size_t numel_padding = (padded_data_size - unpadded_data_size) / sizeof(uint32_t);
        if (numel_padding > 0) {
            write_data.resize(write_data.size() + numel_padding, 0);
        }

        auto write_region =
            BufferRegion(write_offset_per_device_range.at(device_range), write_data.size() * sizeof(uint32_t));
        mesh_cq.enqueue_write_shard_to_sub_grid(
            *(trace_buffer->mesh_buffer), write_data.data(), device_range, kBlocking, write_region);
        write_offset_per_device_range.at(device_range) += mesh_trace_data.data.size() * sizeof(uint32_t);
    }
    // auto bcast_device_range = LogicalDeviceRange({0, 0}, {mesh_device->num_cols() - 1, mesh_device->num_rows() - 1});
    // std::vector<uint32_t>& trace_data = trace_buffer->desc->ordered_trace_data[bcast_device_range];
    // uint64_t unpadded_size = trace_data.size() * sizeof(uint32_t);
    // size_t page_size = interleaved_page_size(
    //     unpadded_size,
    //     mesh_cq.device()->allocator()->get_num_banks(BufferType::DRAM),
    //     kExecBufPageMin,
    //     kExecBufPageMax);
    // uint64_t padded_size = round_up(unpadded_size, page_size);
    // size_t numel_padding = (padded_size - unpadded_size) / sizeof(uint32_t);
    // if (numel_padding > 0) {
    //     trace_data.resize(trace_data.size() + numel_padding, 0 /*padding value*/);
    // }
    // const auto current_trace_buffers_size = mesh_cq.device()->get_trace_buffers_size();
    // mesh_cq.device()->set_trace_buffers_size(current_trace_buffers_size + padded_size);
    // auto trace_region_size = mesh_cq.device()->allocator()->get_config().trace_region_size;
    // TT_FATAL(
    //     mesh_cq.device()->get_trace_buffers_size() <= trace_region_size,
    //     "Creating trace buffers of size {}B on MeshDevice {}, but only {}B is allocated for trace region.",
    //     mesh_cq.device()->get_trace_buffers_size(),
    //     mesh_cq.device()->id(),
    //     trace_region_size);

    // DeviceLocalBufferConfig device_local_trace_buf_config = {
    //     .page_size = page_size,
    //     .buffer_type = BufferType::TRACE,
    //     .buffer_layout = TensorMemoryLayout::INTERLEAVED,
    // };

    // ReplicatedBufferConfig global_trace_buf_config = {
    //     .size = padded_size,
    // };

    // // Commit trace to device DRAM
    // trace_buffer->mesh_buffer =
    //     MeshBuffer::create(global_trace_buf_config, device_local_trace_buf_config, mesh_cq.device());
    // EnqueueWriteMeshBuffer(mesh_cq, trace_buffer->mesh_buffer, trace_data, kBlocking);
}

}  // namespace tt::tt_metal::distributed
