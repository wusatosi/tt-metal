// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/impl/buffers/semaphore.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"

namespace tt {
namespace tt_metal {

// Represents the status of Program Kernel Binaries in Device DRAM with respect to the dispatcher
enum class ProgramBinaryStatus : uint8_t {
    NotSent = 0, // Binaries have not been written
    InFlight = 1, // Fast Dispatch Commands to write the binaries to DRAM has been issued
    Committed = 2, // Binaries have been commited to DRAM
};

// Contains the program's worker memory map
struct ProgramConfig {
    uint32_t rta_offset;
    std::array<uint32_t, DISPATCH_CLASS_MAX> crta_offsets;
    std::array<uint32_t, DISPATCH_CLASS_MAX> crta_sizes;
    uint32_t sem_offset;
    uint32_t sem_size;
    uint32_t cb_offset;
    uint32_t cb_size;
    uint32_t local_cb_size;
    uint32_t kernel_text_offset; // offset of first kernel bin
    uint32_t kernel_text_size;   // max size of all kernel bins across all kernel groups
};

inline namespace v0 {
    class Program;
}


class Workload;

namespace program_dispatch {
    template<typename T>
    void finalize_program_offsets(Workload<T>& workload_type, Device* device);
}
struct KernelGroup;

// The LogicalDeviceRange concept is fundamentally identical to the CoreRange concept
// Use this definition for now, since CoreRange contains several utility functions required
// in the MeshWorkload context. CoreRange can eventually be renamed to Range2D.
using LogicalDeviceRange = CoreRange;

template<typename GenericDevice>
class Workload {
    public:
     Workload() = default;

     Workload(const Workload &other) = delete;
     Workload& operator=(const Workload &other) = delete;

     Workload(Workload &&other) noexcept = default;
     Workload& operator=(Workload &&other) noexcept = default;
     virtual uint32_t get_sem_base_addr(GenericDevice device, CoreCoord logical_core, CoreType core_type) = 0;
     virtual uint32_t get_sem_size(GenericDevice device, CoreCoord logical_core, CoreType core_type) const = 0;
     virtual uint32_t get_cb_base_addr(GenericDevice device, CoreCoord logical_core, CoreType core_type) = 0;
     virtual uint32_t get_cb_size(GenericDevice device, CoreCoord logical_core, CoreType core_type) const = 0;
     virtual const std::vector<LogicalDeviceRange> get_logical_device_ranges() const { return {LogicalDeviceRange({0, 0}, {1, 1})}; }
     virtual Program& get_program_on_device_range(const LogicalDeviceRange& device_range) = 0;

    protected:
     virtual bool runs_on_noc_multicast_only_cores() = 0;
     virtual bool runs_on_noc_unicast_only_cores() = 0;
     virtual void generate_dispatch_commands(GenericDevice device) = 0;
     virtual std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>& get_kernels(uint32_t programmable_core_type_index) = 0;
     virtual std::vector<std::shared_ptr<KernelGroup>>& get_kernel_groups(uint32_t programmable_core_type_index) = 0;
     virtual const std::vector<Semaphore>& semaphores() const = 0;
     virtual std::vector<uint32_t>& get_program_config_sizes() const = 0;
     virtual bool kernel_binary_always_stored_in_ringbuffer() = 0;
     virtual bool is_finalized() const = 0;
     virtual void set_finalized() = 0;
     virtual ProgramBinaryStatus get_program_binary_status(std::size_t device_id) const = 0;
     virtual void set_program_binary_status(std::size_t mesh_id, ProgramBinaryStatus status) = 0;
     virtual ProgramConfig& get_program_config(uint32_t index) = 0;
     template <typename T>
     friend void program_dispatch::finalize_program_offsets(Workload<T>&, Device*);
};

} // namespace tt
} // namespace tt_metal