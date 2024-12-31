// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "mesh_device.hpp"
#include "tt_metal/impl/program/program_dispatch_utils.hpp"
#include "tt_metal/host_api.hpp"

namespace tt::tt_metal::distributed {
using LogicalDeviceRange = CoreRange;
using RuntimeArgsPerCore = std::vector<std::vector<RuntimeArgsData>>;

class MeshCommandQueue;

class MeshWorkload {
    // A MeshWorkload can be fully described using a set of programs mapped to different Logical Device Regions
    // in a Mesh + configurable runtime Args
private:
    bool runs_on_noc_multicast_only_cores();
    bool runs_on_noc_unicast_only_cores();
    void compile(std::shared_ptr<MeshDevice>& mesh_device);
    std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>& get_kernels(uint32_t programmable_core_type_index);
    std::vector<std::shared_ptr<KernelGroup>>& get_kernel_groups(uint32_t programmable_core_type_index);
    std::vector<Semaphore>& semaphores();
    void load_binaries(std::shared_ptr<MeshDevice>& mesh_device, uint8_t cq_id);
    ProgramBinaryStatus get_program_binary_status(std::shared_ptr<MeshDevice> mesh_device);
    std::vector<uint32_t> get_program_config_sizes();
    std::unordered_set<SubDeviceId> determine_sub_device_ids(std::shared_ptr<MeshDevice> mesh_device);
    bool kernel_binary_always_stored_in_ringbuffer();
    ProgramBinaryStatus program_binary_status = ProgramBinaryStatus::NotSent;
    bool compiled_ = false;
    std::unordered_set<std::shared_ptr<Buffer>> kernel_bin_buffers_ = {};
    std::vector<std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>> kernels_ = {};
    std::vector<std::vector<std::shared_ptr<KernelGroup>>> kernel_groups_ = {};
    std::vector<Semaphore> semaphores_ = {};
    std::unordered_map<LogicalDeviceRange, Program> programs_;
    bool finalized_ = false;
    std::unordered_map<LogicalDeviceRange, std::unordered_map<KernelHandle, RuntimeArgsPerCore>> runtime_args_;

    template <typename T>
    friend void program_utils::finalize(T&, Device*);
    friend MeshCommandQueue;

public:
    MeshWorkload();
    void add_program(const LogicalDeviceRange& device_range, Program& program);
    std::unordered_map<LogicalDeviceRange, Program>& get_programs() { return this->programs_; }
    void enqueue(std::shared_ptr<MeshDevice>& mesh_device, uint8_t cq_id, bool blocking);
    bool is_compiled() const { return this->compiled_; }
    bool is_finalized() const { return this->finalized_; }
    void set_finalized() { this->finalized_ = true; };
    ProgramCommandSequence& get_dispatch_cmds_for_program(Program& program);
};
}  // namespace tt::tt_metal::distributed
