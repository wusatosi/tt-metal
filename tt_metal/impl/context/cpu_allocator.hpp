// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/core_descriptor.hpp>

#include <unordered_set>
#include <unordered_map>

namespace tt::tt_metal {

class CpuAllocator {
public:
    CpuAllocator(
        const std::unordered_set<chip_id_t>& device_ids,
        bool use_numa_node_based_thread_binding,
        const uint8_t num_hw_cqs);

    uint32_t worker_thread_cpu_core(chip_id_t device_id) { return device_id_to_worker_thread_cpu_core_.at(device_id); };
    uint32_t cq_reader_cpu_core(chip_id_t device_id) { return device_id_to_cq_reader_cpu_core_.at(device_id); };

private:
    // Read the available cpu cores on device and populate free_cores_ and num_node_to_num_cpu_cores_.
    void parse_cpu_cores();

    // Assign cpu cores for dispatch threads, first core returned is for worker thread, second core returned is for
    // completion queue reader.
    std::pair<int, int> get_cpu_cores_for_dispatch_threads(
        int mmio_controlled_device_id, uint32_t num_devices, bool use_separate_procs);

    // Populate maps from device id to cpu core
    void populate_cpu_core_assignments(
        const std::unordered_set<chip_id_t>& device_ids,
        bool use_numa_node_based_thread_binding,
        const uint8_t num_hw_cqs);

    // Bind any remaining free cores to the main thread
    void bind_current_thread_to_free_cores();

    // Determine which CPU cores the worker threads need to be placed on for each device
    std::unordered_map<chip_id_t, uint32_t> device_id_to_worker_thread_cpu_core_;
    std::unordered_map<chip_id_t, uint32_t> device_id_to_cq_reader_cpu_core_;

    // Collection of free cpu cores, as well as cpu cores per numa node
    std::unordered_set<uint32_t> free_cores_;
    std::unordered_map<int, std::vector<uint32_t>> numa_node_to_cpu_cores_;

    bool use_numa_node_based_thread_binding_ = false;
};

}  // namespace tt::tt_metal
