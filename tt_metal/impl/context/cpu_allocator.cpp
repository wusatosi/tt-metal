// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cpu_allocator.hpp"
#include "impl/context/metal_context.hpp"

#include <numa.h>
#include <unistd.h>  // Warning Linux Only, needed for _SC_NPROCESSORS_ONLN

namespace tt::tt_metal {

CpuAllocator::CpuAllocator(
    const std::unordered_set<chip_id_t>& device_ids,
    bool use_numa_node_based_thread_binding,
    const uint8_t num_hw_cqs) {
    if (use_numa_node_based_thread_binding) {
        parse_cpu_cores();
    }
    populate_cpu_core_assignments(device_ids, use_numa_node_based_thread_binding, num_hw_cqs);
    if (use_numa_node_based_thread_binding) {
        bind_current_thread_to_free_cores();
    }
}

void CpuAllocator::parse_cpu_cores() {
    if (numa_available() != -1) {
        // Host has NUMA enabled. Group CPU IDs by the NUMA nodes they belong to.
        for (int cpu = 0; cpu < numa_num_configured_cpus(); ++cpu) {
            int node = numa_node_of_cpu(cpu);
            if (numa_node_to_cpu_cores_.find(node) == numa_node_to_cpu_cores_.end()) {
                numa_node_to_cpu_cores_.insert({node, {}});
            }
            free_cores_.insert(cpu);
            numa_node_to_cpu_cores_.at(node).push_back(cpu);
        }
    } else {
        // Host does not have NUMA. Place all CPU Ids under a single node (0).
        log_warning(tt::LogMetal, "Host does not use NUMA. May see reduced performance.");
        for (int cpu = 0; cpu < sysconf(_SC_NPROCESSORS_ONLN); ++cpu) {
            free_cores_.insert(cpu);
        }
    }
}

std::pair<int, int> CpuAllocator::get_cpu_cores_for_dispatch_threads(
    int mmio_controlled_device_id, uint32_t num_devices, bool use_separate_procs) {
    int core_assigned_to_device_worker_thread = 0;
    int core_assigned_to_device_completion_queue_reader = 0;
    uint32_t num_online_processors = sysconf(_SC_NPROCESSORS_ONLN);
    // Get NUMA node that the current device is mapped to through UMD
    int numa_node_for_device =
        tt::tt_metal::MetalContext::instance().get_cluster().get_numa_node_for_device(mmio_controlled_device_id);

    if (numa_available() != -1 and
        numa_node_to_cpu_cores_.find(numa_node_for_device) != numa_node_to_cpu_cores_.end()) {
        // NUMA node reported by UMD exists on host. Choose a core on this numa-node using round robin policy
        const auto& cpu_core_for_numa_node = numa_node_to_cpu_cores_.at(numa_node_for_device);
        int num_cores_in_numa_node = cpu_core_for_numa_node.size();
        core_assigned_to_device_worker_thread =
            cpu_core_for_numa_node.at(mmio_controlled_device_id % num_cores_in_numa_node);
        if (use_separate_procs) {
            core_assigned_to_device_completion_queue_reader =
                cpu_core_for_numa_node.at((mmio_controlled_device_id + num_devices) % num_cores_in_numa_node);
        } else {
            core_assigned_to_device_completion_queue_reader = core_assigned_to_device_worker_thread;
        }
    } else {
        // NUMA node reported by UMD does not exist on host. Use round-robin binding policy for this worker thread.
        log_warning(
            tt::LogMetal,
            "NUMA node {} for device {} does not exist on host or NUMA is not available.",
            numa_node_for_device,
            mmio_controlled_device_id);
        core_assigned_to_device_worker_thread = mmio_controlled_device_id % num_online_processors;
        if (use_separate_procs) {
            core_assigned_to_device_completion_queue_reader =
                (mmio_controlled_device_id + num_devices) % num_online_processors;
        } else {
            core_assigned_to_device_completion_queue_reader = core_assigned_to_device_worker_thread;
        }
    }

    free_cores_.erase(core_assigned_to_device_worker_thread);
    if (use_separate_procs) {
        free_cores_.erase(core_assigned_to_device_completion_queue_reader);
    }
    return std::make_pair(core_assigned_to_device_worker_thread, core_assigned_to_device_completion_queue_reader);
}

void CpuAllocator::populate_cpu_core_assignments(
    const std::unordered_set<chip_id_t>& device_ids,
    bool use_numa_node_based_thread_binding,
    const uint8_t num_hw_cqs) {
    uint32_t num_online_processors = sysconf(_SC_NPROCESSORS_ONLN);
    constexpr uint32_t max_num_procs_per_device = 2;
    // When using multiple command queues, assign separate CPU cores to worker and completion queue reader threads,
    // if enough processors exist on host. Atleast one core is given to the main thread.
    bool separate_procs_for_worker_and_reader =
        (num_hw_cqs > 1) && (max_num_procs_per_device * device_ids.size() <= num_online_processors - 1);
    if (use_numa_node_based_thread_binding) {
        for (const auto& device_id : device_ids) {
            auto [worker_thread_core, completion_queue_reader_core] =
                get_cpu_cores_for_dispatch_threads(device_id, device_ids.size(), separate_procs_for_worker_and_reader);
            device_id_to_worker_thread_cpu_core_.insert({device_id, worker_thread_core});
            device_id_to_cq_reader_cpu_core_.insert({device_id, completion_queue_reader_core});
        }
    } else {
        // Round Robin CPU assignment for worker and completion queue reader threads
        for (const auto& device_id : device_ids) {
            uint32_t worker_thread_proc = device_id % num_online_processors;
            device_id_to_worker_thread_cpu_core_.insert({device_id, worker_thread_proc});
            if (separate_procs_for_worker_and_reader) {
                uint32_t completion_queue_reader_proc = (device_id + device_ids.size()) % num_online_processors;
                device_id_to_cq_reader_cpu_core_.insert({device_id, completion_queue_reader_proc});
            } else {
                device_id_to_cq_reader_cpu_core_.insert({device_id, worker_thread_proc});
            }
        }
    }
}

void CpuAllocator::bind_current_thread_to_free_cores() {
    cpu_set_t cpuset;
    pthread_t current_thread = pthread_self();
    CPU_ZERO(&cpuset);

    for (const auto& free_core : free_cores_) {
        CPU_SET(free_core, &cpuset);
    }
    int rc = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    if (rc) {
        log_warning(
            tt::LogMetal,
            "Unable to bind main thread to free CPU cores. May see performance degradation. Error Code: {}",
            rc);
    }
}

}  // namespace tt::tt_metal
