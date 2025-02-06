// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "mesh_buffer.hpp"
#include "trace_buffer.hpp"

namespace tt::tt_metal::distributed {

// Data structure containing MeshTrace staging information
// For each MeshWorkload in the trace, this contains:
//   - The device_range each program in the MeshWorkload runs on
//   - The sysmem_manager the associated dispatch commands are stored in
//   - The offset and size of the dispatch commands in the sysmem_manager
//     staging vector
struct MeshTraceStagingMetadata {
    LogicalDeviceRange device_range = LogicalDeviceRange({0, 0});
    DeviceCoord sysmem_manager_coord = DeviceCoord(0, 0);
    std::size_t offset = 0;
    std::size_t size = 0;
};

// Finalized/Consolidated dispatch commands on a device_range, corresponding
// to a trace
struct MeshTraceData {
    LogicalDeviceRange device_range = LogicalDeviceRange({0, 0});
    std::vector<uint32_t> data = {};
};

// Wrapper around the MeshTraceData. Captures the complete state of a MeshTrace
// (including the dispatch commands across devices, the SubDevices the trace runs on
// the size of the trace and the number of workers in the trace) on host
class MeshTraceDescriptor {
public:
    // Mapping of sub_device_id to descriptor
    std::unordered_map<SubDeviceId, TraceWorkerDescriptor> descriptors;
    // Store the keys of the map in a vector after descriptor has finished being populated
    // This is an optimization since we sometimes need to only pass the keys in a container
    std::vector<SubDeviceId> sub_device_ids;
    // Trace data per logical Device in a Mesh.
    std::vector<MeshTraceData> ordered_trace_data;
    uint32_t total_trace_size = 0;
    // Once the trace is captured/staged inside the sysmem_managers on a MeshDevice, assemble all
    // dispatch commands related to the MeshTrace
    void assemble_dispatch_commands(MeshDevice* device, std::vector<MeshTraceStagingMetadata>& mesh_trace_md);
};

// Ties a MeshTraceDescriptor (host side state) to a MeshBuffer (device side state)
struct MeshTraceBuffer {
    // The trace descriptor associated with a MeshTrace
    std::shared_ptr<MeshTraceDescriptor> desc;
    // The MeshBuffer this trace will be serialized to, before being run on a
    // MeshDevice
    std::shared_ptr<MeshBuffer> mesh_buffer;
};

// Top level class - Manages MeshTrace
class MeshTrace {
private:
    // A unique ID assigned to each Trace
    static std::atomic<uint32_t> global_trace_id;

public:
    // Get global (unique) ID for trace
    static uint32_t next_id();
    // Create an empty MeshTraceBuffer, which needs to be populated
    // with a MeshTraceDescriptor and a MeshBuffer, to get tied to a MeshDevice.
    static std::shared_ptr<MeshTraceBuffer> create_empty_mesh_trace_buffer();
    // Once the Trace Data per logical device has been captured in the
    // MeshTraceDescriptor corresponding to this MeshTraceBuffer,
    // it can be binarized to a MeshDevice through a Command Queue.
    static void populate_mesh_buffer(MeshCommandQueue& mesh_cq, std::shared_ptr<MeshTraceBuffer>& trace_buffer);
};

}  // namespace tt::tt_metal::distributed
