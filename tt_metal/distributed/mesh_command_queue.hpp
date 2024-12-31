// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "mesh_device.hpp"
#include "mesh_workload.hpp"

namespace tt::tt_metal::distributed {

class MeshCommandQueue {
private:
    tt::tt_metal::WorkerConfigBufferMgr config_buffer_mgr_;
    LaunchMessageRingBufferState worker_launch_message_buffer_state_;
    uint32_t expected_num_workers_completed_ = 0;
    std::shared_ptr<MeshDevice> mesh_device_;
    uint32_t id_;

public:
    MeshCommandQueue(std::shared_ptr<MeshDevice> mesh_device, uint32_t id);
    const std::shared_ptr<MeshDevice>& mesh_device() { return this->mesh_device_; }
    uint32_t id() { return this->id_; }
    void enqueue_mesh_workload(MeshWorkload& mesh_workload, bool blocking);
    void finish();
};

}  // namespace tt::tt_metal::distributed
