#pragma once

#include <core/ttnn_all_includes.hpp>

namespace ttml::core::distributed {

class DistributedConfig {
public:
    DistributedConfig() = default;
    DistributedConfig(const YAML::Node& yaml_config);

    // default copy constructors
    DistributedConfig(const DistributedConfig&) = default;
    DistributedConfig& operator=(const DistributedConfig&) = default;

    // default move constructors
    DistributedConfig(DistributedConfig&&) = default;
    DistributedConfig& operator=(DistributedConfig&&) = default;

    [[nodiscard]] uint32_t get_devices_per_tp() const;
    [[nodiscard]] uint32_t get_devices_per_ddp() const;

    [[nodiscard]] bool is_ddp() const;
    [[nodiscard]] bool is_tensor_parallel() const;

    [[nodiscard]] tt::tt_metal::distributed::MeshShape get_mesh_shape() const;

private:
    bool enable_tp = false;
    bool enable_ddp = false;

    uint32_t devices_per_tp = 0;
    uint32_t devices_per_ddp = 0;
};

};  // namespace ttml::core::distributed
