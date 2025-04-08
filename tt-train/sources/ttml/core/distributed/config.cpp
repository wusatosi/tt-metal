#include "config.hpp"

namespace ttml::core::distributed {

namespace {

const std::string kEnableTensorParallel = "enable_tensor_parallel";
const std::string kEnableDDP = "enable_ddp";
const std::string kDevicesPerTP = "devices_per_tp";
const std::string kDevicesPerDDP = "devices_per_ddp";

}  // namespace

DistributedConfig::DistributedConfig(const YAML::Node& yaml_config) {
    enable_ddp = yaml_config[kEnableDDP].as<bool>(enable_ddp);
    enable_tp = yaml_config[kEnableTensorParallel].as<bool>(enable_tp);
    devices_per_tp = yaml_config[kDevicesPerTP].as<uint32_t>(devices_per_tp);
    devices_per_ddp = yaml_config[kDevicesPerDDP].as<uint32_t>(devices_per_ddp);
};

uint32_t DistributedConfig::get_devices_per_tp() const {
    return devices_per_tp;
}

uint32_t DistributedConfig::get_devices_per_ddp() const {
    return devices_per_ddp;
}

bool DistributedConfig::is_ddp() const {
    return enable_ddp;
}

bool DistributedConfig::is_tensor_parallel() const {
    return enable_tp;
}

tt::tt_metal::distributed::MeshShape DistributedConfig::get_mesh_shape() const {
    if (enable_tp && enable_ddp) {
        return tt::tt_metal::distributed::MeshShape(devices_per_ddp, devices_per_tp);
    } else if (enable_tp) {
        return tt::tt_metal::distributed::MeshShape(1, devices_per_tp);
    } else if (enable_ddp) {
        return tt::tt_metal::distributed::MeshShape(devices_per_ddp, 1);
    }
    return tt::tt_metal::distributed::MeshShape(1, 1);
}

};  // namespace ttml::core::distributed
