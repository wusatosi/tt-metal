// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <optional>

#include <tt-metalium/mesh_coord.hpp>

namespace tt::tt_metal::distributed {

using chip_id_t = int;

// Specifies the configuration of a MeshDevice.
class MeshDeviceConfig {
public:
    // Constructs a MeshDeviceConfig.
    // `offset` is the optional parameter that specifies the offset of the mesh device within the connected system mesh.
    // `physical_device_ids` is the optional parameter that allows to override physical device IDs used to create the
    // mesh device.
    MeshDeviceConfig(
        const MeshShape& mesh_shape,
        const std::optional<MeshCoordinate>& offset = std::nullopt,
        const std::vector<chip_id_t>& physical_device_ids = {},
        std::optional<MeshShape> local_mesh_shape = std::nullopt) :
        mesh_shape_(mesh_shape), offset_(offset), physical_device_ids_(physical_device_ids), local_mesh_shape_(local_mesh_shape.value_or(mesh_shape)) {}

    const MeshShape& mesh_shape() const { return mesh_shape_; }
    const std::optional<MeshCoordinate>& offset() const { return offset_; }
    const std::vector<chip_id_t>& physical_device_ids() const { return physical_device_ids_; }
    const MeshShape& local_mesh_shape() const { return local_mesh_shape_; }

private:
    MeshShape mesh_shape_;
    MeshShape local_mesh_shape_;

    std::optional<MeshCoordinate> offset_;
    std::vector<chip_id_t> physical_device_ids_;
};

}  // namespace tt::tt_metal::distributed
