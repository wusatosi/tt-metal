// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mesh_coord.hpp>

namespace tt::tt_metal::distributed {

// PhysicalMeshCoordinate is a 2D-coordinate in the physical mesh as defined by the Fabric layer.
// MeshCoordinate[0] is the mesh_id and MeshCoordinate[1] is the physical_device_id.
class PhysicalMeshCoordinate {
public:
    using chip_id_t = uint32_t;
    using mesh_id_t = uint32_t;

    PhysicalMeshCoordinate(const MeshCoordinate& mesh_coordinate) : mesh_coordinate_(mesh_coordinate) {}
    const MeshCoordinate& mesh_coordinate() const { return mesh_coordinate_; }
    mesh_id_t mesh_id() const { return mesh_coordinate_[0]; }
    chip_id_t physical_device_id() const { return mesh_coordinate_[1]; }

private:
    MeshCoordinate mesh_coordinate_;
};

// Returns a map of all physical mesh coordinates in the system.
const MeshContainer<PhysicalMeshCoordinate>& get_system_mesh_coordinate_translation_map();

}  // namespace tt::tt_metal::distributed
