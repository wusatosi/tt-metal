// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/coordinate_translation.hpp"
#include "tt_metal/api/tt-metalium/tt_metal.hpp"

#include "indestructible.hpp"
namespace tt::tt_metal::distributed {

const MeshContainer<PhysicalMeshCoordinate>& get_system_mesh_coordinate_translation_map() {
    static tt::stl::Indestructible<MeshContainer<PhysicalMeshCoordinate>> kTranslationMap([]() {
        tt::tt_metal::detail::InitializeFabricConfig(tt::FabricConfig::DISABLED);
        const auto* control_plane = tt::Cluster::instance().get_control_plane();
        TT_FATAL(control_plane != nullptr, "Control plane must be initialized before MeshDevice can be created.");

        auto num_physical_meshes = control_plane->get_user_physical_mesh_ids().size();
        if (num_physical_meshes >= 1) {
            tt::log_warning(LogMetal, "Only one user physical mesh is supported, using the first one");
        }

        for (const auto mesh_id : control_plane->get_user_physical_mesh_ids()) {
            auto mesh_shape = control_plane->get_physical_mesh_shape(mesh_id);
            MeshContainer<PhysicalMeshCoordinate> physical_coordinates(
                mesh_shape, PhysicalMeshCoordinate(MeshCoordinate::zero_coordinate(mesh_shape.dims())));
            for (int logical_chip_id = 0; logical_chip_id < mesh_shape.mesh_size(); ++logical_chip_id) {
                // Query the control plane to get the physical chip id from logical chip id
                auto logical_row_idx = logical_chip_id / mesh_shape[1];
                auto logical_col_idx = logical_chip_id % mesh_shape[1];
                auto logical_coordinate = MeshCoordinate(logical_row_idx, logical_col_idx);
                auto physical_chip_id =
                    control_plane->get_physical_chip_id_from_mesh_chip_id({mesh_id, logical_chip_id});

                auto physical_coordinate = PhysicalMeshCoordinate(MeshCoordinate(mesh_id, physical_chip_id));
                physical_coordinates.at(logical_coordinate) = physical_coordinate;
            }
            return physical_coordinates;
        }
        TT_THROW("There are no user physical meshes in the system found by control plane.");
    }());
    return kTranslationMap.get();
}

}  // namespace tt::tt_metal::distributed
