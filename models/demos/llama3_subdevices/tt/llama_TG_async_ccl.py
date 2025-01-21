# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import ttnn
from tests.ttnn.unit_tests.operations.ccl.test_ccl_common import (
    create_and_load_sub_device_manager_with_fabric_interface,
    teardown_fabric_interface,
    create_global_semaphore_with_same_address,
)


class CCLConfig:
    def __init__(
        self, mesh_device, enable_persistent_fabric=True, create_persistent_fabric=True, teardown_persistent_fabric=True
    ):
        if create_persistent_fabric:
            assert enable_persistent_fabric
        if teardown_persistent_fabric:
            assert enable_persistent_fabric

        self.mesh_device = mesh_device
        self.enable_persistent_fabric = enable_persistent_fabric
        self.teardown_persistent_fabric = teardown_persistent_fabric

        for d in mesh_device.get_devices():
            ttnn.enable_program_cache(d)

        sub_device_stall_group = []
        ccl_sub_device_crs = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
            ]
        )
        worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
        worker_sub_device_id = ttnn.SubDeviceId(0)
        self.worker_sub_device_id = worker_sub_device_id
        if create_persistent_fabric:
            logger.info("Create persistent fabric interface")
            self.mesh_sub_device_manager_id = create_and_load_sub_device_manager_with_fabric_interface(
                mesh_device, [worker_sub_device], 0, 0, enable_persistent_fabric
            )
            logger.info("Done Create persistent fabric interface")
            self.sub_device_stall_group = [worker_sub_device_id]
        mesh_device.set_sub_device_stall_group(self.sub_device_stall_group)

        self.all_gather_ccl_semaphore_handles = create_global_semaphore_with_same_address(
            mesh_device, ccl_sub_device_crs, 0
        )
        self.all_reduce_from_remote_semaphore_handles = create_global_semaphore_with_same_address(
            mesh_device, ccl_sub_device_crs, 1
        )
        self.all_reduce_to_remote_semaphore_handles = create_global_semaphore_with_same_address(
            mesh_device, ccl_sub_device_crs, 2
        )
        self.all_reduce_gather_semaphore_handles = create_global_semaphore_with_same_address(
            mesh_device, ccl_sub_device_crs, 3
        )
        self.reduce_scatter_from_remote_semaphore_handles = create_global_semaphore_with_same_address(
            mesh_device, ccl_sub_device_crs, 4
        )
        self.reduce_scatter_to_remote_semaphore_handles = create_global_semaphore_with_same_address(
            mesh_device, ccl_sub_device_crs, 5
        )

    def all_gather(self, tensor, output_mem_config, cluster_axis, dim=0, num_links=1):
        ttnn_tensor_out = ttnn.experimental.all_gather_async(
            tensor,
            dim,
            cluster_axis=cluster_axis,
            mesh_device=self.mesh_device,
            topology=ttnn.Topology.Linear,
            multi_device_global_semaphore=self.all_gather_ccl_semaphore_handles,
            num_links=num_links,
            memory_config=output_mem_config,
            subdevice_id=self.worker_sub_device_id,
            enable_persistent_fabric_mode=self.enable_persistent_fabric,
        )

        if self.enable_persistent_fabric:
            ttnn.synchronize_devices(self.mesh_device, sub_device_ids=self.sub_device_stall_group)
        return ttnn_tensor_out

    def all_reduce(self, tensor, output_mem_config, cluster_axis, num_links=1, math_op=ttnn.ReduceType.Sum):
        ttnn_tensor_out = ttnn.experimental.all_reduce_async(
            tensor,
            cluster_axis=cluster_axis,
            mesh_device=self.mesh_device,
            from_remote_multi_device_global_semaphore=self.all_reduce_from_remote_semaphore_handles,
            to_remote_multi_device_global_semaphore=self.all_reduce_to_remote_semaphore_handles,
            gather_multi_device_global_semaphore=self.all_reduce_gather_semaphore_handles,
            math_op=math_op,
            num_links=num_links,
            memory_config=output_mem_config,
            topology=ttnn.Topology.Linear,
            subdevice_id=self.worker_sub_device_id,
        )
        if self.enable_persistent_fabric:
            ttnn.synchronize_devices(self.mesh_device, sub_device_ids=self.sub_device_stall_group)
        return ttnn_tensor_out

    def reduce_scatter(self, tensor, output_mem_config, dim, cluster_axis, num_links=1, math_op=ttnn.ReduceType.Sum):
        ttnn_tensor_out = ttnn.experimental.reduce_scatter_async(
            tensor,
            dim=dim,
            cluster_axis=cluster_axis,
            mesh_device=self.mesh_device,
            from_remote_multi_device_global_semaphore=self.reduce_scatter_from_remote_semaphore_handles,
            to_remote_multi_device_global_semaphore=self.reduce_scatter_to_remote_semaphore_handles,
            math_op=math_op,
            memory_config=output_mem_config,
            topology=ttnn.Topology.Linear,
            num_links=num_links,
            subdevice_id=self.worker_sub_device_id,
        )
        if self.enable_persistent_fabric:
            ttnn.synchronize_devices(self.mesh_device, sub_device_ids=self.sub_device_stall_group)
        return ttnn_tensor_out

    def teardown_persistent_fabric(self):
        if self.enable_persistent_fabric and self.teardown_persistent_fabric:
            logger.info("Tearing down persistent fabric interface")
            self.mesh_device.reset_sub_device_stall_group()
            teardown_fabric_interface(mesh_device)
            logger.info("Done tearing down persistent fabric interface")
