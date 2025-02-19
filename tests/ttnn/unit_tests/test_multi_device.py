# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import typing
import pytest
import ttnn
import tempfile
from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc


from ttnn import ShardTensorToMesh, ReplicateTensorToMesh, ConcatMeshToTensor, ListMeshToTensor


#######
# Test MultiDevice Initialization, Open/Close
#######
def test_mesh_device_open_close_explicit(silicon_arch_name, silicon_arch_wormhole_b0):
    """Manually open and close multi-device"""
    num_pcie_devices = ttnn.get_num_pcie_devices()
    if num_pcie_devices <= 1:
        pytest.skip("Requires multiple devices to run")

    multi_device = ttnn.open_mesh_device(ttnn.MeshShape(2, 2))
    ttnn.close_mesh_device(multi_device)


def test_multi_device_subset_mesh(silicon_arch_name, silicon_arch_wormhole_b0):
    """Manually open and close multi-device"""
    num_pcie_devices = ttnn.get_num_pcie_devices()
    if num_pcie_devices <= 1:
        pytest.skip("Requires multiple devices to run")

    mesh_shape = ttnn.MeshShape(1, 2)
    multi_device = ttnn.open_mesh_device(mesh_shape)
    assert multi_device.get_num_devices() == 2
    ttnn.close_mesh_device(multi_device)

    multi_device = ttnn.open_mesh_device(mesh_shape)
    assert multi_device.get_num_devices() == 2
    ttnn.close_mesh_device(multi_device)


@pytest.mark.parametrize(
    "device_params",
    [
        {"dispatch_core_axis": ttnn.DispatchCoreAxis.ROW, "num_command_queues": 1},
        {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "num_command_queues": 1},
        {"dispatch_core_axis": ttnn.DispatchCoreAxis.ROW, "num_command_queues": 2},
        {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "num_command_queues": 2},
    ],
    indirect=True,
)
def test_multi_device_open_close_full_mesh_device_fixture(mesh_device):
    """Using `mesh_device` pytest fixture defined in conftest.py"""
    pass


def test_multi_device_open_close_using_context_manager(silicon_arch_name, silicon_arch_wormhole_b0):
    """Using context manager to open and close multi-device"""
    if ttnn.get_num_devices() < 4:
        pytest.skip()
    mesh_shape = ttnn.MeshShape(2, 2)
    with ttnn.create_mesh_device(mesh_shape) as mesh_device:
        # Do something with multi_device
        pass


def test_multi_device_open_close_galaxy_mesh(silicon_arch_name, silicon_arch_wormhole_b0):
    if ttnn.get_num_devices() < 32:
        pytest.skip("Test is only valid on Galaxy")

    """Manually open and close multi-device"""
    mesh_shape, device_ids = ttnn.MeshShape(1, 4), ttnn.get_device_ids()
    multi_device = ttnn.open_mesh_device(mesh_shape, device_ids)
    assert multi_device.get_num_devices() == 4
    ttnn.close_mesh_device(multi_device)

    mesh_shape, device_ids = ttnn.MeshShape(8, 1), ttnn.get_device_ids()
    multi_device = ttnn.open_mesh_device(mesh_shape, device_ids)
    assert multi_device.get_num_devices() == 8
    ttnn.close_mesh_device(multi_device)

    mesh_shape, device_ids = ttnn.MeshShape(8, 4), ttnn.get_device_ids()
    multi_device = ttnn.open_mesh_device(mesh_shape, device_ids)
    assert multi_device.get_num_devices() == 32
    ttnn.close_mesh_device(multi_device)

    mesh_shape = ttnn.MeshShape(3, 2)
    multi_device = ttnn.open_mesh_device(mesh_shape, device_ids)
    assert multi_device.get_num_devices() == 6
    ttnn.close_mesh_device(multi_device)
