# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def test_reshape_mesh_device_2D(reset_seeds, mesh_device):
    input_tensor = torch.rand((128, 64, 6, 6), dtype=torch.bfloat16)  # NCHW
    x = torch.reshape(input_tensor, (input_tensor.shape[0], -1))

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    input_tensor = ttnn.from_torch(
        input_tensor,
        mesh_mapper=inputs_mesh_mapper,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    y = ttnn.reshape(input_tensor, (input_tensor.shape[0], -1))

    y = ttnn.to_torch(y, mesh_composer=output_mesh_composer)

    assert_with_pcc(x, y, 1)
