# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from models.common.lightweightmodule import LightweightModule
from models.demos.llama3.tt.llama_common import precompute_freqs, get_rot_transformation_mat, gather_cos_sin
from models.utility_functions import nearest_32
from loguru import logger


def moe_gate(
    mesh_device,
    num_users,
    num_experts,
    gate="uniform_random",
):
    """'
    Assumptions:
    - uniformly distribute users across experts
    """

    assert num_users % num_experts == 0, f"num_users: {num_users} must be divisible by num_experts: {num_experts}"
    num_users_per_expert = num_users // num_experts

    user_mapping = torch.zeros(num_users, dtype=torch.int)
    user_count_per_expert = [0] * num_experts

    if gate == "uniform_random":
        for i in range(num_users):
            expert = torch.randint(0, num_experts, (1,), dtype=torch.int).item()
            while user_count_per_expert[expert] >= num_users_per_expert:
                expert = torch.randint(0, num_experts, (1,), dtype=torch.int).item()
            user_mapping[i] = expert
            user_count_per_expert[expert] += 1

    else:
        raise ValueError(f"Invalid gate type: {gate}")

    # Convert to ttnn tensor
    tt_user_mapping = ttnn.from_torch(
        user_mapping,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return tt_user_mapping, user_mapping


def dispatch_golden(
    input_tensor,
    user_mapping,
):
    _, num_experts, num_users_per_expert, dim = input_tensor.shape
    num_users = num_users_per_expert * num_experts

    out = torch.zeros(1, num_experts, num_users, dim)
    dispatch_table = torch.zeros(num_experts, num_users)

    for u in range(num_users):
        expert = user_mapping[u].item()

        # Dispatch (CCL) part
        out[0, expert, u] = input_tensor[0, expert, u % num_users_per_expert]
        dispatch_table[expert, u] = 1

    return out, dispatch_table


@pytest.mark.parametrize(
    "num_users, num_experts",
    [
        (
            64,
            8,
        ),
    ],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 8), id="t3k")], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 23887872}],
    indirect=True,
)
def test_run_moe(
    mesh_device,
    num_users,
    num_experts,
):
    logger.info(f"Mesh device shape: {mesh_device.shape}")
    dim = 128
    num_users_per_expert = num_users // num_experts

    # Input tensor
    input_tensor = torch.randn(1, num_experts, num_users_per_expert, dim).float()

    # MOE gate
    tt_user_mapping, user_mapping = moe_gate(mesh_device, num_users, num_experts, gate="uniform_random")

    # Dispatch golden
    dispatch_out, dispatch_table = dispatch_golden(input_tensor, user_mapping)

    assert True
