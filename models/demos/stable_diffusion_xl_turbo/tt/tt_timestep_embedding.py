# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def timestep_embedding(x, parameters, device):
    hifi2_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
    )

    mm_a_y = 8
    mm_a_x = 6
    mm_a_x_strategy = ttnn.ShardStrategy.WIDTH
    mm_a_x_memory_config = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG

    x = ttnn.linear(
        x,
        parameters.linear_1.weight,
        bias=parameters.linear_1.bias,
        activation="silu",
        memory_config=mm_a_x_memory_config,
        core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_a_x),
        compute_kernel_config=hifi2_kernel_config,
    )

    x = ttnn.linear(
        x,
        parameters.linear_2.weight,
        bias=parameters.linear_2.bias,
        memory_config=mm_a_x_memory_config,
        core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_a_x),
        compute_kernel_config=hifi2_kernel_config,
    )

    return x
