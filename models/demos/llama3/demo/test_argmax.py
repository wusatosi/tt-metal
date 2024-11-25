# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import ttnn
from models.utility_functions import is_wormhole_b0, is_grayskull, skip_for_wormhole_b0
import torch
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_equal,
    comp_pcc,
)


@pytest.mark.skipif(is_grayskull(), reason="GS does not support fp32")
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 128 * 64),
        (1, 1, 16, 128 * 64),
        (1, 1, 8, 128 * 64),
        (1, 1, 4, 128 * 64),
        (1, 1, 2, 128 * 64),
        (1, 1, 1, 128 * 64),
    ],
)
def test_argmax(
    device,
    shape,
    function_level_defaults,
):
    pt_input = torch.randn(shape)

    tt_input = ttnn.as_tensor(
        pt_input,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    logger.info(f"pt_input shape: {pt_input.shape}")
    logger.info(f"tt_input shape: {tt_input.shape}")

    pt_out = torch.argmax(pt_input, dim=3, keepdim=True).transpose(-1, -2)

    tt_out = ttnn.argmax(
        tt_input,
        dim=3,
        use_multicore=False,
    )
    tt_out = ttnn.to_torch(tt_out)

    logger.info(f"pt_out shape: {pt_out.shape}")
    logger.info(f"tt_out shape: {tt_out.shape}")

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@pytest.mark.skipif(is_grayskull(), reason="GS does not support fp32")
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 128 * 64),
        (1, 1, 16, 128 * 64),
        (1, 1, 8, 128 * 64),
        (1, 1, 4, 128 * 64),
        (1, 1, 2, 128 * 64),
        (1, 1, 1, 128 * 64),
    ],
)
def test_argmax_t3k(
    t3k_mesh_device,
    shape,
    function_level_defaults,
):
    pt_input = torch.randn(shape)

    tt_input = ttnn.as_tensor(
        pt_input,
        device=t3k_mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(t3k_mesh_device),
    )
    logger.info(f"pt_input shape: {pt_input.shape}")
    logger.info(f"tt_input shape: {tt_input.shape}")

    pt_out = torch.argmax(pt_input, dim=3, keepdim=True).transpose(-1, -2)

    tt_out = ttnn.argmax(
        tt_input,
        dim=3,
        use_multicore=False,
    )
    tt_out = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(t3k_mesh_device, dim=-1))[..., : shape[-2]]

    logger.info(f"pt_out shape: {pt_out.shape}")
    logger.info(f"tt_out shape: {tt_out.shape}")

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing
