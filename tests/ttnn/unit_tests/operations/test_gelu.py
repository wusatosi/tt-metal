# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
import torch.nn.functional as F


def test_gelu(device):
    torch.manual_seed(0)
    torch_input_tensor = torch.rand((2, 4096, 6144), dtype=torch.bfloat16)
    torch_output_tensor = F.gelu(torch_input_tensor)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    output_tensor = ttnn.gelu(input_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.99)
