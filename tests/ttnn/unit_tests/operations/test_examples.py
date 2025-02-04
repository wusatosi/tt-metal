# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_equal


@pytest.mark.parametrize("height", [32 * 56])
@pytest.mark.parametrize("width", [32])
def test_example(device, height, width, use_program_cache):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((height, width), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    for x in range(800):
        output_tensor = ttnn.prim.example(input_tensor)
    ttnn.DumpDeviceProfiler(device)
