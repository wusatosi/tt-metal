# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
import numpy as np

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("w", [1, 4, 8, 32])
def test_plus_one_int32(device, w):
    torch_input_tensor = torch.randint(32000, (w,))
    torch_output_tensor = torch_input_tensor + 1

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.int32, device=device)
    ttnn.plus_one(input_tensor)
    output_tensor = ttnn.to_torch(input_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.parametrize("w", [1, 4, 8, 32])
def test_plus_one_uint32(device, w):
    torch_input_tensor = torch.randint(0, 2**32 - 1, (w,), dtype=torch.int64).numpy().astype(np.uint32)
    torch_output_tensor = torch_input_tensor + 1

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.uint32, device=device)
    ttnn.plus_one(input_tensor)
    output_tensor = ttnn.to_torch(input_tensor).numpy().astype(np.uint32)
    assert np.array_equal(output_tensor, torch_output_tensor)


@pytest.mark.parametrize("shape", [(1, 32), (4, 5)])
def test_plus_one_2D(device, shape):
    torch_input_tensor = torch.randint(0, 2**32 - 1, shape, dtype=torch.int64).numpy().astype(np.uint32)
    torch_output_tensor = torch_input_tensor + 1

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.uint32, device=device)
    ttnn.plus_one(input_tensor)
    output_tensor = ttnn.to_torch(input_tensor).numpy().astype(np.uint32)
    assert np.array_equal(output_tensor, torch_output_tensor)
