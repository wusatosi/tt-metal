# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
import pytest
from models.utility_functions import comp_allclose_and_pcc
from loguru import logger

from tests.ttnn.unit_tests.operations.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
    to_torch,
    to_ttnn,
)


def test_moreh_bug_report(device):
    input_torch = torch.rand(32, 32).to(torch.float32)
    input_npu = ttnn.from_torch(input_torch, dtype=ttnn.bfloat16, device=device)
    ttnn.operations.moreh.bug_report(input_npu)
