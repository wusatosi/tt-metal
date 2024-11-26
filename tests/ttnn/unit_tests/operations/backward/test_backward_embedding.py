# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests import (
    comparison_funcs,
)
from models.utility_functions import skip_for_grayskull
from loguru import logger

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RESET = "\033[0m"  # Reset to default color


@skip_for_grayskull()
@pytest.mark.parametrize(
    "batch_size, seq_len, embedding_dim, num_embeddings",
    [
        (1, 32, 32, 32),
        # (3, 32, 384, 320),
        # (2, 1024, 4096, 3200),
    ],
)
@pytest.mark.parametrize(
    "output_dtype",
    [
        ttnn.bfloat16,
        # ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        # ttnn.uint32,
    ],
)
def test_embedding_bw(input_dtype, output_dtype, batch_size, seq_len, embedding_dim, num_embeddings, device):
    torch.manual_seed(1234)

    if input_dtype == ttnn.bfloat16 and num_embeddings > 256:
        pytest.skip("Skipping tests with large vocab sizes for bfloat16 indices!")

    input_shape = (batch_size, seq_len)  # (1,32)
    input_index = torch.randint(0, num_embeddings, input_shape)
    # input_index = torch.arange(0, 32).reshape(input_shape)
    input_tensor = ttnn.from_torch(input_index, dtype=input_dtype, device=device)

    weights_shape = (num_embeddings, embedding_dim)  # (32,32)
    # weights = torch.randn(weights_shape, requires_grad=True)
    weights = torch.arange(1, num_embeddings * embedding_dim + 1, dtype=torch.bfloat16).reshape(weights_shape)
    weights.requires_grad = True
    weights_ttnn = ttnn.from_torch(weights, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    grad_shape = (1, 1, batch_size * seq_len, embedding_dim)  # (1,1,32,32)
    # grad_data = torch.randn(grad_shape, requires_grad=True)
    grad_data = torch.arange(0, 1024, dtype=torch.bfloat16).reshape(grad_shape)
    grad_data.requires_grad = True
    grad_tensor = ttnn.from_torch(grad_data, dtype=output_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    tt_output_tensor_on_device = ttnn.embedding_bw(input_tensor, weights_ttnn, grad_tensor, dtype=output_dtype)
    tt_output_tensor = ttnn.to_torch(tt_output_tensor_on_device)

    # PyTorch reference
    weights.retain_grad()
    pyt_y = torch.nn.functional.embedding(input_index, weights).reshape(grad_shape)
    pyt_y.backward(gradient=grad_data)
    golden_output_tensor = weights.grad

    torch.set_printoptions(profile="full")

    # print("$"*50)
    # print(weights[input_shape[0]][:].size()) # 160

    # print("*"*50)
    # print("INPUT")
    # print(ttnn.to_torch(input_tensor))
    # print("*"*50)
    # print("WEIGHTS")
    # print(ttnn.to_torch(weights_ttnn))
    # print("*"*50)
    # print("GRAD")
    # print(ttnn.to_torch(grad_tensor))

    tt_output_tensor_flat = torch.flatten(tt_output_tensor)
    golden_output_tensor_flat = torch.flatten(golden_output_tensor)

    # print("*" * 50)
    # print("TT_OUTPUT")
    # print(tt_output_tensor_flat)
    # print("*" * 50)
    # print("GOLDEN")
    # print(golden_output_tensor_flat)
    # print("*" * 50)
    # print(tt_output_tensor_flat.size())
    # print(golden_output_tensor_flat.size())

    for i in range(0, 1024, 16):
        # tweeking some values for better undestanding of results
        if i < 512:
            if (i / 16) % 2 == 1:
                tt_output_tensor_flat[i : i + 16] /= 2

        print(f"{i} : {RED}{tt_output_tensor_flat[i:i+16].tolist()}{RESET}")
        print(f"{i} : {GREEN}{golden_output_tensor_flat[i:i+16].tolist()}{RESET}")
        truth = []

        for j in range(i, i + 16):
            truth.append(1 if tt_output_tensor_flat[j] == golden_output_tensor_flat[j] else 0)
        print(f"{YELLOW}{truth}{RESET}")
        print(f"{CYAN}")
        print("*" * 80)

        if i % 256 == 0 and i != 0:
            print(f"{MAGENTA}")
            print("-" * 80)

    comp_pass, comp_out = comparison_funcs.comp_pcc(golden_output_tensor, tt_output_tensor)

    logger.debug("DEBUG")
    logger.debug(comp_out)
    assert comp_pass
