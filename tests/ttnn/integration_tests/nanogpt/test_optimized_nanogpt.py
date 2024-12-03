# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from models.demos.nanogpt.tt import ttnn_optimized_nanogpt
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import is_wormhole_b0, skip_for_grayskull


@pytest.mark.parametrize("model_name", ["gpt-2"])
@pytest.mark.parametrize("batch_size", [8])
def test_nanogpt_model(device, model_name, batch_size, reset_seeds):
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    config = model.config
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    prompt = batch_size * ["Hello, my dog is a little"]
    inputs = tokenizer(prompt, return_tensors="pt", padding=False)

    torch_model = model

    input_shape = inputs.input_ids.size()
    input_ids = inputs.input_ids.view(-1, input_shape[-1])
    batch_size = inputs.input_ids.shape[0]
    position_ids = torch.arange(0, input_shape[-1] + 0, dtype=torch.long).unsqueeze(0)
    torch_output = torch_model(inputs.input_ids)

    parameters = preprocess_model_parameters(
        model_name=model,
        initialize_model=lambda: model,
        device=device,
    )
    block_size = 1024
    T = 7
    bias = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
    ones_t = torch.zeros(batch_size, 12, 7, 7)
    ones_t = ones_t.masked_fill(bias[:, :, :T, :T] == 0, -10000)
    bias = ttnn.from_torch(ones_t, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_inputs = ttnn.from_torch(inputs.input_ids, dtype=ttnn.uint32, device=device)
    tt_position_ids = ttnn.from_torch(position_ids, dtype=ttnn.uint32, device=device)

    tt_output = ttnn_optimized_nanogpt.nanogpt_model(
        model.config,
        input_idx=tt_inputs,
        position_ids=tt_position_ids,
        att_bias=bias,
        parameters=parameters,
        device=device,
    )
    tt_output_tensor = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output[0], tt_output_tensor, 0.99)
