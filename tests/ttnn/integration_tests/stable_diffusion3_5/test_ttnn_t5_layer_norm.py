# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
)
from models.utility_functions import skip_for_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_stable_diffusion3_5.reference.t5_layer_norm import T5LayerNorm
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_t5_layer_norm import ttnn_T5LayerNorm


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@skip_for_grayskull()
def test_t5_layer_norm(device, reset_seeds):
    reference_model = T5LayerNorm(hidden_size=4096).to(dtype=torch.bfloat16)

    reference_model.eval()

    hidden_states = torch.randn(1, 256, 4096, dtype=torch.bfloat16)

    torch_output = reference_model(hidden_states)

    parameters = preprocess_model_parameters(initialize_model=lambda: reference_model, device=device)

    ttnn_model = ttnn_T5LayerNorm(parameters)

    ttnn_hidden_states = ttnn.from_torch(
        hidden_states,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    ttnn_output = ttnn_model(ttnn_hidden_states)

    assert_with_pcc(torch_output, ttnn.to_torch(ttnn_output), pcc=0.99)
