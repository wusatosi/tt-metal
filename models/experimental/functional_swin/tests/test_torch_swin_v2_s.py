# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torchvision import models
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_grayskull
from models.experimental.functional_swin.reference.swin_transformer import SwinTransformer


def test_torch_swin_s_transformer(device, reset_seeds):
    model = models.swin_v2_s(weights="IMAGENET1K_V1").eval()
    state_dict = model.state_dict()

    torch_model = SwinTransformer(
        patch_size=[4, 4], embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24], window_size=[8, 8]
    )

    torch_model.load_state_dict(state_dict)
    torch_model.eval()
    print(torch_model)
    torch_input_tensor = torch.randn(1, 3, 512, 512)  # Sample input tensor
    torch_output_tensor = torch_model(torch_input_tensor)
    pretrained_output = model(torch_input_tensor)
    assert_with_pcc(torch_output_tensor, pretrained_output)
