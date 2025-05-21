# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from models.experimental.centernet.tt.tt_resnet import TtBasicBlock, TtResNet
from models.experimental.centernet.tt.model_preprocessing import custom_preprocessor
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_centernet_backbone(device, reset_seeds):
    state_dict = torch.load(
        "models/experimental/centernet/centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth"
    )["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if "backbone" in k:
            new_state_dict[k] = v

    parameters = custom_preprocessor(device, state_dict)
    backbone = TtResNet(TtBasicBlock, [2, 2, 2, 2], parameters=parameters, base_address="backbone", device=device)
    # backbone.eval()

    input = torch.load("backbone_centernet_input_model.pt")
    tt_input = ttnn.from_torch(input, dtype=ttnn.bfloat16, device=device)
    # assert_with_pcc(torch.load("torch_conv_one.pt"), torch.load("conv_two.pt"), 0.999999)
    # assert_with_pcc(torch.load("out1.pt"), torch.load("backbone_out_0.pt"), 0.999999)
    # assert_with_pcc(torch.load("conv1.pt"), torch.load("tt_conv1.pt"), 0.999)
    # assert_with_pcc(torch.load("bn1.pt"), torch.load("tt_bn1.pt"), 0.999)
    # assert_with_pcc(torch.load("relu1.pt"), torch.load("tt_relu1.pt"), 0.999)
    # assert_with_pcc(torch.load("conv2.pt"), torch.load("tt_conv2.pt"), 0.999)
    # assert_with_pcc(torch.load("bn2.pt"), torch.load("tt_bn2.pt"), 0.999)
    ouput = backbone.forward(tt_input)
    # assert_with_pcc(torch.load("torch_three.pt"), torch.load("tt_torch_enter.pt"), 0.999)
    # assert_with_pcc(torch.load("Enter.pt"), torch.load("tt_enter.pt"), 0.999999)
    # assert_with_pcc(torch.load("out1.pt"), torch.load("backbone_out_0.pt"), 0.999999)

    assert_with_pcc(ttnn.to_torch(ouput[0]), torch.load("backbone_out_0.pt"), 0.99)
    assert_with_pcc(ttnn.to_torch(ouput[1]), torch.load("backbone_out_1.pt"), 0.99)
    assert_with_pcc(ttnn.to_torch(ouput[2]), torch.load("backbone_out_2.pt"), 0.99)
    assert_with_pcc(ttnn.to_torch(ouput[3]), torch.load("backbone_out_3.pt"), 0.99)
