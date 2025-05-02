# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from models.experimental.centernet.reference.ct_resnet_neck import CTResNetNeck
from tests.ttnn.utils_for_testing import assert_with_pcc


def test_torch_neck():
    # state_dict = torch.load("models/experimental/centernet/centernet_resnet18_140e_coco_20210705_093630-bb5b3bf7.pth")[
    #     "state_dict"
    # ]
    state_dict = torch.load(
        "models/experimental/centernet/centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth"
    )["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if "neck" in k:
            new_state_dict[k] = v
    for k, v in new_state_dict.items():
        print(k)
    with torch.no_grad():
        reference = CTResNetNeck(parameters=new_state_dict)
        reference.eval()
    # reference.load_state_dict(new_state_dict)
    print(reference)

    # input = torch.rand(1, 3, 32, 32)
    input = (
        torch.load("ct_input_0.pt"),
        torch.load("ct_input_1.pt"),
        torch.load("ct_input_2.pt"),
        torch.load("ct_input_3.pt"),
    )
    for k in input:
        print(k.shape)

    # torch.Size([1, 64, 112, 168])
    # torch.Size([1, 128, 56, 84])
    # torch.Size([1, 256, 28, 42])
    # torch.Size([1, 512, 14, 21])

    output = reference.forward(input)
    assert_with_pcc(output[0], torch.load("ct_output.pt"), 0.99)
