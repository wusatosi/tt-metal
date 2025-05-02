# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from models.experimental.centernet.reference.centernet_head import CTResNetHead
from tests.ttnn.utils_for_testing import assert_with_pcc


def test_torch_head():
    # state_dict = torch.load("models/experimental/centernet/centernet_resnet18_140e_coco_20210705_093630-bb5b3bf7.pth")[
    #     "state_dict"
    # ]
    state_dict = torch.load(
        "models/experimental/centernet/centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth"
    )["state_dict"]
    new_state_dict = {}

    for k, v in state_dict.items():
        if "bbox_head" in k:
            new_state_dict[k] = v
    for k, v in new_state_dict.items():
        print(k)
    with torch.no_grad():
        reference = CTResNetHead(parameters=new_state_dict)
        reference.eval()

    input = torch.load("centernethead.pt")
    output = reference.forward(input)
    torch_output_1 = torch.load("center_heatmap_pred.pt")
    torch_output_2 = torch.load("wh_pred.pt")
    torch_output_3 = torch.load("offset_pred.pt")
    assert_with_pcc(output[0], torch_output_1, 0.999999)
    assert_with_pcc(output[1], torch_output_2, 0.999999)
    assert_with_pcc(output[2], torch_output_3, 0.999999)
