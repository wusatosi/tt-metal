# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from models.experimental.centernet.tt.tt_centernet_head import TtCTResNetHead
from models.experimental.centernet.tt.model_preprocessing import custom_preprocessor
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_centernet_head(device, reset_seeds):
    state_dict = torch.load(
        "models/experimental/centernet/centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth"
    )["state_dict"]

    parameters = custom_preprocessor(device, state_dict)
    box_head = TtCTResNetHead(parameters=parameters, device=device)

    input = torch.load("centernethead.pt")
    tt_input = ttnn.from_torch(input, dtype=ttnn.bfloat16, device=device)

    output = box_head.forward(tt_input)

    torch_output_1 = torch.load("center_heatmap_pred.pt")
    torch_output_2 = torch.load("wh_pred.pt")
    torch_output_3 = torch.load("offset_pred.pt")
    assert_with_pcc(ttnn.to_torch(output[0]), torch_output_1, 0.99)
    assert_with_pcc(ttnn.to_torch(output[1]), torch_output_2, 0.99)
    assert_with_pcc(ttnn.to_torch(output[2]), torch_output_3, 0.99)
