# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from models.experimental.centernet.tt.tt_centernet_neck import TtCTResNetNeck
from models.experimental.centernet.tt.model_preprocessing import custom_preprocessor
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_centernet_neck(device, reset_seeds):
    state_dict = torch.load(
        "models/experimental/centernet/centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth"
    )["state_dict"]

    parameters = custom_preprocessor(device, state_dict)

    Neck = TtCTResNetNeck(parameters=parameters, device=device)

    input = (
        ttnn.from_torch(torch.load("ct_input_0.pt"), dtype=ttnn.bfloat16, device=device),
        ttnn.from_torch(torch.load("ct_input_1.pt"), dtype=ttnn.bfloat16, device=device),
        ttnn.from_torch(torch.load("ct_input_2.pt"), dtype=ttnn.bfloat16, device=device),
        ttnn.from_torch(torch.load("ct_input_3.pt"), dtype=ttnn.bfloat16, device=device),
    )

    output = Neck.forward(input)

    assert_with_pcc(ttnn.to_torch(output[0]), torch.load("ct_output.pt"), 0.99)
