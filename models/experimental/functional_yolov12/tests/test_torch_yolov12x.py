# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys
import ttnn
import torch
import pytest
import torch.nn as nn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_yolov12.reference import yolov12
from models.experimental.functional_yolov12.demo.demo_utils import attempt_load

try:
    sys.modules["ultralytics"] = yolov12
    sys.modules["ultralytics.nn.tasks"] = yolov12
    sys.modules["ultralytics.nn.modules.conv"] = yolov12
    sys.modules["ultralytics.nn.modules.block"] = yolov12
    sys.modules["ultralytics.nn.modules.head"] = yolov12

except KeyError:
    print("models.experimental.functional_yolov12.reference.yolov12 not found.")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolov12x(device, use_program_cache, reset_seeds):
    batch_size = 1
    input_channels = 3
    input_height, input_width = 640, 640
    # torch_input = torch.randn(batch_size, input_channels, input_height, input_width)
    torch_input = torch.load(f"/home/ubuntu/keerthana/tt-metal/models/experimental/functional_yolov12/dumps/input.pt")
    state_dict = None
    use_pretrained_weight = True
    if use_pretrained_weight:
        torch_model = attempt_load("yolo12x.pt", map_location="cpu")
        state_dict = torch_model.state_dict()

    torch_model = yolov12.YoloV12x()
    state_dict = torch_model.state_dict() if state_dict is None else state_dict

    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2

    torch_model.load_state_dict(new_state_dict, strict=False)
    torch_model.eval()
    torch_output = torch_model(torch_input)
    # torch.save(torch_output, f"/home/ubuntu/keerthana/tt-metal/models/experimental/functional_yolov12/dumps/torch_op.pt")
    ttnn_output = torch.load("/home/ubuntu/keerthana/tt-metal/models/experimental/functional_yolov12/dumps/output.pt")
    assert_with_pcc(torch_output, ttnn_output[0], 0.99999999999999)
