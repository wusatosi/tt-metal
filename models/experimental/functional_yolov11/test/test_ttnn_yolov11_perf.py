# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch
from tests.ttnn.utils_for_testing import assert_with_pcc
import sys

from models.experimental.functional_yolov11.reference import yolov11

from models.experimental.functional_yolov11.tt.model_preprocessing import (
    create_yolov11_input_tensors,
    create_yolov11_model_parameters,
)
from models.experimental.functional_yolov11.tt import ttnn_yolov11
import torch.nn as nn

try:
    sys.modules["ultralytics"] = yolov11
    sys.modules["ultralytics.nn.tasks"] = yolov11
    sys.modules["ultralytics.nn.modules.conv"] = yolov11
    sys.modules["ultralytics.nn.modules.block"] = yolov11
    sys.modules["ultralytics.nn.modules.head"] = yolov11

except KeyError:
    print("models.experimental.functional_yolov11.reference.yolov11 not found.")


class Ensemble(nn.ModuleList):
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        y = torch.cat(y, 1)
        return y, None


def attempt_load(weights, map_location=None):
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        w = "models/experimental/functional_yolov11/reference/yolo11n.pt"
        ckpt = torch.load(w, map_location=map_location)
        model.append(ckpt["ema" if ckpt.get("ema") else "model"].float().eval())
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None

    if len(model) == 1:
        return model[-1]
    else:
        for k in ["names", "stride"]:
            setattr(model, k, getattr(model[-1], k))
        return model


def get_expected_times(name):
    base = {"yolov11": ()}
    return base[name]


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("input_tensor", [torch.rand((1, 3, 224, 224))], ids=["input_tensor"])
def test_yolov11(device, input_tensor):
    disable_persistent_kernel_cache()
    torch_model = attempt_load("yolov11.pt", map_location="cpu")
    state_dict = torch_model.state_dict()
