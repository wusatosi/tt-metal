# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from models.utility_functions import skip_for_grayskull
from models.demos.yolov4_updated.reference.downsample1 import DownSample1
from models.demos.yolov4_updated.ttnn.downsample1 import Down1
from models.demos.yolov4_updated.ttnn.model_preprocessing import create_ds1_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc
import pytest
import time
from loguru import logger
import os


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "resolution",
    [
        (640, 640),
    ],
)
def test_down1(device, reset_seeds, model_location_generator, resolution):
    torch.manual_seed(0)

    weights_pth = "tests/ttnn/integration_tests/yolov4_updated/yolov4.pth"
    if not os.path.exists(weights_pth):
        os.system("tests/ttnn/integration_tests/yolov4_updated/yolov4_weights_download.sh")

    torch_input = torch.randn((1, 3, resolution[0], resolution[1]), dtype=torch.bfloat16)
    torch_input = torch_input.float()
    torch_model = DownSample1()

    torch_dict = torch.load(weights_pth)
    ds_state_dict = {k: v for k, v in torch_dict.items() if (k.startswith("down1."))}
    new_state_dict = dict(zip(torch_model.state_dict().keys(), ds_state_dict.values()))
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()
    ref = torch_model(torch_input)

    parameters = create_ds1_model_parameters(torch_model, torch_input, resolution, device)

    ttnn_model = Down1(device, parameters, parameters.conv_args)

    torch_input = torch_input.permute(0, 2, 3, 1)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16)

    result_ttnn = ttnn_model(ttnn_input)
    if resolution[0] == 320:
        start_time = time.time()
        for x in range(100):
            result_ttnn = ttnn_model(ttnn_input)
        logger.info(f"Time taken: {time.time() - start_time}")

    result = ttnn.to_torch(result_ttnn)
    result = result.permute(0, 3, 1, 2)
    result = result.reshape(ref.shape)
    assert_with_pcc(result, ref, 0.99)
