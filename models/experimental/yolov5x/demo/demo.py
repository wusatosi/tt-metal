# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import ttnn
import torch
import pytest
from loguru import logger
from ultralytics import YOLO
from models.utility_functions import run_for_wormhole_b0
from models.experimental.yolov5x.tt.ttnn_yolov5x import Yolov5
from models.experimental.yolov5x.reference.yolov5x import YOLOv5
from models.utility_functions import disable_persistent_kernel_cache
from models.experimental.yolov5x.tt.model_preprocessing import (
    create_yolov5x_input_tensors,
    create_yolov5x_model_parameters,
)
from models.experimental.yolov5x.demo.demo_utils import load_coco_class_names
from models.experimental.yolo_evaluation.yolo_common_evaluation import save_yolo_predictions_by_model
from models.experimental.yolo_evaluation.yolo_evaluation_utils import LoadImages, preprocess, postprocess


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "source",
    [
        # "models/sample_data/huggingface_cat_image.jpg",
        "models/demos/yolov4/resources/giraffe_320.jpg",  # Uncomment to run the demo with another image.
        # "models/experimental/yolov5x/demo/cycle_girl.jpg",
        # "models/experimental/yolov5x/demo/dog.jpg",
        # "models/experimental/yolov5x/demo/bus.jpg"
    ],
)
@pytest.mark.parametrize(
    "model_type",
    [
        "tt_model",
        # "torch_model", # Uncomment to run the demo with torch model.
    ],
)
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [
        # "False", # Uncomment to run the demo with random weights.
        "True",
    ],
)
def test_demo(device, source, model_type, use_weights_from_ultralytics, use_program_cache, reset_seeds):
    disable_persistent_kernel_cache()

    if model_type == "torch_model":
        state_dict = None
        if use_weights_from_ultralytics:
            torch_model = YOLO("yolov5xu.pt")
            state_dict = torch_model.state_dict()

        model = YOLOv5()
        state_dict = model.state_dict() if state_dict is None else state_dict

        ds_state_dict = {k: v for k, v in state_dict.items()}
        new_state_dict = {}
        for (name1, parameter1), (name2, parameter2) in zip(state_dict.items(), ds_state_dict.items()):
            if isinstance(parameter2, torch.FloatTensor):
                new_state_dict[name1] = parameter2
        model.load_state_dict(new_state_dict)
        model.eval()
        logger.info("Inferencing [Torch] Model")
    else:
        torch_input, ttnn_input = create_yolov5x_input_tensors(device)
        state_dict = None
        if use_weights_from_ultralytics:
            torch_model = YOLO("yolov5xu.pt").model.eval()
            state_dict = torch_model.state_dict()

        torch_model = YOLOv5().model
        state_dict = torch_model.state_dict() if state_dict is None else state_dict
        ds_state_dict = {k: v for k, v in state_dict.items()}
        new_state_dict = {}
        for (name1, parameter1), (name2, parameter2) in zip(state_dict.items(), ds_state_dict.items()):
            if isinstance(parameter2, torch.FloatTensor):
                new_state_dict[name1] = parameter2

        torch_model.load_state_dict(new_state_dict)
        torch_model.eval()
        parameters = create_yolov5x_model_parameters(torch_model, torch_input, device=device)
        model = Yolov5(device, parameters, parameters)
        logger.info("Inferencing [TTNN] Model")

    save_dir = "models/experimental/yolov5x/demo/runs"
    dataset = LoadImages(path=source)
    model_save_dir = os.path.join(save_dir, model_type)
    os.makedirs(model_save_dir, exist_ok=True)
    names = load_coco_class_names()

    for batch in dataset:
        paths, im0s, s = batch
        im = preprocess(im0s, res=640)
        if model_type == "torch_model":
            preds = model(im)
        else:
            img = torch.permute(im, (0, 2, 3, 1))
            img = img.reshape(
                1,
                1,
                img.shape[0] * img.shape[1] * img.shape[2],
                img.shape[3],
            )
            ttnn_im = ttnn.from_torch(
                img,
                dtype=ttnn.bfloat8_b,
                device=device,
                layout=ttnn.TILE_LAYOUT,
            )
            preds = model(ttnn_im)
            preds = ttnn.to_torch(preds, dtype=torch.float32)

        results = postprocess(preds, im, im0s, batch, names)[0]
        save_yolo_predictions_by_model(results, save_dir, source, model_type)

    logger.info("Inference done")
