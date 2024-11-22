# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import json
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
from models.demos.yolov4.tests.yolov4_perfomant_webdemo import Yolov4Trace2CQ
import ttnn

import cv2
import numpy as np
import torch
import time
import os

app = FastAPI(
    title="YOLOv4 object detection",
    description="Inference engine to detect objects in image.",
    version="0.0",
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


def get_dispatch_core_type():
    # TODO: 11059 move dispatch_core_type to device_params when all tests are updated to not use WH_ARCH_YAML env flag
    dispatch_core_type = ttnn.device.DispatchCoreType.WORKER
    # if ("WH_ARCH_YAML" in os.environ) and os.environ["WH_ARCH_YAML"] == "wormhole_b0_80_arch_eth_dispatch.yaml":
    if os.environ["WH_ARCH_YAML"] == "wormhole_b0_80_arch_eth_dispatch.yaml":
        dispatch_core_type = ttnn.device.DispatchCoreType.ETH
    return dispatch_core_type


@app.on_event("startup")
async def startup():
    # Set the environment variable
    # os.environ['WH_ARCH_YAML'] = 'wormhole_b0_80_arch_eth_dispatch.yaml'
    # Verify that it's set
    print("WH_ARCH_YAML:", os.environ.get("WH_ARCH_YAML"))

    device_id = 0
    print()
    print("\n\n\n\ndevice_id is set to: ", device_id)
    # device = ttnn.open_device(device_id=device_id)
    # ttnn.enable_program_cache(device)
    # device = ttnn.CreateDevice(device_id, dispatch_core_type=get_dispatch_core_type(), l1_small_size=24576, trace_region_size=1617920, num_command_queues=2)
    device = ttnn.CreateDevice(
        device_id,
        dispatch_core_type=get_dispatch_core_type(),
        l1_small_size=24576,
        trace_region_size=1622016,
        num_command_queues=2,
    )
    ttnn.enable_program_cache(device)
    print()
    print("\n\n\nDevice is created!")
    # ttnn.enable_program_cache(device)
    print("\n\n\n\nwe do reach the point after enabling program cache!\n\n\n\n")
    global model
    model = Yolov4Trace2CQ()
    model.initialize_yolov4_trace_2cqs_inference(device)


@app.on_event("shutdown")
async def shutdown():
    model.release_yolov4_trace_2cqs_inference()


def process_request(output):
    # Convert all tensors to lists for JSON serialization
    output_serializable = {"output": [tensor.tolist() for tensor in output]}
    return output_serializable


@app.post("/objdetection_v2")
async def objdetection_v2(file: UploadFile = File(...)):
    contents = await file.read()

    # Load and convert the image to RGB
    image = Image.open(BytesIO(contents)).convert("RGB")
    image = np.array(image)
    if type(image) == np.ndarray and len(image.shape) == 3:  # cv2 image
        image = torch.from_numpy(image).float().div(255.0).unsqueeze(0)
    elif type(image) == np.ndarray and len(image.shape) == 4:
        image = torch.from_numpy(image).float().div(255.0)
    else:
        print("unknow image type")
        exit(-1)
    t1 = time.time()
    response = model.run_traced_inference(image)
    t2 = time.time()
    print("the inference on the sever side took: ", t2 - t1)

    # Convert response tensors to JSON-serializable format
    output = process_request(response)
    return output
