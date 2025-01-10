# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger
from transformers import AutoImageProcessor
import pytest
import ttnn

from models.utility_functions import (
    profiler,
)

from models.demos.wormhole.stable_diffusion_3_5.tests.sd3_6_test_infra import create_test_infra

from models.perf.perf_utils import prep_perf_report

from diffusers import StableDiffusion3Pipeline
from models.experimental.functional_stable_diffusion3_5.reference.sd3_transformer_2d_model import SD3Transformer2DModel

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


def buffer_address(tensor):
    addr = []
    for ten in ttnn.get_device_tensors(tensor):
        addr.append(ten.buffer_address())
    return addr


def dump_device_profiler(device):
    if isinstance(device, ttnn.Device):
        ttnn.DumpDeviceProfiler(device)
    else:
        for dev in device.get_device_ids():
            ttnn.DumpDeviceProfiler(device.get_device(dev))


# TODO: Create ttnn apis for this
ttnn.dump_device_profiler = dump_device_profiler

model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}

# TODO: Create ttnn apis for this
ttnn.buffer_address = buffer_address


def run_trace_model(
    device,
    torch_input_x,
    torch_input_conditioning_embedding,
    test_infra,
    num_warmup_iterations,
    num_measurement_iterations,
):
    ttnn_input_x, ttnn_input_conditioning_embedding = test_infra.setup_inputs(device)
    # Compile
    profiler.start("compile")
    test_infra.input_tensor_x = ttnn_input_x.to(device)
    test_infra.input_tensor_conditioning_embedding = ttnn_input_conditioning_embedding.to(device)
    x_shape = test_infra.input_tensor_x.shape
    x_dtype = test_infra.input_tensor_x.dtype
    conditioning_embedding_layout = test_infra.input_tensor_conditioning_embedding.layout
    conditioning_embedding_shape = test_infra.input_tensor_conditioning_embedding.shape
    conditioning_embedding_dtype = test_infra.input_tensor_conditioning_embedding.dtype
    conditioning_embedding_layout = test_infra.input_tensor_conditioning_embedding.layout

    _ = ttnn.from_device(test_infra.run(), blocking=True)
    profiler.end("compile")
    ttnn.dump_device_profiler(device)
    test_infra.output_tensor.deallocate(force=True)

    profiler.start("cache")
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    _ = ttnn.from_device(test_infra.run(), blocking=True)
    profiler.end("cache")
    ttnn.dump_device_profiler(device)

    # Capture
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    test_infra.output_tensor.deallocate(force=True)
    trace_input_addr = ttnn.buffer_address(test_infra.input_tensor)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    tt_output_res = test_infra.run()
    tt_image_res = ttnn.allocate_tensor_on_device(
        shape,
        dtype,
        layout,
        device,
        input_mem_config,
    )
    ttnn.end_trace_capture(device, tid, cq_id=0)
    assert trace_input_addr == ttnn.buffer_address(tt_image_res)
    ttnn.dump_device_profiler(device)

    for iter in range(0, num_warmup_iterations):
        ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        _ = ttnn.from_device(tt_output_res, blocking=True)
        ttnn.dump_device_profiler(device)

    ttnn.synchronize_devices(device)
    if use_signpost:
        signpost(header="start")
    outputs = []
    profiler.start(f"run")
    for iter in range(0, num_measurement_iterations):
        ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        outputs.append(ttnn.from_device(tt_output_res, blocking=False))
    ttnn.synchronize_devices(device)
    profiler.end(f"run")
    if use_signpost:
        signpost(header="stop")
    ttnn.dump_device_profiler(device)

    ttnn.release_trace(device, tid)


def run_perf_sd35(
    device_batch_size,
    expected_inference_time,
    expected_compile_time,
    device,
    mode,
):
    profiler.clear()
    # if device_batch_size <= 2:
    #     pytest.skip("Batch size 1 and 2 are not supported with sharded data")

    is_mesh_device = isinstance(device, ttnn.MeshDevice)
    num_devices = device.get_num_devices() if is_mesh_device else 1
    batch_size = device_batch_size * num_devices
    print("batch_size: ", batch_size)
    print("device_batch_size: ", device_batch_size)
    first_key = f"first_iter_batchsize{batch_size}"
    second_key = f"second_iter_batchsize{batch_size}"
    cpu_key = f"ref_key_batchsize{batch_size}"
    model_name = "stabilityai/stable-diffusion-3.5-medium"

    pipe = StableDiffusion3Pipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    config = pipe.transformer.config
    reference_model = SD3Transformer2DModel(
        sample_size=128,
        patch_size=2,
        in_channels=16,
        num_layers=24,
        attention_head_dim=64,
        num_attention_heads=24,
        joint_attention_dim=4096,
        caption_projection_dim=1536,
        pooled_projection_dim=2048,
        out_channels=16,
        pos_embed_max_size=384,
        dual_attention_layers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        qk_norm="rms_norm",
        config=config,
    )
    reference_model.load_state_dict(pipe.transformer.state_dict())
    reference_model = reference_model.norm_out.to(dtype=torch.bfloat16)
    reference_model.eval()

    torch_input_x = torch.randn([2, 1024, 1536], dtype=torch.bfloat16)
    torch_input_conditioning_embedding = torch.randn(2, 1536, dtype=torch.bfloat16)

    test_infra = create_test_infra(
        device,
        device_batch_size,
        model_config["ACTIVATIONS_DTYPE"],
        model_config["WEIGHTS_DTYPE"],
        model_config["MATH_FIDELITY"],
        config,
        dealloc_input=True,
        final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn.synchronize_devices(device)

    num_warmup_iterations = 5
    num_measurement_iterations = 15

    with torch.no_grad():
        profiler.start(cpu_key)
        logits = reference_model(torch_input_x, torch_input_conditioning_embedding)
        profiler.end(cpu_key)

        run_trace_model(
            device,
            torch_input_x,
            torch_input_conditioning_embedding,
            test_infra,
            num_warmup_iterations,
            num_measurement_iterations,
        )

    first_iter_time = profiler.get(f"compile") + profiler.get(f"cache")

    # ensuring inference time fluctuations is not noise
    inference_time_avg = profiler.get("run") / num_measurement_iterations

    cpu_time = profiler.get(cpu_key)
    compile_time = first_iter_time - 2 * inference_time_avg
    prep_perf_report(
        model_name=f"ttnn_{model_version}_batch_size{batch_size}",
        batch_size=batch_size,
        inference_and_compile_time=first_iter_time,
        inference_time=inference_time_avg,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comments,
        inference_time_cpu=cpu_time,
    )

    logger.info(
        f"{model_name} {comments} inference time (avg): {inference_time_avg}, FPS: {batch_size/inference_time_avg}"
    )
    logger.info(f"{model_name} compile time: {compile_time}")
