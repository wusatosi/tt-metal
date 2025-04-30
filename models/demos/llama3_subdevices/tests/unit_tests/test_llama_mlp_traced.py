# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import os
import ttnn
from models.demos.llama3_subdevices.tt.llama_mlp import TtLlamaMLP
from models.demos.llama3_subdevices.tt.model_config import TtModelArgs
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import FeedForward
from models.utility_functions import skip_for_grayskull
from models.demos.llama3_subdevices.tt.prefetcher_common import TtLlamaPrefetcherSetup
from models.demos.llama3_subdevices.tt.llama_ccl import TT_CCL


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "seq_len",
    (32,),
)
@pytest.mark.parametrize(
    "batch_size",
    (32,),
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 23887872,
        }
    ],
    indirect=True,
)
def test_llama_mlp_inference(seq_len, batch_size, mesh_device, use_program_cache, reset_seeds):
    dtype = ttnn.bfloat8_b
    mode = "decode" if seq_len <= 32 else "prefill"

    model_args = TtModelArgs(mesh_device, max_batch_size=batch_size, dummy_weights=False, max_seq_len=128)
    model_args.n_layers = 80
    run_iterations = 2000
    state_dict = model_args.load_state_dict()

    prefetcher_setup = TtLlamaPrefetcherSetup(
        mesh_device,
        n_tensors=3,
        n_layers=model_args.n_layers,
    )
    mesh_device.set_sub_device_stall_group(
        [prefetcher_setup.prefetcher_sub_device_id, prefetcher_setup.worker_sub_device_id]
    )

    tt_ccl = TT_CCL(mesh_device, model_args, prefetcher_setup.worker_sub_device_id)

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    first_layer_prefix = model_args.get_state_dict_prefix("TtLlamaMLP", 0)
    partial_state_dict = {
        k[len(first_layer_prefix) + 1 :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    model_args.WEIGHTS_DTYPE = dtype
    reference_model = FeedForward(
        dim=model_args.dim,
        hidden_dim=4 * model_args.dim,
        multiple_of=model_args.multiple_of,
        ffn_dim_multiplier=model_args.ffn_dim_multiplier,
    )
    reference_model.load_state_dict(partial_state_dict)

    tt_model_layers = [
        TtLlamaMLP(
            mesh_device=mesh_device,
            args=model_args,
            state_dict=state_dict,
            weight_cache_path=model_args.weight_cache_path(dtype),
            layer_num=0,
            dtype=dtype,
            model_config=model_args.get_model_config(),
            prefetcher_setup=prefetcher_setup,
            tt_ccl=tt_ccl,
        )
        for _ in range(model_args.n_layers)
    ]

    tt_tensors = prefetcher_setup.get_input_tensors()

    torch_input = torch.randn(1, 1, seq_len, model_args.dim)
    prev_pcc = None

    logger.info("Run Llama_MLP_PF")
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 3) if model_args.is_galaxy else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),  # When both dims are None, the mapper used is `ReplicateTensorToMesh`
        dtype=ttnn.bfloat8_b,
        memory_config=(
            model_args.model_config["SHARDED_FF12_RING_MEMCFG"]
            if model_args.is_galaxy
            else model_args.model_config["SHARDED_MLP_INPUT_MEMCFG"]
        )
        if mode == "decode"
        else ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_output = tt_input
    compile_iters = 1
    for i in range(compile_iters):
        prefetcher_setup.create_global_cb()
        ttnn.dram_prefetcher(
            tt_tensors,
            num_layers=model_args.n_layers,
            global_cb=prefetcher_setup.global_circular_buffer,
        )
        mesh_device.set_sub_device_stall_group([prefetcher_setup.worker_sub_device_id])

        logger.info("Run Llama_MLP")

        for layer in tt_model_layers:
            tt_output = layer(tt_output, mode=mode)
            tt_output = ttnn.to_memory_config(tt_output, model_args.model_config["SHARDED_FF12_RING_MEMCFG"])

        logger.info("llama MLP Done")

    # capture trace
    logger.info("Capture trace")
    breakpoint()
    tt_ccl.reset_gather_and_buffer_idx()
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)

    prefetcher_setup.create_global_cb()
    ttnn.dram_prefetcher(
        tt_tensors,
        num_layers=model_args.n_layers,
        global_cb=prefetcher_setup.global_circular_buffer,
    )
    mesh_device.set_sub_device_stall_group([prefetcher_setup.worker_sub_device_id])

    logger.info("Run Llama_MLP")

    for layer in tt_model_layers:
        tt_output = layer(tt_output, mode=mode)
        tt_output = ttnn.to_memory_config(tt_output, model_args.model_config["SHARDED_FF12_RING_MEMCFG"])

    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    logger.info("Capture trace done")

    # execute trace
    logger.info("Execute trace")
    for i in range(run_iterations):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        logger.info(f"llama MLP Iteration: {i} done")
    # Release trace
    ttnn.release_trace(mesh_device, trace_id)
    tt_ccl.close()
