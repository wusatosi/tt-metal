# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import os
import ttnn
from models.demos.llama3_subdevices.tt.llama_attention import TtLlamaAttention
from models.demos.llama3_subdevices.tt.llama_rope import TtLlamaRotarySetup
from models.demos.llama3_subdevices.tt.model_config import TtModelArgs
from models.demos.llama3_subdevices.tt.llama_common import (
    precompute_freqs,
    PagedAttentionConfig,
)
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import Attention
from models.utility_functions import skip_for_grayskull
from models.demos.llama3_subdevices.tt.prefetcher_common import TtLlamaPrefetcherSetup
from models.demos.llama3_subdevices.tt.llama_ccl import TT_CCL


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
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
    "paged_attention",
    (
        True,
        # False,
    ),
    ids=(
        "paged_attention",
        # "default_attention",
    ),
)
@pytest.mark.parametrize(
    "page_params",
    [{"page_block_size": 32, "page_max_num_blocks": 1024}],
)
@pytest.mark.parametrize(
    "batch_size",
    (32,),
)
@pytest.mark.parametrize(
    "max_seq_len",
    (256,),  # For decode-only unit test, there's no need to run with large sequence lengths
)
def test_llama_attention_inference(
    max_seq_len,
    batch_size,
    paged_attention,
    page_params,
    mesh_device,
    use_program_cache,
    reset_seeds,
):
    dtype = ttnn.bfloat8_b
    pcc = 0.99

    model_args = TtModelArgs(mesh_device, dummy_weights=False, max_batch_size=batch_size, max_seq_len=max_seq_len)
    model_args.n_layers = 80  # For the unit test, just run a sigle layer

    state_dict = model_args.load_state_dict()

    first_layer_prefix = model_args.get_state_dict_prefix("TtLlamaAttention", 0) + "."
    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    reference_model = Attention(args=model_args)
    reference_model.load_state_dict(partial_state_dict)

    seq_len = 1

    generation_start_pos = 127
    generation_length = 20000
    all_tests_pass = True
    enable_performance_mode = True

    # Setup RoPE transformation matrices
    rope_setup = TtLlamaRotarySetup(
        mesh_device,
        batch_size,
        model_args.head_dim,
        model_args.max_seq_len,
        model_args.rope_theta,
        model_args.use_scaled_rope,
        model_args.rope_scaling_factor,
    )

    transformation_mats = rope_setup.get_both_trans_mats()

    page_table_tt = None
    paged_attention_config = None

    if paged_attention:
        paged_attention_config = PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks"],
        )

        # Implied shuffling of blocks
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        # Page table which maps virtual blocks to physical
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(
            model_args.batch_size_per_device_group,
            paged_attention_config.max_num_blocks // model_args.batch_size_per_device_group,
        )
        page_table_tt = ttnn.from_torch(
            page_table,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, None),
                mesh_shape=model_args.cluster_shape,
            ),
        )

    prefetcher_setup = TtLlamaPrefetcherSetup(
        mesh_device,
        n_tensors=2,
        n_layers=model_args.n_layers,
    )
    mesh_device.set_sub_device_stall_group(
        [prefetcher_setup.prefetcher_sub_device_id, prefetcher_setup.worker_sub_device_id]
    )

    tt_ccl = TT_CCL(mesh_device, model_args, prefetcher_setup.worker_sub_device_id)

    tt_attention_layers = [
        TtLlamaAttention(
            mesh_device,
            state_dict,
            weight_cache_path=model_args.weight_cache_path(dtype),
            layer_num=i,
            dtype=dtype,
            transformation_mats=transformation_mats,
            configuration=model_args,
            paged_attention_config=paged_attention_config,
            prefetcher_setup=prefetcher_setup,
            tt_ccl=tt_ccl,
        )
        for i in range(model_args.n_layers)
    ]

    tt_tensors = prefetcher_setup.get_input_tensors()

    cos, sin = precompute_freqs(
        model_args.head_dim,
        model_args.max_seq_len * 2,
        model_args.rope_theta,
        model_args.use_scaled_rope,
        model_args.rope_scaling_factor,
    )
    freqs_cis = torch.complex(cos, sin)

    # Initial positions
    current_pos = torch.tensor([generation_start_pos for _ in range(batch_size)])
    current_pos_tensor = ttnn.from_torch(
        current_pos,
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0) if (model_args.is_galaxy and batch_size > 1) else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
    )
    # Explicitly allocate global CB to avoid memory fragmentation

    # 70B attention block typically sees tensors with mean 0 and std 0.03 - 0.05 in layer 1
    pt_attention_input = torch.randn(batch_size, seq_len, model_args.dim) * 0.05

    tt_attention_input = pt_attention_input.clone()

    attention_output = model_args.prepare_residual_tensor_decode(
        tt_attention_input,
        model_args.model_config["SHARDED_ATTN_INPUT_RING_MEMCFG"],
        force_replicated=False if model_args.is_galaxy else True,
    )

    # Get cos/sin matrices for the current position of each user
    rot_mats = rope_setup.get_rot_mats(current_pos)

    compile_iters = 1
    for i in range(compile_iters):
        prefetcher_setup.create_global_cb()
        ttnn.dram_prefetcher(
            tt_tensors,
            num_layers=model_args.n_layers,
            global_cb=prefetcher_setup.global_circular_buffer,
            enable_performance_mode=enable_performance_mode,
        )
        mesh_device.set_sub_device_stall_group([prefetcher_setup.worker_sub_device_id])

        for layer in tt_attention_layers:
            attention_output = layer(
                attention_output,
                current_pos_tensor,
                rot_mats=rot_mats,
                mode="decode",
                page_table=page_table_tt,
            )
            attention_output = ttnn.to_memory_config(
                attention_output,
                model_args.model_config["SHARDED_ATTN_INPUT_RING_MEMCFG"],
            )
    logger.info("Capture trace begin")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    prefetcher_setup.create_global_cb()
    ttnn.dram_prefetcher(
        tt_tensors,
        num_layers=model_args.n_layers,
        global_cb=prefetcher_setup.global_circular_buffer,
        enable_performance_mode=enable_performance_mode,
    )
    mesh_device.set_sub_device_stall_group([prefetcher_setup.worker_sub_device_id])
    for layer in tt_attention_layers:
        attention_output = layer(
            attention_output,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table_tt,
        )
        attention_output = ttnn.to_memory_config(
            attention_output,
            model_args.model_config["SHARDED_ATTN_INPUT_RING_MEMCFG"],
        )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    logger.info("Capture trace done")

    logger.info("run trace")
    for i in range(generation_length):
        logger.info(f"[Attention] Generating token {i}")
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)

    tt_ccl.close()
