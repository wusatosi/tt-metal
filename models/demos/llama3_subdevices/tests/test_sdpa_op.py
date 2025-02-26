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
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull
from tests.ttnn.unit_tests.operations.prefetcher_common import TtLlamaPrefetcherSetup
from models.demos.llama3_subdevices.tt.llama_ccl import TT_CCL


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize("device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=True)
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
        # True,
        False,
    ),
    ids=(
        # "paged_attention",
        "default_attention",
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
def test_sdpa_op(
    max_seq_len,
    batch_size,
    paged_attention,
    page_params,
    mesh_device,
    use_program_cache,
    reset_seeds,
    ensure_gc,
):
    dtype = ttnn.bfloat8_b
    pcc = 0.99

    # mesh_device.enable_async(True)

    model_args = TtModelArgs(mesh_device, dummy_weights=True, max_batch_size=batch_size, max_seq_len=max_seq_len)
    model_args.n_layers = 1  # For the unit test, just run a sigle layer

    attn_input = torch.randn((1, 8, 8, 128), dtype=torch.float32)
    keys = torch.zeros((8, 1, 256, 128), dtype=torch.float32)
    values = torch.zeros((8, 1, 256, 128), dtype=torch.float32)
    current_pos = torch.ones((1), dtype=torch.int32) * 127
    scale = 0.08838834764831845
    # create grid for this {[(x=1,y=0) - (x=3,y=1)], [(x=1,y=2) - (x=2,y=2)]}
    core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 1)),
            ttnn.CoreRange(ttnn.CoreCoord(1, 2), ttnn.CoreCoord(2, 2)),
        ]
    )

    # shard_spec = ttnn.ShardSpec(
    #     core_grid,
    #     [32, 128],
    #     shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
    # )
    mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            core_grid,
            [32, 128],
            shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    # put attn_input on the device
    tt_attn_input = ttnn.as_tensor(
        attn_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=mem_config,
    )

    tt_keys = ttnn.as_tensor(
        keys,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    )

    tt_values = ttnn.as_tensor(
        values,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    )

    tt_current_pos = ttnn.as_tensor(
        current_pos,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    )
    batch_size_per_device_group = 8

    print(f'program_config: {model_args.model_config["SDPA_DECODE_PROGCFG"]}')
    print(f'compute_kernel_config: {model_args.model_config["SDPA_DECODE_COMPUTE_PROGCFG"]}')
    print(f'memory_config: {model_args.model_config["SCORES_BATCHED_MM_OUTPUT_MEMCFG"](batch_size_per_device_group)}')

    sdpa_op = ttnn.transformer.scaled_dot_product_attention_decode(
        tt_attn_input,
        tt_keys,
        tt_values,
        tt_current_pos,
        scale,
        program_config=model_args.model_config["SDPA_DECODE_PROGCFG"],
        compute_kernel_config=model_args.model_config["SDPA_DECODE_COMPUTE_PROGCFG"],
        memory_config=model_args.model_config["SCORES_BATCHED_MM_OUTPUT_MEMCFG"](batch_size_per_device_group),
    )

    # get results from the device
    sdpa_op_result = ttnn.from_torch(sdpa_op)

    print(sdpa_op_result)
    print(sdpa_op_result.shape)

    # prepare attn_input
