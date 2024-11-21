import pytest
import torch
import ttnn
from models.demos.t3000.llama2_70b.tt.model_config import get_model_config, num_to_corerange_set
from models.utility_functions import nearest_32

from datetime import datetime
from tqdm import tqdm


@pytest.fixture
def model_config():
    return get_model_config(llama_version="llama3", max_batch_size=32)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 20000000}], indirect=True)
def test_decode_qkv_matmul(model_config, mesh_device, use_program_cache):
    mesh_device.enable_async(True)
    # Test the QKV matmul in attn_qkv()
    xs = ttnn.as_tensor(
        torch.randn(1, 1, 32, 8192),  # seqlen=1, batch=1, batch_size=32, hidden_size
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=model_config["HIDDEN_WIDTH_16_CORES_MEMCFG"],
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    qkv = ttnn.as_tensor(
        torch.randn(1, 1, 8192, 1280),  # hidden_size x (n_heads + 2*n_kv_heads) * head_dim
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def run_qkv():
        fused_qkv = ttnn.matmul(
            xs,
            qkv,
            program_config=model_config["FUSED_QKV_MM_PROGCFG"],
            memory_config=model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"],
            dtype=ttnn.bfloat16,
            compute_kernel_config=model_config["COMPUTE_KERNEL_CONFIG"],
        )

    loopit("QKV", run_qkv, mesh_device)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 20000000}], indirect=True)
def test_decode_selfout_matmul(model_config, mesh_device, use_program_cache):
    mesh_device.enable_async(True)
    # Test the self-attention output matmul in attn_selfout()
    mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            num_to_corerange_set(8),
            [
                32,
                1024 // 8,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    attn_output = ttnn.as_tensor(
        torch.randn(1, 1, 32, 1024),  # seqlen=1, batch=1, batch_size=32, hidden_size
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    wo = ttnn.as_tensor(
        torch.randn(1, 1, 8192, 1024),
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Reference lines 341-351 for self-attention output matmul
    def run_selfout():
        _, matmul_out, _ = ttnn.experimental.all_gather_matmul(
            attn_output,
            wo,
            dim=3,
            all_gather_core_grid_offset=(0, 4),
            num_links=1,
            memory_config_ag=model_config["ATTN_ALL_GATHER_OUTPUT_MEMCFG"],
            memory_config_mm=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=model_config["SELFOUT_MM_PROGCFG"],
            compute_kernel_config=model_config["COMPUTE_KERNEL_CONFIG"],
        )

    loopit("SELF OUT", run_selfout, mesh_device)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 20000000}], indirect=True)
def test_decode_sdpa(model_config, mesh_device, use_program_cache):
    # Test the scaled dot-product attention matmuls

    mesh_device.enable_async(True)
    query = ttnn.as_tensor(
        torch.randn(1, 32, 8, 128),  # seqlen, n_local_heads, batch_size, head_dim
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=model_config["SDPA_OUTPUT_MEMCFG"],
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    keys = ttnn.as_tensor(
        torch.randn(32, 1, 4096, 128),  # batch_size, n_kv_heads, max_seq_len, head_dim
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    values = ttnn.as_tensor(
        torch.randn(32, 1, 4096, 128),  # batch_size, n_kv_heads, max_seq_len, head_dim
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    pos_ids = ttnn.as_tensor(
        torch.tensor([4000] * 32),  # One position per batch
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Reference lines 304-328 for SDPA matmuls
    def run_sdpa():
        ttnn.transformer.scaled_dot_product_attention_decode(
            query,
            keys,
            values,
            cur_pos_tensor=pos_ids,  # One position per batch
            scale=None,
            program_config=model_config["SDPA_DECODE_PROGRAM_CONFIG"],
            compute_kernel_config=model_config["SDPA_COMPUTE_KERNEL_CONFIG"],
            memory_config=model_config["SDPA_OUTPUT_MEMCFG"],
        )

    loopit("SDPA", run_sdpa, mesh_device)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 20000000}], indirect=True)
def test_decode_ff1_matmul(model_config, mesh_device, use_program_cache):
    mesh_device.enable_async(True)

    weight_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(mesh_device.dram_grid_size().x - 1, mesh_device.dram_grid_size().y - 1),
            )
        }
    )
    H = 8 * 1024
    H4 = 28 * 1024
    w1_shard_shape = (H, nearest_32(H4 // model_config["NUM_DEVICES"] // 12))  # padded cols to divide by 12
    w1_shard_spec = ttnn.ShardSpec(weight_grid, w1_shard_shape, ttnn.ShardOrientation.ROW_MAJOR, False)
    w1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, w1_shard_spec)

    # Test the FF1 matmul in MLP
    xs = ttnn.as_tensor(
        torch.randn(1, 1, 32, 8192),  # seqlen=1, batch=1, batch_size=32, hidden_size
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=model_config["HIDDEN_WIDTH_16_CORES_MEMCFG"],
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    w1 = ttnn.as_tensor(
        torch.randn(1, 1, 8192, 28672),  # hidden_size x mlp_dim
        device=mesh_device,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=w1_mem_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )

    def run_ff1():
        ttnn.matmul(
            xs,
            w1,
            program_config=model_config["PADDED_FF3_MM_PROGCFG"],
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=model_config["COMPUTE_KERNEL_CONFIG_LOFI"],
        )

    loopit("FF1", run_ff1, mesh_device)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 20000000}], indirect=True)
def test_decode_ff2_matmul(model_config, mesh_device, use_program_cache):
    mesh_device.enable_async(True)

    weight_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(mesh_device.dram_grid_size().x - 1, mesh_device.dram_grid_size().y - 1),
            )
        }
    )
    H = 8 * 1024
    H4 = 28 * 1024
    w1_shard_shape = (H4 // model_config["NUM_DEVICES"], nearest_32(H // 12))  # padded cols to divide by 12
    w1_shard_spec = ttnn.ShardSpec(weight_grid, w1_shard_shape, ttnn.ShardOrientation.ROW_MAJOR, False)
    w1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, w1_shard_spec)

    mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            num_to_corerange_set(16),
            [
                32,
                3584 // 16,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    # Test the FF1 matmul in MLP
    xs = ttnn.as_tensor(
        torch.randn(1, 1, 32, 3584),  # seqlen=1, batch=1, batch_size=32, hidden_size
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    w1 = ttnn.as_tensor(
        torch.randn(1, 1, 28672, 8192),  # hidden_size x mlp_dim
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=w1_mem_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
    )

    def run_ff2():
        ttnn.matmul(
            xs,
            w1,
            program_config=model_config["PADDED_FF2_MM_PROGCFG"],
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=model_config["COMPUTE_KERNEL_CONFIG"],
        )

    loopit("FF2", run_ff2, mesh_device)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 20000000}], indirect=True)
def test_decode_lm_head_matmul(model_config, mesh_device, use_program_cache):
    mesh_device.enable_async(True)
    # Test the LM head matmul
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            num_to_corerange_set(32),
            [
                32,
                8192 // 32,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    xs = ttnn.as_tensor(
        torch.randn(1, 1, 32, 8192),  # seqlen=1, batch=1, batch_size=32, hidden_size
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    lm_head = ttnn.as_tensor(
        torch.randn(1, 1, 8192, 16 * 1024),  # hidden_size x padded_vocab_size
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 4),
        in0_block_w=8,
        out_subblock_h=1,
        out_subblock_w=4,
        per_core_M=1,
        per_core_N=16,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )

    def run_lm_head():
        ttnn.matmul(
            xs,
            lm_head,
            program_config=program_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=model_config["COMPUTE_KERNEL_CONFIG"],
        )

    loopit("LM HEAD", run_lm_head, mesh_device)


def update_heartbeat():
    """Write current timestamp to a heartbeat file"""
    heartbeat_file = f"/tmp/heartbeat.txt"
    with open(heartbeat_file, "w") as f:
        f.write(str(datetime.now().timestamp()))


def loopit(name, callable, mesh_device):
    """
    given a callable which takes no arguments, capture a trace and then execute
    the callable 10000 times, updating the heartbeat whenever necessary.
    every 10 iterations, log the name and the time elapsed
    """
    # Compile it
    # callable()
    # trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    # callable()
    # ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    print(f"loopit {name}")
    for i in tqdm(range(5_000_000)):
        # ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        callable()
        if i % 100 == 0:
            for dev in mesh_device.get_devices():
                ttnn.synchronize_device(dev)
            update_heartbeat()
