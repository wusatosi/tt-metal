import pytest
import ttnn
import torch
from tests.ttnn.utils_for_testing import assert_with_pcc


def test_linear_sd35(device):
    input_tens = torch.randn(2, 4096, 1536)
    weights_tens = torch.randn(1536, 6144)
    bias_tens = torch.randn(6144)
    mm_a_y = 8
    mm_a_x = 8
    mm_a_x_strategy = ttnn.ShardStrategy.BLOCK
    mm_a_x_memory_config = ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
    input_memory_config = ttnn.create_sharded_memory_config(
        input_tens.shape,
        core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_a_x),
        strategy=mm_a_x_strategy,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    ttnn_input = ttnn.from_torch(
        input_tens,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        device=device,
        memory_config=input_memory_config,
    )
    hifi2_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
    )
    tt_w = ttnn.from_torch(
        weights_tens,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        device=device,
    )
    tt_b = ttnn.from_torch(
        bias_tens,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        device=device,
    )
    torch_output_tensor = torch.nn.functional.linear(input_tens, weights_tens.T.contiguous(), bias=bias_tens)
    tt_out = ttnn.linear(
        ttnn_input,
        input_tensor_b=tt_w,
        bias=tt_b,
        core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_a_x),
        compute_kernel_config=hifi2_kernel_config,
        memory_config=mm_a_x_memory_config,
    )
    tt_out_torch = ttnn.to_torch(tt_out)
    print(tt_out.shape, torch_output_tensor.shape)
    assert_with_pcc(tt_out_torch, torch_output_tensor, 0.99)
