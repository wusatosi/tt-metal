import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def test_my_op(device):
    input_tensor_a = torch.ones(128, 128) * 4
    input_tensor_b = torch.ones(128, 128) * 5

    tt_input_tensor_a = ttnn.from_torch(
        input_tensor_a,
        ttnn.bfloat16,
        pad_value=0,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_input_tensor_b = ttnn.from_torch(
        input_tensor_b,
        ttnn.bfloat16,
        pad_value=0,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    scalar = 1.0

    tt_output_tensor = ttnn.my_operation(tt_input_tensor_a, tt_input_tensor_b, scalar)
    torch_output_tensor = torch.mul(torch.sqrt(input_tensor_a + input_tensor_b), scalar)

    assert_with_pcc(
        ttnn.from_device(tt_output_tensor).to_torch(),
        torch_output_tensor,
        pcc=0.99,
    )
