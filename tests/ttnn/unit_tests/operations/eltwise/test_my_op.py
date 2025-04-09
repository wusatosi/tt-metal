import pytest
import torch
import ttnn
import pdb


def test_my_op(device):
    input_tensor_a = torch.ones(128, 128) * 9
    input_tensor_b = torch.ones(128, 128) * 16

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
    scalar = 2.0

    tt_output_tensor = ttnn.my_operation(tt_input_tensor_a, tt_input_tensor_b, scalar)
    pdb.set_trace()
    print(tt_output_tensor)
