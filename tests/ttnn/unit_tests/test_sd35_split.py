import torch
import ttnn
import pytest


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_split(device, reset_seeds):
    torch_hidden_states = torch.randn(2, 1024, 1536, dtype=torch.bfloat16)
    torch_emb = torch.randn(2, 1536, dtype=torch.bfloat16)
    torch_weight = torch.randn(1536, 13824, dtype=torch.bfloat16)
    torch_bias = torch.randn(1, 13824, dtype=torch.bfloat16)

    ttnn_input_hidden_states = ttnn.from_torch(
        torch_hidden_states, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
    )
    ttnn_input_emb = ttnn.from_torch(
        torch_emb, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    ttnn_weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    ttnn_bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)

    emb = ttnn.silu(ttnn_input_emb)
    emb = ttnn.linear(emb, ttnn_weight, bias=ttnn_bias, memory_config=ttnn.L1_MEMORY_CONFIG)

    emb = ttnn.to_layout(emb, ttnn.ROW_MAJOR_LAYOUT)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_msa2, scale_msa2, gate_msa2 = ttnn.split(
        emb, 9, 1
    )
