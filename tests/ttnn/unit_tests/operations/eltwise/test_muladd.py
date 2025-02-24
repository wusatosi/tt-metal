import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_grayskull


@skip_for_grayskull()
@pytest.mark.parametrize(
    "input_shapes",
    [
        [5, 3, 64 * 8, 64 * 8],
    ],
)
def test_muladd(device, input_shapes):
    torch.manual_seed(0)
    torch_ina = torch.randn(input_shapes, dtype=torch.bfloat16)
    torch_inb = torch.randn(input_shapes, dtype=torch.bfloat16)
    torch_inc = torch.randn(input_shapes, dtype=torch.bfloat16)
    torch_ind = torch.rand(input_shapes, dtype=torch.bfloat16) + 1

    # torch_ina = torch.ones(input_shapes, dtype=torch.bfloat16)
    # torch_inb = torch.ones(input_shapes, dtype=torch.bfloat16)
    # torch_inc = torch.ones(input_shapes, dtype=torch.bfloat16)
    # torch_ind = torch.ones(input_shapes, dtype=torch.bfloat16)

    in_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    out_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    tt_ina = ttnn.from_torch(
        torch_ina,
        tile=ttnn.Tile((32, 32)),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=in_memory_config,
    )
    tt_inb = ttnn.from_torch(
        torch_inb,
        tile=ttnn.Tile((32, 32)),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=in_memory_config,
    )
    tt_inc = ttnn.from_torch(
        torch_inc,
        tile=ttnn.Tile((32, 32)),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=in_memory_config,
    )
    tt_ind = ttnn.from_torch(
        torch_ind,
        tile=ttnn.Tile((32, 32)),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=in_memory_config,
    )
    torch_output = torch.div((torch_ina + torch_inb) * torch_inc, torch_ind)

    output = ttnn.muladd(
        tt_ina,
        tt_inb,
        tt_inc,
        tt_ind,
        dtype=ttnn.bfloat16,
        memory_config=out_memory_config,
        math_fidelity=ttnn.MathFidelity.HiFi2,
    )

    print(output)
    output = ttnn.to_torch(output)

    pcc = ttnn.pearson_correlation_coefficient(torch_output, output)
    print(f"PCC: {pcc}")
    assert pcc >= 0.99
