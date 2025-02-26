import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_grayskull


@pytest.mark.parametrize("input_shapes", [[1, 1, 3 * 64 * 32, 4 * 32]])
@pytest.mark.parametrize("ina_sharded", [True, False])
@pytest.mark.parametrize("inb_sharded", [True, False])
@pytest.mark.parametrize("inc_sharded", [True, False])
@pytest.mark.parametrize("ind_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
def test_muladd(device, input_shapes, ina_sharded, inb_sharded, inc_sharded, ind_sharded, out_sharded):
    torch.manual_seed(0)

    compute_grid_size = device.compute_with_storage_grid_size()
    compute_grid_x = compute_grid_size.x
    compute_grid_y = compute_grid_size.y
    # compute_grid_x=1
    # compute_grid_y =4

    base = [[i + j * 32 for i in range(4 * 32)] for j in range(64 * 32)]
    # torch_ina = torch.concatenate((torch.ones(32,32,dtype=torch.bfloat16),torch.ones(32,32,dtype=torch.bfloat16)*2),dim=1)

    # torch_ina = torch.ones(32,32,dtype=torch.bfloat16)

    torch_ina = torch.randn(input_shapes, dtype=torch.bfloat16)
    torch_inb = torch.randn(input_shapes, dtype=torch.bfloat16)
    torch_inc = torch.randn(input_shapes, dtype=torch.bfloat16)
    torch_ind = torch.rand(input_shapes, dtype=torch.bfloat16) + 1

    # torch_ina = torch.concatenate((torch.tensor(base,dtype=torch.bfloat16),torch.tensor(base,dtype=torch.bfloat16)*2),dim=1)
    # torch_inb = torch.concatenate((torch.tensor(base,dtype=torch.bfloat16)*4,torch.tensor(base,dtype=torch.bfloat16)*3),dim=1)
    # torch_inc = torch.concatenate((torch.tensor(base,dtype=torch.bfloat16)*4,torch.tensor(base,dtype=torch.bfloat16)*6),dim=1)
    # torch_ind = torch.concatenate((torch.tensor(base,dtype=torch.bfloat16)*10,torch.tensor(base,dtype=torch.bfloat16)*5),dim=1)

    # base = [[ j*i for i in range(32)] for j in range(input_shapes[2])]
    # torch_ina = torch.tensor(base, dtype=torch.bfloat16) *1
    # torch_inb = torch.tensor(base, dtype=torch.bfloat16) *2
    # torch_inc = torch.tensor(base, dtype=torch.bfloat16) *4
    # torch_ind = torch.tensor(base, dtype=torch.bfloat16) *2

    in_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    out_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    if out_sharded:
        out_memory_config = ttnn.create_sharded_memory_config(
            input_shapes,
            core_grid=ttnn.CoreGrid(x=compute_grid_x, y=compute_grid_y),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )

    ina_memory_config = in_memory_config
    if ina_sharded:
        ina_memory_config = ttnn.create_sharded_memory_config(
            input_shapes,
            core_grid=ttnn.CoreGrid(x=compute_grid_x, y=compute_grid_y),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )

    inb_memory_config = in_memory_config
    if inb_sharded:
        inb_memory_config = ttnn.create_sharded_memory_config(
            input_shapes,
            core_grid=ttnn.CoreGrid(x=compute_grid_x, y=compute_grid_y),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )

    inc_memory_config = in_memory_config
    if inc_sharded:
        inc_memory_config = ttnn.create_sharded_memory_config(
            input_shapes,
            core_grid=ttnn.CoreGrid(x=compute_grid_x, y=compute_grid_y),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )

    ind_memory_config = in_memory_config
    if ind_sharded:
        ind_memory_config = ttnn.create_sharded_memory_config(
            input_shapes,
            core_grid=ttnn.CoreGrid(x=compute_grid_x, y=compute_grid_y),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )

    tt_ina = ttnn.from_torch(
        torch_ina,
        tile=ttnn.Tile((32, 32)),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=ina_memory_config,
    )
    tt_inb = ttnn.from_torch(
        torch_inb,
        tile=ttnn.Tile((32, 32)),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=inb_memory_config,
    )
    tt_inc = ttnn.from_torch(
        torch_inc,
        tile=ttnn.Tile((32, 32)),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=inc_memory_config,
    )
    tt_ind = ttnn.from_torch(
        torch_ind,
        tile=ttnn.Tile((32, 32)),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=ind_memory_config,
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

    # print(torch_ina)
    # print("\n\n")
    print(output)
    output = ttnn.to_torch(output)
    pcc = ttnn.pearson_correlation_coefficient(torch_output, output)
    print(f"PCC: {pcc}")
    assert pcc >= 0.99
