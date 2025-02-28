import torch
import pytest
import ttnn
from enum import Enum


class ShardStrat(Enum):
    NONE = 0
    BLOCK = 1
    HEIGHT = 2


@pytest.mark.parametrize("input_shapes", [[1, 1, 16 * 16 * 32, 2 * 8 * 32], [1, 1, 3 * 8 * 8 * 32, 3 * 8 * 32]])
@pytest.mark.parametrize("ina_shard_strat", [ShardStrat.NONE, ShardStrat.BLOCK, ShardStrat.HEIGHT])
@pytest.mark.parametrize("inb_shard_strat", [ShardStrat.NONE, ShardStrat.BLOCK, ShardStrat.HEIGHT])
@pytest.mark.parametrize("inc_shard_strat", [ShardStrat.NONE, ShardStrat.BLOCK, ShardStrat.HEIGHT])
@pytest.mark.parametrize("ind_shard_strat", [ShardStrat.NONE, ShardStrat.BLOCK, ShardStrat.HEIGHT])
@pytest.mark.parametrize("out_shard_strat", [ShardStrat.NONE, ShardStrat.BLOCK, ShardStrat.HEIGHT])
def test_muladd(
    device, input_shapes, ina_shard_strat, inb_shard_strat, inc_shard_strat, ind_shard_strat, out_shard_strat
):
    torch.manual_seed(0)

    shard_strats = set([ina_shard_strat, inb_shard_strat, inc_shard_strat, ind_shard_strat, out_shard_strat])
    if ShardStrat.BLOCK in shard_strats and ShardStrat.HEIGHT in shard_strats:
        pytest.skip("Cannot test both block and height sharding at the same time")

    compute_grid_size = device.compute_with_storage_grid_size()
    compute_grid_x = compute_grid_size.x
    compute_grid_y = compute_grid_size.y
    torch_ina = torch.randn(input_shapes, dtype=torch.bfloat16)
    torch_inb = torch.randn(input_shapes, dtype=torch.bfloat16)
    torch_inc = torch.randn(input_shapes, dtype=torch.bfloat16)
    torch_ind = torch.rand(input_shapes, dtype=torch.bfloat16) + 1

    ina_sharded = ina_shard_strat != ShardStrat.NONE
    inb_sharded = inb_shard_strat != ShardStrat.NONE
    inc_sharded = inc_shard_strat != ShardStrat.NONE
    ind_sharded = ind_shard_strat != ShardStrat.NONE
    out_sharded = out_shard_strat != ShardStrat.NONE

    in_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    out_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    if out_sharded:
        out_memory_config = ttnn.create_sharded_memory_config(
            input_shapes,
            core_grid=ttnn.CoreGrid(x=compute_grid_x, y=compute_grid_y),
            strategy=ttnn.ShardStrategy.BLOCK if out_shard_strat == ShardStrat.BLOCK else ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )

    ina_memory_config = in_memory_config
    if ina_sharded:
        ina_memory_config = ttnn.create_sharded_memory_config(
            input_shapes,
            core_grid=ttnn.CoreGrid(x=compute_grid_x, y=compute_grid_y),
            strategy=ttnn.ShardStrategy.BLOCK if ina_shard_strat == ShardStrat.BLOCK else ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )

    inb_memory_config = in_memory_config
    if inb_sharded:
        inb_memory_config = ttnn.create_sharded_memory_config(
            input_shapes,
            core_grid=ttnn.CoreGrid(x=compute_grid_x, y=compute_grid_y),
            strategy=ttnn.ShardStrategy.BLOCK if inb_shard_strat == ShardStrat.BLOCK else ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )

    inc_memory_config = in_memory_config
    if inc_sharded:
        inc_memory_config = ttnn.create_sharded_memory_config(
            input_shapes,
            core_grid=ttnn.CoreGrid(x=compute_grid_x, y=compute_grid_y),
            strategy=ttnn.ShardStrategy.BLOCK if inc_shard_strat == ShardStrat.BLOCK else ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )

    ind_memory_config = in_memory_config
    if ind_sharded:
        ind_memory_config = ttnn.create_sharded_memory_config(
            input_shapes,
            core_grid=ttnn.CoreGrid(x=compute_grid_x, y=compute_grid_y),
            strategy=ttnn.ShardStrategy.BLOCK if ind_shard_strat == ShardStrat.BLOCK else ttnn.ShardStrategy.HEIGHT,
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
    output = ttnn.to_torch(output)
    pcc = ttnn.pearson_correlation_coefficient(torch_output, output)
    print(f"PCC: {pcc}")
    assert pcc >= 0.99
