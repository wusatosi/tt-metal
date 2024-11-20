import ttnn
import torch


def test_fail_program_cache(device):
    device.enable_program_cache()

    """
    This test will run two different ops. The first op has been modified to have the total number of rt args to be > 16128
    """

    a = torch.ones(32, 1, 32, 32)
    b = torch.ones(32, 1, 32, 32)

    a = ttnn.from_torch(a, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b = ttnn.from_torch(b, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    for i in range(10):
        print(f"{i=}")
        c = ttnn.mul(a, b)
        d = ttnn.exp(c)
        ttnn.synchronize_device(device)
