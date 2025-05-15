import ttnn
import torch


def test_matmul_broadcast(device):
    x = torch.randn(1, 1, 1, 1, 3, 3)
    y = torch.randn(1, 4, 160, 160, 3, 1)
    golden = torch.matmul(x, y)

    tt_x = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)
    tt_y = ttnn.from_torch(y, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = ttnn.matmul(tt_x, tt_y)
    tt_out = ttnn.to_torch(tt_out)

    assert ttnn.pearson_correlation_coefficient(golden, tt_out) >= 0.99
