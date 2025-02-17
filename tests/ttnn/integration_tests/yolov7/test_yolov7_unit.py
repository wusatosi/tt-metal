import ttnn
import torch
from tests.ttnn.utils_for_testing import assert_with_pcc


def test_workaround(device, reset_seeds):
    y = torch.randn(1, 3, 80, 80, 85)
    grid = torch.randn(1, 1, 80, 80, 2)
    stride = torch.tensor([8.0, 16.0, 32.0])[0]
    anchor_grid = torch.randn(1, 3, 1, 1, 2)

    torch_y = y.clone()
    ttnn_y = y.clone()
    test_y = y.clone()

    ## Original implementation
    test_y[..., 0:2] = (test_y[..., 0:2] * 2.0 - 0.5 + grid) * stride
    test_y[..., 2:4] = (test_y[..., 2:4] * 2) ** 2 * anchor_grid

    # Alternative work for the above implementation
    scaled_xy = torch_y[..., 0:2]
    scaled_xy = scaled_xy * 2.0
    shifted_xy = scaled_xy - 0.5
    grid_offset_xy = shifted_xy + grid
    torch_y[..., 0:2] = grid_offset_xy * stride

    scaled_wh = torch_y[..., 2:4]
    scaled_wh = scaled_wh * 2
    squared_wh = scaled_wh**2
    squared_wh = squared_wh * anchor_grid
    torch_y[..., 2:4] = squared_wh
    assert_with_pcc(test_y, torch_y)  # PASSED with PCC = 1.0

    # TTNN implementation for the alternative method.
    ttnn_grid = ttnn.from_torch(grid, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_anchor_grid = ttnn.from_torch(anchor_grid, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    scaled_xy = ttnn_y[..., 0:2]
    scaled_xy = ttnn.from_torch(scaled_xy, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    scaled_xy = scaled_xy * 2.0
    shifted_xy = scaled_xy - 0.5
    grid_offset_xy = ttnn.add(shifted_xy, ttnn_grid)
    ttnn_y[..., 0:2] = ttnn.to_torch(grid_offset_xy * stride)  # indexing item assignment is not supported in ttnn.

    scaled_wh = ttnn_y[..., 2:4]
    scaled_wh = ttnn.from_torch(scaled_wh, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    scaled_wh = ttnn.mul(scaled_wh, 2)
    squared_wh = ttnn.pow(scaled_wh, 2)
    squared_wh = squared_wh * ttnn_anchor_grid
    ttnn_y[..., 2:4] = ttnn.to_torch(squared_wh)

    assert_with_pcc(torch_y, ttnn_y)  # FAILED PCC = 0.7678344808851415
