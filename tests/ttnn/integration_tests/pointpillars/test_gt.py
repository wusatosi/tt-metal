import torch
import ttnn
import pytest

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_greater(device, reset_seeds):
    int_tensor = torch.randint(low=0, high=100, size=(4352, 1), dtype=torch.int)
    max_num = torch.arange(64, dtype=torch.int).view([1, 64])

    greater_output = int_tensor > max_num
    print("greater_output output", greater_output.shape)

    ttnn_int_tensor = ttnn.from_torch(int_tensor, device=device)
    ttnn_max_num = ttnn.unsqueeze(ttnn.arange(0, 64, device=device, dtype=ttnn.uint32), dim=0)
    ttnn_int_tensor = ttnn.to_layout(ttnn_int_tensor, layout=ttnn.TILE_LAYOUT)
    ttnn_max_num = ttnn.to_layout(ttnn_max_num, layout=ttnn.TILE_LAYOUT)

    paddings_indicator = ttnn.gt(ttnn_int_tensor, ttnn_max_num, use_legacy=False)

    print("paddings_indicator", paddings_indicator)

    print("greater_output,", greater_output)
    print("paddings_indicator", paddings_indicator)
    print("ttnn.to_torch(paddings_indicator)", ttnn.to_torch(paddings_indicator))

    assert_with_pcc(greater_output, ttnn.to_torch(paddings_indicator), pcc=0.99)
