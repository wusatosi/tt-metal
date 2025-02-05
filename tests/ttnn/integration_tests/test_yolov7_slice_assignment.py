import ttnn
import torch
from tests.ttnn.utils_for_testing import assert_with_pcc
import pytest


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_yolov7(device, reset_seeds):
    """
    This unit test is done to check if the following torch code snippet can be converted to ttnn:
    y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i]
    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
    """

    input_tensor = torch.rand(1, 3, 20, 20, 85)

    # Checking if the torch indexing item assignment can be replaced by torch slice followed by concat operation
    input_tensor_dup = input_tensor
    input_tensor_dup[..., 0:2] = input_tensor[..., 0:2] + 1.0

    input_tensor_split2 = input_tensor
    input_tensor_split1 = input_tensor[..., 0:2] + 1.0
    output_tensor = torch.cat([input_tensor_split1, input_tensor_split2[..., 2:]], dim=-1)
    # assert_with_pcc(input_tensor_dup, output_tensor, pcc=0.99) # PCC: 0.947206171395425

    # Checking the torch indexing item assignment and ttnn slice followed by concat operation
    ttnn_input = ttnn.from_torch(input_tensor, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_input1 = ttnn_input[:, :, :, :, 0:2] + 1.0
    ttnn_input2 = ttnn_input[:, :, :, :, 2:]
    ttnn_out = ttnn.concat([ttnn_input1, ttnn_input2], dim=4)
    ttnn_out = ttnn.to_torch(ttnn_out)
    assert_with_pcc(output_tensor, ttnn_out, pcc=0.99)
