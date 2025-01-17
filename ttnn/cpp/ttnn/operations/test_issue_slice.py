import torch
import ttnn
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test(device, reset_seeds):
    a = torch.randn(1, 3, dtype=torch.bfloat16)
    a_ttnn = ttnn.from_torch(a, device=device)
    print("a", a)  # tensor([[ 1.1016, -0.8672,  1.0469]], dtype=torch.bfloat16)
    print(
        "a_ttnn", a_ttnn
    )  # ttnn.Tensor([[ 1.10156, -0.86719,  1.04688]], shape=Shape([1, 3]), dtype=DataType::BFLOAT16, layout=Layout::ROW_MAJOR)
    print("a[0][1]", a[0:1, 1:2])  # tensor([[-0.8672]], dtype=torch.bfloat16)
    print(
        "a_ttnn[0][1]", a_ttnn[0:1, 1:2]
    )  # ttnn.Tensor([[ 1.10156]], shape=Shape([1, 1]), dtype=DataType::BFLOAT16, layout=Layout::ROW_MAJOR)
    print(
        "a_ttnn[0][1]", ttnn.slice(a_ttnn, [0, 1], [1, 2])
    )  # a_ttnn[0][1] ttnn.Tensor([[ 1.10156]], shape=Shape([1, 1]), dtype=DataType::BFLOAT16, layout=Layout::ROW_MAJOR)
    assert a[0:1, 1:2] == ttnn.to_torch(a_ttnn[0:1, 1:2]), "False"
