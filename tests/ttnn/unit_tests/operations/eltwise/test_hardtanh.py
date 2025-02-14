from loguru import logger
import random
import pytest
import torch
import ttnn
import traceback

from tests.ttnn.utils_for_testing import assert_with_pcc

aten = torch.ops.aten


def load_tensor_from_csv(filename, shape):
    import pandas as pd

    df = pd.read_csv(filename)
    data = df.values
    tensor = torch.tensor(data)
    tensor = tensor.view(shape)
    return tensor


def run_hardtanh_tests(
    file_path,
    input_shape,
    dtype,
    dlayout,
    device,
):
    x = load_tensor_from_csv(file_path[0], input_shape[0]).to(torch.bfloat16)

    try:
        # get ref result
        ref_value = torch.nn.functional.hardtanh(x)

        tt_x = ttnn.from_torch(x, dtype=dtype[0], layout=dlayout[0], device=device)

        tt_result = ttnn.hardtanh(tt_x)
        tt_result = ttnn.to_torch(tt_result)
    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        print(traceback.format_exc())
        raise e

    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape
    pcc = ttnn.pearson_correlation_coefficient(ref_value, tt_result)
    print("PCC = ", pcc)
    assert pcc >= 0.999
    # assert_with_pcc(ref_value, tt_result, 0.999)


test_sweep_args = [
    (
        ["/home/ubuntu/Kalai/tt-metal/ttnn_hardtanh_26.csv"],
        [(1, 1024, 7, 7)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
    ),
]


@pytest.mark.parametrize(
    "file_path, input_shape, dtype, dlayout",
    (test_sweep_args),
)
def test_hardtanh(file_path, input_shape, dtype, dlayout, device):
    run_hardtanh_tests(file_path, input_shape, dtype, dlayout, device)
