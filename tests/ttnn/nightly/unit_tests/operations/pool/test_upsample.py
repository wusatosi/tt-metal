import ttnn
import pytest
import torch

from tests.ttnn.utils_for_testing import assert_with_pcc


def write_to_file(file_name, tensor):
    tensor = tensor.cpu().detach().numpy()
    with open(file_name, "w") as f:
        for i in range(1):
            for j in range(tensor.shape[1]):
                for k in range(tensor.shape[2]):
                    for l in range(tensor.shape[3]):
                        # f.write(str(round(tensor[i][j][k][l]), 2) + " ")
                        f.write("{:.2f}".format(tensor[i][j][k][l]) + " ")
                    f.write("\n")


def init_tensor(tensor, shape):
    for batch in range(shape[0]):
        for channels in range(shape[1]):
            for h in range(shape[2]):
                for w in range(shape[3]):
                    tensor[batch, channels, h, w] = h * shape[3] + w + 1

    return tensor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_upsample(device):
    in_channels = 256
    input_shape = [1, in_channels, 10, 66]
    input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    # input_tensor = init_tensor(input_tensor, input_shape)
    # input_tensor = torch.ones(input_shape, dtype=torch.bfloat16) * 3.12
    input_shard_shape = (20, in_channels)
    core_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3)),
            ttnn.CoreRange(ttnn.CoreCoord(0, 4), ttnn.CoreCoord(0, 4)),
        }
    )
    input_shard_spec = ttnn.ShardSpec(core_grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    # torch_upsample = ttnn.upsample.golden_function(input_tensor, (2, 2))
    torch_upsample = torch.nn.Upsample(scale_factor=(2, 2), mode="nearest")
    expected = torch_upsample(input_tensor)

    ttnn_input_tensor = input_tensor.permute(0, 2, 3, 1)
    ttnn_input_tensor = ttnn.from_torch(ttnn_input_tensor, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn_input_tensor = ttnn.to_memory_config(ttnn_input_tensor, memory_config=input_mem_config)
    actual = ttnn.upsample(ttnn_input_tensor, (2, 2))
    actual = ttnn.to_torch(actual)
    expected = expected.permute(0, 2, 3, 1)

    write_to_file("output.txt", expected.float())
    write_to_file("ttnn.txt", actual.float())
    assert_with_pcc(expected, actual)
