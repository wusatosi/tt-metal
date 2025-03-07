import ttnn
import torch

from tests.ttnn.utils_for_testing import assert_with_pcc


def test_silu(device):
    shape = [1, 1, 8192, 320]

    block_sharded = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(7, 7),
                    ),
                }
            ),
            [1024, 40],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    torch_input = torch.randn(shape).bfloat16().float()
    torch_output = torch.nn.SiLU()(torch_input)

    input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    input = ttnn.to_device(input, device, memory_config=block_sharded)

    output = ttnn.silu(input)
    resharded = output.cpu().to_torch()

    _, message = assert_with_pcc(resharded, torch_output, 0.99)
    print(message)
