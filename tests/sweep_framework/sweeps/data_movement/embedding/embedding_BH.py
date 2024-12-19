# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
import random
import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

TIMEOUT = 10
# seed for random
random.seed(0)

parameters = {
    "nightly": {
        "embedding_specs": [
            {"weight_shape": [256, 128], "indices_shape": [1, 32]},
        ],
    }
}


def run(
    embedding_specs,
    *,
    device,
):
    device.enable_async(False)

    # Extract the weight and indices shape from embedding_specs
    weight_shape = embedding_specs["weight_shape"]
    indices_shape = embedding_specs["indices_shape"]
    padding_idx = embedding_specs.get("padding_idx", None)  # Optional padding index

    # Create random weight and indices tensors in PyTorch
    weight = torch_random(weight_shape, -0.1, 0.1, dtype=torch.bfloat16)
    indices = torch.randint(0, weight_shape[0], indices_shape, dtype=torch.int32)

    # Create a PyTorch embedding layer and apply it
    torch_embedding = torch.nn.Embedding.from_pretrained(weight, padding_idx=padding_idx)
    torch_output_tensor = torch_embedding(indices)

    # Convert the weight and indices to ttnn tensor format
    ttnn_weight = ttnn.from_torch(weight, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
    ttnn_indices = ttnn.from_torch(indices, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32)

    # Measure performance of the embedding operation in ttnn
    start_time = start_measuring_time()

    # Apply embedding in ttnn
    ttnn_output_tensor = ttnn.embedding(
        ttnn_indices,
        ttnn_weight,
        padding_idx=padding_idx,
        layout=ttnn.TILE_LAYOUT,
        embeddings_type=ttnn.EmbeddingsType.GENERIC,  # Default embeddings type
        dtype=ttnn.bfloat16,
        output_tensor=None,  # No preallocated output tensor
        memory_config=None,  # Default memory config
        queue_id=0,  # Default queue id
    )

    e2e_perf = stop_measuring_time(start_time)

    # Convert the ttnn tensor back to PyTorch for comparison
    ttnn_output_tensor = ttnn.to_torch(ttnn_output_tensor)

    # Compare the results and return performance and accuracy check
    result = check_with_pcc(torch_output_tensor, ttnn_output_tensor, 0.999)

    return [result, e2e_perf]
