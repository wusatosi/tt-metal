# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from typing import Optional, Tuple


class ttnn_AdaLayerNormZero:
    def __init__(self, embedding_dim: int, num_embeddings: Optional[int] = None, norm_type="layer_norm", bias=True):
        self.emb = None

        self.silu = ttnn.silu
        self.linear = ttnn.linear
        if norm_type == "layer_norm":
            self.norm = ttnn.layer_norm
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def __call__(
        self,
        x: torch.Tensor,
        timestep: Optional[ttnn.Tensor] = None,
        class_labels: Optional[ttnn.Tensor] = None,
        hidden_dtype=None,
        emb: Optional[ttnn.Tensor] = None,
        parameters=None,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        hifi2_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
        )
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
        emb = self.linear(
            self.silu(emb),
            parameters["linear"]["weight"],
            bias=parameters["linear"]["bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=hifi2_kernel_config,
        )

        emb = ttnn.to_layout(emb, ttnn.ROW_MAJOR_LAYOUT)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ttnn.split(
            emb, 6, 1, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        shift_msa = ttnn.to_layout(shift_msa, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        scale_msa = ttnn.to_layout(scale_msa, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        gate_msa = ttnn.to_layout(gate_msa, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        shift_mlp = ttnn.to_layout(shift_mlp, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        scale_mlp = ttnn.to_layout(scale_mlp, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        gate_mlp = ttnn.to_layout(gate_mlp, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

        x = self.norm(x, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=hifi2_kernel_config) * (
            1 + ttnn.reshape(scale_msa, (scale_msa.shape[0], 1, scale_msa.shape[1]))
        ) + ttnn.reshape(
            shift_msa, (shift_msa.shape[0], 1, shift_msa.shape[1])
        )  # shift_msa[:, None] replaced with ttnn.reshape(shift_msa,(shift_msa.shape[0],1,shift_msa.shape[1])) same for scale_msa[:,None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp
