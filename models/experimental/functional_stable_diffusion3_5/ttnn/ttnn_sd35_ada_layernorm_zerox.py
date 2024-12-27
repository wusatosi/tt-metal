# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from typing import Optional, Tuple
import ttnn


class ttnn_SD35AdaLayerNormZeroX:
    r"""
    Norm layer adaptive layer norm zero (AdaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, norm_type: str = "layer_norm", bias: bool = True) -> None:
        self.silu = ttnn.silu
        self.linear = ttnn.linear
        if norm_type == "layer_norm":
            self.norm = ttnn.layer_norm
        else:
            raise ValueError(f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm'.")

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        emb: Optional[ttnn.Tensor] = None,
        parameters=None,
    ) -> Tuple[ttnn.Tensor, ...]:
        hifi2_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
        )
        emb = self.linear(
            self.silu(emb),
            parameters["linear"]["weight"],
            bias=parameters["linear"]["bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=hifi2_kernel_config,
        )
        emb = ttnn.to_torch(emb)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_msa2, scale_msa2, gate_msa2 = emb.chunk(
            9, dim=1
        )
        shift_msa = ttnn.from_torch(
            shift_msa, layout=ttnn.TILE_LAYOUT, device=hidden_states.device(), memory_config=ttnn.L1_MEMORY_CONFIG
        )
        scale_msa = ttnn.from_torch(
            scale_msa, layout=ttnn.TILE_LAYOUT, device=hidden_states.device(), memory_config=ttnn.L1_MEMORY_CONFIG
        )
        gate_msa = ttnn.from_torch(
            gate_msa, layout=ttnn.TILE_LAYOUT, device=hidden_states.device(), memory_config=ttnn.L1_MEMORY_CONFIG
        )
        shift_mlp = ttnn.from_torch(
            shift_mlp, layout=ttnn.TILE_LAYOUT, device=hidden_states.device(), memory_config=ttnn.L1_MEMORY_CONFIG
        )
        scale_mlp = ttnn.from_torch(
            scale_mlp, layout=ttnn.TILE_LAYOUT, device=hidden_states.device(), memory_config=ttnn.L1_MEMORY_CONFIG
        )
        gate_mlp = ttnn.from_torch(
            gate_mlp, layout=ttnn.TILE_LAYOUT, device=hidden_states.device(), memory_config=ttnn.L1_MEMORY_CONFIG
        )
        shift_msa2 = ttnn.from_torch(
            shift_msa2, layout=ttnn.TILE_LAYOUT, device=hidden_states.device(), memory_config=ttnn.L1_MEMORY_CONFIG
        )
        scale_msa2 = ttnn.from_torch(
            scale_msa2, layout=ttnn.TILE_LAYOUT, device=hidden_states.device(), memory_config=ttnn.L1_MEMORY_CONFIG
        )
        gate_msa2 = ttnn.from_torch(
            gate_msa2, layout=ttnn.TILE_LAYOUT, device=hidden_states.device(), memory_config=ttnn.L1_MEMORY_CONFIG
        )

        norm_hidden_states = self.norm(hidden_states, compute_kernel_config=hifi2_kernel_config)
        hidden_states = norm_hidden_states * (
            1 + ttnn.reshape(scale_msa, (scale_msa.shape[0], 1, scale_msa.shape[1]))
        ) + ttnn.reshape(shift_msa, (shift_msa.shape[0], 1, shift_msa.shape[1]))
        norm_hidden_states2 = norm_hidden_states * (
            1 + ttnn.reshape(scale_msa2, (scale_msa2.shape[0], 1, scale_msa2.shape[1]))
        ) + ttnn.reshape(shift_msa2, (shift_msa2.shape[0], 1, shift_msa2.shape[1]))
        return hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2
