# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.sentence_bert.ttnn.common import layernorm_program_config


def p(x, a="x"):
    print(f"{a}'s  shape: {x.shape}")
    print(f"{a}'s  layout: {x.layout}")
    print(f"{a}'s  dtype: {x.dtype}")
    print(f"{a}'s config: {x.memory_config()}")


class TtnnSentenceBertEmbeddings:
    def __init__(self, parameters, config):
        self.parameters = parameters
        self.config = config
        self.word_embeddings = ttnn.embedding
        self.position_embeddings = ttnn.embedding
        self.token_type_embeddings = ttnn.embedding
        self.LayerNorm = ttnn.layer_norm

    def __call__(self, input_ids: ttnn.Tensor, token_type_ids: ttnn.Tensor, position_ids: ttnn.Tensor, device):
        if input_ids.is_sharded():
            input_ids = ttnn.sharded_to_interleaved(input_ids, ttnn.L1_MEMORY_CONFIG)

        word_embeddings = self.word_embeddings(
            input_ids,
            weight=self.parameters.word_embeddings.weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            padding_idx=self.config.pad_token_id,
        )

        token_type_embeddings = self.token_type_embeddings(
            token_type_ids,
            self.parameters.token_type_embeddings.weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        position_embeddings = self.position_embeddings(
            position_ids,
            self.parameters.position_embeddings.weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        embeddings = word_embeddings + token_type_embeddings + position_embeddings
        ttnn.deallocate(word_embeddings)
        ttnn.deallocate(token_type_embeddings)
        ttnn.deallocate(position_embeddings)
        p(embeddings, "after add")
        embeddings = ttnn.unsqueeze(embeddings, dim=1)
        p(embeddings, "after unsqueeze")
        embeddings = ttnn.to_memory_config(
            embeddings,
            memory_config=ttnn.create_sharded_memory_config(
                embeddings.shape,
                core_grid=ttnn.CoreGrid(y=8, x=6),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
            dtype=ttnn.bfloat8_b,
        )
        p(embeddings, "after SHARDING")
        embeddings = self.LayerNorm(
            embeddings,
            weight=self.parameters.LayerNorm.weight,
            bias=self.parameters.LayerNorm.bias,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            epsilon=self.config.layer_norm_eps,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
            program_config=layernorm_program_config,
        )
        p(embeddings, "embed out")
        return embeddings
