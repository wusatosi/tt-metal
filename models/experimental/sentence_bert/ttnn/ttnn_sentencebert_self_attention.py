# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.sentence_bert.ttnn.common import query_key_value_matmul_program_config


def p(x, a="x"):
    print(f"{a}'s  shape: {x.shape}")
    print(f"{a}'s  layout: {x.layout}")
    print(f"{a}'s  dtype: {x.dtype}")
    print(f"{a}'s config: {x.memory_config()}")


class TtnnSentenceBertSelfAttention:
    def __init__(self, parameters, config):
        self.parameters = parameters
        self.config = config
        self.query = ttnn.linear
        self.key = ttnn.linear
        self.value = ttnn.linear
        self.num_attention_heads = config.num_attention_heads

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: ttnn.Tensor,
        device=None,
        is_optimised=False,
        is_optimised_sharded=False,
    ):
        p(hidden_states, "input")
        is_optimised_sharded = True
        num_heads = self.config.num_attention_heads
        print("numhewbfeurfb", num_heads)
        *_, hidden_size = hidden_states.shape
        head_size = hidden_size // num_heads
        if is_optimised:  # optimised version of attention
            query_key_value = ttnn.linear(
                hidden_states,
                self.parameters.query_key_value.weight,
                bias=self.parameters.query_key_value.bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
                core_grid=ttnn.CoreGrid(y=8, x=8),
            )
            p(query_key_value, "qkv out")
            (
                query,
                key,
                value,
            ) = ttnn.transformer.split_query_key_value_and_split_heads(
                query_key_value,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                num_heads=num_heads,
            )

            ttnn.deallocate(query_key_value)
            attention_scores = ttnn.matmul(
                query,
                key,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
                core_grid=ttnn.CoreGrid(y=8, x=8),
            )
            ttnn.deallocate(query)
            ttnn.deallocate(key)

            # to work  pcc drop for this op
            # attention_scores = ttnn.to_torch(attention_scores)
            # attention_mask = ttnn.to_torch(attention_mask)
            # attention_scores = attention_scores / math.sqrt(64)
            # if attention_mask is not None:
            #     attention_scores = attention_scores + attention_mask

            # attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
            # attention_probabilities = ttnn.from_torch(
            #     attention_probs,
            #     dtype=ttnn.bfloat8_b,
            #     layout=ttnn.TILE_LAYOUT,
            #     device=device,
            #     memory_config=ttnn.L1_MEMORY_CONFIG,
            # )
            print("inputs are", attention_scores.shape, attention_mask.shape, head_size)
            attention_probabilities = ttnn.transformer.attention_softmax_(
                attention_scores,
                attention_mask=attention_mask,
                head_size=head_size,
            )
            print("output is", attention_probabilities.shape)
            context_layer = ttnn.matmul(
                attention_probabilities,
                value,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,  # pcc drops for fp8
                core_grid=ttnn.CoreGrid(y=8, x=8),
            )
            ttnn.deallocate(attention_probabilities)
            ttnn.deallocate(value)

            context_layer = ttnn.transformer.concatenate_heads(
                context_layer,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            p(context_layer, "self attn opt out")
            return context_layer
        elif is_optimised_sharded:
            query_key_value = ttnn.linear(
                hidden_states,
                self.parameters.query_key_value.weight,
                bias=self.parameters.query_key_value.bias,
                memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
                program_config=query_key_value_matmul_program_config,
            )
            p(query_key_value, "qkv")
            (
                query,
                key,
                value,
            ) = ttnn.transformer.split_query_key_value_and_split_heads(
                query_key_value,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                num_heads=num_heads,
            )
            # query = torch.load("/home/ubuntu/venkatesh_latest/tt-metal/models/experimental/sentence_bert/reference/query")
            # query = ttnn.from_torch(query,dtype=ttnn.bfloat8_b,layout=ttnn.TILE_LAYOUT,device=device,memory_config=ttnn.L1_MEMORY_CONFIG)
            # key = torch.load("/home/ubuntu/venkatesh_latest/tt-metal/models/experimental/sentence_bert/reference/key")
            # key = key.transpose(-1, -2)
            # key = ttnn.from_torch(key,dtype=ttnn.bfloat8_b,layout=ttnn.TILE_LAYOUT,device=device,memory_config=ttnn.L1_MEMORY_CONFIG)
            # value = torch.load("/home/ubuntu/venkatesh_latest/tt-metal/models/experimental/sentence_bert/reference/value")
            # value = ttnn.from_torch(value,dtype=ttnn.bfloat8_b,layout=ttnn.TILE_LAYOUT,device=device,memory_config=ttnn.L1_MEMORY_CONFIG)
            # query = ttnn.to_memory_config(
            # query,
            # memory_config=ttnn.create_sharded_memory_config_(
            #     (768,64),
            #     core_grid=ttnn.CoreGrid(y=6, x=8),
            #     strategy=ttnn.ShardStrategy.BLOCK,
            #     orientation=ttnn.ShardOrientation.COL_MAJOR,
            #     tile_layout=True,
            # ),
            # )
            # key = ttnn.to_memory_config(key,
            #     memory_config=ttnn.create_sharded_memory_config(
            #     key.shape,
            #     core_grid=ttnn.CoreGrid(y=6, x=8),
            #     strategy=ttnn.ShardStrategy.BLOCK,
            #     orientation=ttnn.ShardOrientation.COL_MAJOR,
            #     )
            # )
            # value = ttnn.to_memory_config(value,
            # memory_config=ttnn.create_sharded_memory_config(
            #     value.shape,
            #     core_grid=ttnn.CoreGrid(y=6, x=8),
            #     strategy=ttnn.ShardStrategy.BLOCK,
            #     orientation=ttnn.ShardOrientation.COL_MAJOR,
            # )
            # )
            p(query, "q")
            p(key, "k")
            p(value, "v")
            # ss
            # torch.save(ttnn.to_torch(query),"/home/ubuntu/venkatesh_latest/tt-metal/models/experimental/sentence_bert/reference/output0")
            # ttnn.deallocate(query_key_value)

            attention_scores = ttnn.matmul(
                query,
                key,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
            )
            ttnn.deallocate(query)
            ttnn.deallocate(key)

            attention_probabilities = ttnn.transformer.attention_softmax_(
                attention_scores,
                attention_mask=attention_mask,
                head_size=head_size,
            )

            context_layer = ttnn.matmul(
                attention_probabilities,
                value,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
            )
            ttnn.deallocate(attention_probabilities)
            ttnn.deallocate(value)

            context_layer = ttnn.transformer.concatenate_heads(
                context_layer,
                memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            )

            return context_layer
