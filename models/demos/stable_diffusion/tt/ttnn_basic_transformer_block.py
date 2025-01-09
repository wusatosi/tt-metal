# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.demos.stable_diffusion.tt.ttnn_attention import sd_attention


def sd_geglu(
    hidden_states,
    parameters,
    device=None,
):
    memory_config = ttnn.create_sharded_memory_config(
        hidden_states.shape,
        core_grid=device.core_grid,
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    print(hidden_states.shape)
    hidden_states = ttnn.linear(
        hidden_states,
        parameters.proj.weight,
        bias=parameters.proj.bias,
        memory_config=memory_config,
        core_grid=device.core_grid,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
        ),
    )
    if hidden_states.is_sharded():
        hidden_states = ttnn.sharded_to_interleaved(hidden_states, ttnn.L1_MEMORY_CONFIG)
    hidden_states = ttnn.geglu(hidden_states)

    return hidden_states


def sd_feed_forward(
    hidden_states,
    parameters,
    device,
):
    hidden_states = sd_geglu(hidden_states, parameters.net[0], device)
    print("geglu")
    memory_config = ttnn.create_sharded_memory_config(
        hidden_states.shape,
        core_grid=device.core_grid,
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    hidden_states = ttnn.to_memory_config(
        hidden_states,
        memory_config=memory_config,
    )
    hidden_states = ttnn.linear(
        hidden_states,
        parameters.net[2].weight,
        bias=parameters.net[2].bias,
        memory_config=memory_config,
        core_grid=device.core_grid,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
        ),
    )
    ss

    return hidden_states


def sd_basic_transformer_block(
    hidden_states,
    encoder_hidden_states=None,
    timestep=None,
    attention_mask=None,
    cross_attention_kwargs=None,
    class_labels=None,
    config=None,
    num_embeds_ada_norm=False,
    cross_attention_dim: int = None,
    only_cross_attention: bool = False,
    attention_head_dim=None,
    *,
    parameters,
    device,
):
    memory_config = ttnn.create_sharded_memory_config(
        hidden_states.shape,
        core_grid=device.core_grid,
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    hidden_states = ttnn.to_memory_config(
        hidden_states,
        memory_config=memory_config,
    )
    norm_hidden_states = ttnn.layer_norm(
        hidden_states,
        epsilon=1e-05,
        weight=parameters.norm1.weight,
        bias=parameters.norm1.bias,
        memory_config=memory_config,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
        ),
    )

    cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
    cross_attention_dim = config.cross_attention_dim if cross_attention_dim is None else cross_attention_dim
    attn_output = sd_attention(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=encoder_hidden_states if only_cross_attention else None,
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
        cross_attention_dim=cross_attention_dim,
        heads=attention_head_dim,
        parameters=parameters.attn1,
        device=device,
    )

    print("attn1")
    if hidden_states.is_sharded():
        hidden_states = ttnn.sharded_to_interleaved(hidden_states, ttnn.L1_MEMORY_CONFIG)
    if attn_output.is_sharded():
        attn_output = ttnn.sharded_to_interleaved(attn_output, ttnn.L1_MEMORY_CONFIG)
    hidden_states = ttnn.add(attn_output, hidden_states, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(norm_hidden_states)

    if cross_attention_dim is not None:
        norm_hidden_states = ttnn.layer_norm(
            hidden_states,
            epsilon=1e-05,
            weight=parameters.norm2.weight,
            bias=parameters.norm2.bias,
            # memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        attn_output = sd_attention(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            cross_attention_dim=cross_attention_dim,
            heads=attention_head_dim,
            parameters=parameters.attn2,
            device=device,
        )
        print("attn2")
        hidden_states = ttnn.add(attn_output, hidden_states, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(attn_output)

        memory_config = ttnn.create_sharded_memory_config(
            hidden_states.shape,
            core_grid=device.core_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        hidden_states = ttnn.to_memory_config(
            hidden_states,
            memory_config=memory_config,
        )

        norm_hidden_states = ttnn.layer_norm(
            hidden_states,
            epsilon=1e-05,
            weight=parameters.norm3.weight,
            bias=parameters.norm3.bias,
            memory_config=memory_config,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
            ),
        )
        print("ff")
        ff_output = sd_feed_forward(hidden_states=norm_hidden_states, parameters=parameters.ff, device=device)
        hidden_states = ttnn.add(ff_output, hidden_states, memory_config=ttnn.L1_MEMORY_CONFIG)
        return hidden_states
