# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import Callable, List, Optional
import torch
import torch.nn.functional as F
from torch import nn
import ttnn

from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_utils import (
    get_inputs,
    get_weights,
    get_mask_tensor,
)


class ttnn_Attention:
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        kv_heads: Optional[int] = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        qk_norm: Optional[str] = None,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block: bool = False,
        processor: Optional["AttnProcessor"] = None,
        out_dim: int = None,
        out_context_dim: int = None,
        context_pre_only=None,
        pre_only=False,
        elementwise_affine: bool = True,
        is_causal: bool = False,
        parameters=None,
        device=None,
    ):
        # To prevent circular import.
        from diffusers.models.normalization import RMSNorm

        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.is_cross_attention = cross_attention_dim is not None
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.dropout = dropout
        self.fused_projections = False
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.out_context_dim = out_context_dim if out_context_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.is_causal = is_causal
        self.eps = eps

        self.device = device  # addedbyme
        self.parameters = parameters  # addedbyme

        self._from_deprecated_attn_block = _from_deprecated_attn_block

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0

        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.sliceable_head_dim = heads

        self.added_kv_proj_dim = added_kv_proj_dim
        self.only_cross_attention = only_cross_attention

        if self.added_kv_proj_dim is None and self.only_cross_attention:
            raise ValueError(
                "`only_cross_attention` can only be set to True if `added_kv_proj_dim` is not None. Make sure to set either `only_cross_attention=False` or define `added_kv_proj_dim`."
            )

        if norm_num_groups is not None:
            self.group_norm = ttnn.group_norm
            self.norm_num_groups = norm_num_groups
        else:
            self.group_norm = None

        if spatial_norm_dim is not None:
            self.spatial_norm = SpatialNorm(f_channels=query_dim, zq_channels=spatial_norm_dim)
        else:
            self.spatial_norm = None

        if qk_norm is None:
            self.norm_q = None
            self.norm_k = None
        elif qk_norm == "layer_norm":
            self.norm_q = nn.LayerNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
            self.norm_k = nn.LayerNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        elif qk_norm == "fp32_layer_norm":
            self.norm_q = FP32LayerNorm(dim_head, elementwise_affine=False, bias=False, eps=eps)
            self.norm_k = FP32LayerNorm(dim_head, elementwise_affine=False, bias=False, eps=eps)
        elif qk_norm == "layer_norm_across_heads":
            # Lumina applies qk norm across all heads
            self.norm_q = nn.LayerNorm(dim_head * heads, eps=eps)
            self.norm_k = nn.LayerNorm(dim_head * kv_heads, eps=eps)
        elif qk_norm == "rms_norm":
            self.norm_q = RMSNorm(dim_head, eps=eps)
            self.norm_k = RMSNorm(dim_head, eps=eps)
        elif qk_norm == "rms_norm_across_heads":
            # LTX applies qk norm across all heads
            self.norm_q = RMSNorm(dim_head * heads, eps=eps)
            self.norm_k = RMSNorm(dim_head * kv_heads, eps=eps)
        elif qk_norm == "l2":
            self.norm_q = LpNorm(p=2, dim=-1, eps=eps)
            self.norm_k = LpNorm(p=2, dim=-1, eps=eps)
        else:
            raise ValueError(f"unknown qk_norm: {qk_norm}. Should be None,'layer_norm','fp32_layer_norm','rms_norm'")

        if cross_attention_norm is None:
            self.norm_cross = None
        elif cross_attention_norm == "layer_norm":
            self.norm_cross = nn.LayerNorm(self.cross_attention_dim)
        elif cross_attention_norm == "group_norm":
            if self.added_kv_proj_dim is not None:
                # The given `encoder_hidden_states` are initially of shape
                # (batch_size, seq_len, added_kv_proj_dim) before being projected
                # to (batch_size, seq_len, cross_attention_dim). The norm is applied
                # before the projection, so we need to use `added_kv_proj_dim` as
                # the number of channels for the group norm.
                norm_cross_num_channels = added_kv_proj_dim
            else:
                norm_cross_num_channels = self.cross_attention_dim

            self.norm_cross = nn.GroupNorm(
                num_channels=norm_cross_num_channels, num_groups=cross_attention_norm_num_groups, eps=1e-5, affine=True
            )
        else:
            raise ValueError(
                f"unknown cross_attention_norm: {cross_attention_norm}. Should be None, 'layer_norm' or 'group_norm'"
            )

        self.to_q = ttnn.linear

        if not self.only_cross_attention:
            # only relevant for the `AddedKVProcessor` classes
            self.to_k = ttnn.linear
            self.to_v = ttnn.linear
        else:
            self.to_k = None
            self.to_v = None

        self.added_proj_bias = added_proj_bias
        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_kv_dim, bias=added_proj_bias)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_kv_dim, bias=added_proj_bias)
            if self.context_pre_only is not None:
                self.add_q_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
        else:
            self.add_q_proj = None
            self.add_k_proj = None
            self.add_v_proj = None

        if not self.pre_only:
            self.to_out = []
            self.to_out.append(ttnn.linear)
        else:
            self.to_out = None

        if self.context_pre_only is not None and not self.context_pre_only:
            self.to_add_out = nn.Linear(self.inner_dim, self.out_context_dim, bias=out_bias)
        else:
            self.to_add_out = None

        if processor is None:
            processor = (
                ttnn_AttnProcessor2_0()
                if hasattr(F, "scaled_dot_product_attention") and self.scale_qk
                else AttnProcessor()
            )
        self.set_processor(processor)

    def set_processor(self, processor: "AttnProcessor") -> None:
        if (
            hasattr(self, "processor")
            and isinstance(self.processor, torch.nn.Module)
            and not isinstance(processor, torch.nn.Module)
        ):
            logger.info(f"You are removing possibly trained weights of {self.processor} with {processor}")
            self._modules.pop("processor")

        self.processor = processor

    def __call__(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        quiet_attn_parameters = {"ip_adapter_masks", "ip_hidden_states"}
        unused_kwargs = [
            k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters and k not in quiet_attn_parameters
        ]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"cross_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}

        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )


class ttnn_AttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states

        input_ndim = len(hidden_states.shape)

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = ttnn.reshape(hidden_states, (batch_size, channel, height * width))
            hidden_states = ttnn.permute(hidden_states, (0, 2, 1))

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states_group_norm = ttnn.permute(hidden_states, (0, 2, 1))
            hidden_states_group_norm = ttnn.unsqueeze(hidden_states_group_norm, 2)
            inner_dim = hidden_states_group_norm.shape[1]

            N, C, H, W = hidden_states_group_norm.shape

            grid_size = ttnn.CoreGrid(y=4, x=8)
            input_mask_tensor = get_mask_tensor(C, attn.norm_num_groups, grid_size.y, attn.device)

            hidden_states_group_norm = ttnn.permute(
                hidden_states_group_norm, (0, 2, 3, 1), memory_config=ttnn.L1_MEMORY_CONFIG
            )
            hidden_states_group_norm = ttnn.reshape(hidden_states_group_norm, (N, 1, W * H, C))
            hidden_states_group_norm = ttnn.to_layout(hidden_states_group_norm, layout=ttnn.ROW_MAJOR_LAYOUT)
            grid_size = ttnn.CoreGrid(y=4, x=8)
            input_mask_tensor = get_mask_tensor(C, attn.norm_num_groups, grid_size.y, attn.device)

            grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
            shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
            shard_shape = N * H * W // grid_size.x, C // grid_size.y
            shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR, False)
            sharded_mem_config = ttnn.MemoryConfig(
                ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
            )
            hidden_states_group_norm = ttnn.to_memory_config(hidden_states_group_norm, sharded_mem_config)

            gamma_t, beta_t = get_weights(
                attn.parameters.group_norm.weight, attn.parameters.group_norm.bias, C, grid_size.y, attn.device
            )

            hidden_states = ttnn.group_norm(
                input_tensor=hidden_states_group_norm,
                num_groups=attn.norm_num_groups,
                input_mask=input_mask_tensor,
                epsilon=attn.eps,
                weight=gamma_t,
                bias=beta_t,
                memory_config=sharded_mem_config,
                core_grid=grid_size,
            )
            ttnn.deallocate(input_mask_tensor)
            hidden_states = ttnn.sharded_to_interleaved(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
            hidden_states = ttnn.permute(hidden_states, (0, 2, 3, 1))
            hidden_states = ttnn.reshape(hidden_states, (N, H * W, inner_dim))

        mm_a_y = 8
        mm_a_x = 8
        mm_a_x_strategy = ttnn.ShardStrategy.BLOCK
        mm_a_x_memory_config = ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
        hidden_states = ttnn.to_memory_config(
            hidden_states,
            memory_config=ttnn.create_sharded_memory_config(
                hidden_states.shape,
                core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_a_x),
                strategy=mm_a_x_strategy,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
            dtype=ttnn.bfloat16,
        )

        query = attn.to_q(
            hidden_states,
            attn.parameters["to_q"]["weight"],
            bias=attn.parameters["to_q"]["bias"],
            memory_config=mm_a_x_memory_config,
            core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_a_x),
        )

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(
            hidden_states,
            attn.parameters["to_k"]["weight"],
            bias=attn.parameters["to_k"]["bias"],
            memory_config=mm_a_x_memory_config,
            core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_a_x),
        )

        value = attn.to_v(
            hidden_states,
            attn.parameters["to_v"]["weight"],
            bias=attn.parameters["to_v"]["bias"],
            memory_config=mm_a_x_memory_config,
            core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_a_x),
        )

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        query = ttnn.to_memory_config(query, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        query = ttnn.permute(ttnn.reshape(query, (batch_size, -1, attn.heads, head_dim)), (0, 2, 1, 3))

        # key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = ttnn.to_memory_config(key, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        key = ttnn.permute(ttnn.reshape(key, (batch_size, -1, attn.heads, head_dim)), (0, 2, 1, 3))
        # value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = ttnn.to_memory_config(value, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        value = ttnn.permute(ttnn.reshape(value, (batch_size, -1, attn.heads, head_dim)), (0, 2, 1, 3))

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        hidden_states = ttnn.transformer.scaled_dot_product_attention(query, key, value, is_causal=False)

        hidden_states = ttnn.permute(hidden_states, (0, 2, 1, 3))
        hidden_states = ttnn.reshape(hidden_states, (batch_size, -1, attn.heads * head_dim))

        # linear proj

        mm_a_y = 8
        mm_a_x = 8
        mm_a_x_strategy = ttnn.ShardStrategy.BLOCK
        mm_a_x_memory_config = ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG

        hidden_states = ttnn.to_memory_config(
            hidden_states,
            memory_config=ttnn.create_sharded_memory_config(
                hidden_states.shape,
                core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_a_x),
                strategy=mm_a_x_strategy,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
            dtype=ttnn.bfloat8_b,
        )

        hidden_states = attn.to_out[0](
            hidden_states,
            attn.parameters["to_out"][0]["weight"],
            bias=attn.parameters["to_out"][0]["bias"],
            memory_config=mm_a_x_memory_config,
            core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_a_x),
        )

        hidden_states = ttnn.to_memory_config(hidden_states, memory_config=ttnn.L1_MEMORY_CONFIG)
        if input_ndim == 4:
            hidden_states = ttnn.permute(hidden_states, (0, 2, 1))
            hidden_states = ttnn.reshape(hidden_states, (batch_size, channel, height, width))

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = ttnn.div(hidden_states, attn.rescale_output_factor)

        return hidden_states
