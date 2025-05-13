# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import ttnn
from models.experimental.stable_diffusion3.tt.linear import TtLinear, TtLinearParameters

from ..reference.transformer import SD3Transformer2DModel
from . import utils
from .normalization import TtLayerNorm, TtLayerNormParameters
from .patch_embedding import TtPatchEmbed, TtPatchEmbedParameters
from .substate import indexed_substates, substate
from .timestep_embedding import TtCombinedTimestepTextProjEmbeddings, TtCombinedTimestepTextProjEmbeddingsParameters
from .transformer_block import TtTransformerBlock, TtTransformerBlockParameters, chunk_time
from .utils import to_torch


@dataclass
class TtSD3Transformer2DModelParameters:
    pos_embed: TtPatchEmbedParameters
    time_text_embed: TtCombinedTimestepTextProjEmbeddingsParameters
    context_embedder: TtLinearParameters
    transformer_blocks: list[TtTransformerBlockParameters]
    time_embed_out: TtLinearParameters
    norm_out: TtLayerNormParameters
    proj_out: TtLinearParameters
    distributed: bool

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        num_heads: int,
        unpadded_num_heads: int,
        embedding_dim: int,
        hidden_dim_padding: int,
        dtype: ttnn.DataType | None = None,
        device: ttnn.MeshDevice,
        use_cpu_fallback: bool = False,
    ) -> TtSD3Transformer2DModelParameters | dict[str, torch.Tensor]:
        if use_cpu_fallback:
            return state

        return cls(
            pos_embed=TtPatchEmbedParameters.from_torch(
                substate(state, "pos_embed"),
                device=device,
                hidden_dim_padding=hidden_dim_padding,
                out_channels=embedding_dim,
            ),
            time_text_embed=TtCombinedTimestepTextProjEmbeddingsParameters.from_torch(
                substate(state, "time_text_embed"), dtype=dtype, device=device
            ),
            context_embedder=TtLinearParameters.from_torch(
                substate(state, "context_embedder"), dtype=dtype, device=device, shard_dim=-1
            ),
            transformer_blocks=[
                TtTransformerBlockParameters.from_torch(
                    s,
                    num_heads=num_heads,
                    unpadded_num_heads=unpadded_num_heads,
                    hidden_dim_padding=hidden_dim_padding,
                    dtype=dtype,
                    device=device,
                )
                for s in indexed_substates(state, "transformer_blocks")
            ],
            time_embed_out=TtLinearParameters.from_torch(
                substate(state, "norm_out.linear"), dtype=dtype, device=device, shard_dim=None, unsqueeze_bias=True
            ),
            norm_out=TtLayerNormParameters.from_torch(
                substate(state, "norm_out.norm"), dtype=dtype, device=device, distributed=False, weight_shape=[2432]
            ),
            proj_out=TtLinearParameters.from_torch(
                substate(state, "proj_out"), dtype=dtype, device=device, shard_dim=None
            ),
            distributed=device.get_num_devices() > 1,
        )


class ShardingProjection:
    def __init__(self, *, dim: int, device: ttnn.MeshDevice) -> None:
        params = TtLinearParameters.from_torch(
            dict(weight=torch.eye(dim)),
            dtype=ttnn.bfloat16,
            device=device,
            shard_dim=-1,
        )
        self._projection = TtLinear(params)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return self._projection(x)


class TtSD3Transformer2DModel:
    def __init__(
        self,
        parameters: TtSD3Transformer2DModelParameters | dict[str, torch.Tensor],
        *,
        guidance_cond: int = 2,
        num_heads: int,
        device: ttnn.MeshDevice,
    ) -> None:
        super().__init__()

        self._device = device

        if isinstance(parameters, dict):
            with torch.device("meta"):
                self._cpu_fallback = SD3Transformer2DModel(
                    num_layers=38,
                    num_attention_heads=num_heads,
                    caption_projection_dim=parameters["context_embedder.weight"].shape[0],
                    pos_embed_max_size=math.isqrt(parameters["pos_embed.pos_embed"].shape[1]),
                )
            self._cpu_fallback.load_state_dict(parameters, assign=True)
            self._cpu_fallback.eval()
            self._cpu_fallback.to(torch.float32)

            self._patch_size = self._cpu_fallback.patch_size
        else:
            self._cpu_fallback = None

            self._pos_embed = TtPatchEmbed(parameters.pos_embed, device=device)
            self._time_text_embed = TtCombinedTimestepTextProjEmbeddings(
                parameters.time_text_embed,
                device=device,
                batch_size=guidance_cond,
            )
            self._context_embedder = TtLinear(parameters.context_embedder)
            self._transformer_blocks = [
                TtTransformerBlock(block, num_heads=num_heads, device=device) for block in parameters.transformer_blocks
            ]
            self._time_embed_out = TtLinear(parameters.time_embed_out)
            self._norm_out = TtLayerNorm(parameters.norm_out, eps=1e-6, use_cpu_fallback=True)
            self._proj_out = TtLinear(parameters.proj_out)

            self._patch_size = self._pos_embed.patch_size

            # TODO: get dimensions from other parameters
            self._sharding = ShardingProjection(dim=2432, device=device)
            self._distributed = parameters.distributed

    def __call__(
        self,
        *,
        spatial: ttnn.Tensor,
        prompt: ttnn.Tensor,
        pooled_projection: ttnn.Tensor,
        timestep: ttnn.Tensor,
        N: int,
        L: int,
    ) -> ttnn.Tensor:
        if self._cpu_fallback is not None:
            torch_spatial = to_torch(spatial).to(torch.float32).permute([0, 3, 1, 2])
            torch_prompt = to_torch(prompt).to(torch.float32)
            torch_pooled_projection = to_torch(pooled_projection).to(torch.float32)
            torch_timestep = to_torch(timestep).to(torch.float32).squeeze(1)

            with torch.no_grad():
                torch_result = self._cpu_fallback(
                    spatial=torch_spatial,
                    prompt_embed=torch_prompt,
                    pooled_projections=torch_pooled_projection,
                    timestep=torch_timestep,
                )

            return ttnn.from_torch(
                torch_result.unsqueeze(1),
                device=self._device,
                layout=ttnn.TILE_LAYOUT,
                dtype=spatial.dtype,
            )

        spatial = self._pos_embed(spatial)
        time_embed = self._time_text_embed(timestep=timestep, pooled_projection=pooled_projection)
        prompt = self._context_embedder(prompt)
        time_embed = time_embed.reshape([time_embed.shape[0], 1, 1, time_embed.shape[1]])
        spatial = ttnn.unsqueeze(spatial, 1)
        prompt = ttnn.unsqueeze(prompt, 1)

        for i, block in enumerate(self._transformer_blocks, start=1):
            spatial, prompt_out = block(
                spatial=spatial,
                prompt=prompt,
                time_embed=time_embed,
                N=N,  # spatial_sequence_length
                L=L,  # prompt_sequence_length
            )
            if prompt_out is not None:
                prompt = prompt_out

        spatial_time = self._time_embed_out(utils.silu(time_embed))
        [scale, shift] = chunk_time(spatial_time, 2)
        if self._distributed:
            spatial = utils.all_gather(spatial, dim=-1)
        spatial = self._norm_out(spatial) * (1 + scale) + shift
        return self._proj_out(spatial)

    def cache_and_trace(
        self,
        *,
        spatial: ttnn.Tensor,
        prompt: ttnn.Tensor,
        pooled_projection: ttnn.Tensor,
        timestep: ttnn.Tensor,
    ) -> TtSD3Transformer2DModelTrace:
        device = spatial.device()

        self(spatial=spatial, prompt=prompt, pooled_projection=pooled_projection, timestep=timestep)

        tid = ttnn.begin_trace_capture(device)
        output = self(spatial=spatial, prompt=prompt, pooled_projection=pooled_projection, timestep=timestep)
        ttnn.end_trace_capture(device, tid)

        return TtSD3Transformer2DModelTrace(
            spatial_input=spatial,
            prompt_input=prompt,
            pooled_projection_input=pooled_projection,
            timestep_input=timestep,
            output=output,
            tid=tid,
        )

    @property
    def patch_size(self) -> int:
        return self._patch_size


@dataclass
class TtSD3Transformer2DModelTrace:
    spatial_input: ttnn.Tensor
    prompt_input: ttnn.Tensor
    pooled_projection_input: ttnn.Tensor
    timestep_input: ttnn.Tensor
    output: ttnn.Tensor
    tid: int

    def __call__(
        self,
        *,
        spatial: ttnn.Tensor,
        prompt: ttnn.Tensor,
        pooled_projection: ttnn.Tensor,
        timestep: ttnn.Tensor,
    ) -> ttnn.Tensor:
        device = self.spatial_input.device()

        ttnn.copy_host_to_device_tensor(spatial, self.spatial_input)
        ttnn.copy_host_to_device_tensor(prompt, self.prompt_input)
        ttnn.copy_host_to_device_tensor(pooled_projection, self.pooled_projection_input)
        ttnn.copy_host_to_device_tensor(timestep, self.timestep_input)

        ttnn.execute_trace(device, self.tid)

        return self.output
