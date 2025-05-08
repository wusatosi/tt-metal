# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import ttnn
from models.experimental.stable_diffusion3.tt.linear import TtLinear, TtLinearParameters

from ..reference.timestep_embedding import CombinedTimestepTextProjEmbeddings
from .substate import substate
from .utils import from_torch_fast, to_torch


@dataclass
class TtEmbeddingParameters:
    linear_1: TtLinearParameters
    linear_2: TtLinearParameters

    @classmethod
    def from_torch(
        cls,
        state: dict[str, ttnn.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtEmbeddingParameters:
        return cls(
            linear_1=TtLinearParameters.from_torch(
                substate(state, "linear_1"), dtype=dtype, device=device, shard_dim=None
            ),
            linear_2=TtLinearParameters.from_torch(
                substate(state, "linear_2"), dtype=dtype, device=device, shard_dim=None
            ),
        )


@dataclass
class TtCombinedTimestepTextProjEmbeddingsParameters:
    timestep_embedder: TtEmbeddingParameters
    text_embedder: TtEmbeddingParameters

    @classmethod
    def from_torch(
        cls,
        state: dict[str, ttnn.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
        use_cpu_fallback: bool = False,
    ) -> TtCombinedTimestepTextProjEmbeddingsParameters | dict[str, torch.Tensor]:
        if use_cpu_fallback:
            return state

        return cls(
            timestep_embedder=TtEmbeddingParameters.from_torch(
                substate(state, "timestep_embedder"), dtype=dtype, device=device
            ),
            text_embedder=TtEmbeddingParameters.from_torch(
                substate(state, "text_embedder"), dtype=dtype, device=device
            ),
        )


class TtCombinedTimestepTextProjEmbeddings:
    def __init__(
        self,
        parameters: TtCombinedTimestepTextProjEmbeddingsParameters,
        *,
        batch_size: int,
        device: ttnn.MeshDevice,
    ) -> None:
        self._device = device

        if isinstance(parameters, dict):
            with torch.device("meta"):
                self._cpu_fallback = CombinedTimestepTextProjEmbeddings(
                    embedding_dim=parameters["timestep_embedder.linear_1.weight"].shape[0],
                    pooled_projection_dim=parameters["text_embedder.linear_1.weight"].shape[1],
                )
            self._cpu_fallback.load_state_dict(parameters, assign=True)
            self._cpu_fallback.eval()
            self._cpu_fallback.to(torch.float32)
        else:
            self._cpu_fallback = None

            self._timestep_embedder = _TimestepEmbedding(parameters.timestep_embedder)
            self._text_embedder = _TimestepEmbedding(parameters.text_embedder)

            self._time_proj_factor = self._create_time_proj_factor(
                num_channels=256, batch_size=batch_size, device=device
            )

    def __call__(self, *, timestep: ttnn.Tensor, pooled_projection: ttnn.Tensor) -> ttnn.Tensor:
        assert timestep.dtype == ttnn.float32

        if self._cpu_fallback is not None:
            torch_timestep = to_torch(timestep).to(torch.float32).squeeze(1)
            torch_pooled_projection = to_torch(pooled_projection).to(torch.float32)
            torch_out = self._cpu_fallback.forward(timestep=torch_timestep, pooled_projection=torch_pooled_projection)
            return from_torch_fast(
                torch_out,
                device=self._device,
                dtype=timestep.dtype,
                layout=timestep.layout,
                # shard_dim=-1,
            )

        batch_size = timestep.shape[0]

        # time_proj_factor = ttnn.repeat(self._time_proj_factor, ttnn.Shape([batch_size, 1]))
        # time_proj_factor = ttnn.to_layout(time_proj_factor, ttnn.TILE_LAYOUT)
        time_proj_factor = ttnn.to_layout(self._time_proj_factor, ttnn.TILE_LAYOUT)

        emb = timestep * time_proj_factor

        c = ttnn.cos(emb)
        s = ttnn.sin(emb)

        timesteps_proj = ttnn.concat([c, s], dim=-1)
        timesteps_proj = ttnn.clone(timesteps_proj, dtype=pooled_projection.dtype)

        time_embed = self._timestep_embedder(timesteps_proj)
        text_embed = self._text_embedder(pooled_projection)

        return time_embed + text_embed

    @staticmethod
    def _create_time_proj_factor(*, num_channels: int, batch_size: int, device: ttnn.Device) -> ttnn.Tensor:
        assert num_channels % 2 == 0
        half_dim = num_channels // 2

        max_period = 10000

        exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32)
        exponent = exponent / half_dim
        factor = torch.exp(exponent).unsqueeze(0).repeat(batch_size, 1)

        return ttnn.from_torch(factor, device=device, mesh_mapper=ttnn.ReplicateTensorToMesh(device))


class _TimestepEmbedding:
    def __init__(self, parameters: TtEmbeddingParameters) -> None:
        super().__init__()

        self._linear_1 = TtLinear(parameters.linear_1)
        self._linear_2 = TtLinear(parameters.linear_2)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = self._linear_1(x)
        x = ttnn.silu(x)
        return self._linear_2(x)

    @property
    def device(self) -> ttnn.Device:
        return self._linear_1.device
