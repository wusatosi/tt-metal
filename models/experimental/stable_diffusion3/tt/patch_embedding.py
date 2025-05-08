# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
import ttnn

from ..reference.patch_embedding import PatchEmbed
from .conv2d import TtConv2d, TtConv2dParameters
from .substate import substate
from .utils import from_torch_fast, to_torch


@dataclass
class TtPatchEmbedParameters:
    proj: TtConv2dParameters
    pos_embed: ttnn.Tensor

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        device: ttnn.MeshDevice,
        hidden_dim_padding: int,
        out_channels: int,
        use_cpu_fallback: bool = False,
    ) -> TtPatchEmbedParameters | dict[str, torch.Tensor]:
        if use_cpu_fallback:
            return state

        pos_embed_param = state["pos_embed"]
        if os.environ["FAKE_DEVICE"] == "T3K":
            pos_embed_param = torch.nn.functional.pad(
                pos_embed_param, pad=(0, hidden_dim_padding), mode="constant", value=0
            )

        return cls(
            proj=TtConv2dParameters.from_torch(
                substate(state, "proj"),
                dtype=ttnn.bfloat16,
                hidden_dim_padding=hidden_dim_padding,
                out_channels=out_channels,
                device=device,
            ),
            pos_embed=from_torch_fast(
                pos_embed_param, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device, shard_dim=-1
            ),
        )

    @property
    def pos_embed_max_size(self) -> int:
        return math.isqrt(self.pos_embed.shape[1])


class TtPatchEmbed:
    def __init__(self, parameters: TtPatchEmbedParameters, *, device: ttnn.MeshDevice) -> None:
        super().__init__()

        self._patch_size = 2

        if isinstance(parameters, dict):
            pos_embed_max_size = math.isqrt(parameters["pos_embed"].shape[1])

            with torch.device("meta"):
                self._cpu_fallback = PatchEmbed(
                    patch_size=self._patch_size,
                    in_channels=parameters["proj.weight"].shape[1],
                    embed_dim=parameters["proj.weight"].shape[0],
                    pos_embed_max_size=pos_embed_max_size,
                )
            self._cpu_fallback.load_state_dict(parameters, assign=True)
            self._cpu_fallback.eval()
            self._cpu_fallback.to(torch.float32)
        else:
            self._cpu_fallback = None

            self._pos_embed_max_size = parameters.pos_embed_max_size
            self._proj = TtConv2d(parameters.proj, device=device)
            self._pos_embed = parameters.pos_embed

        self._device = device

    def __call__(self, latent: ttnn.Tensor) -> ttnn.Tensor:
        if self._cpu_fallback is not None:
            torch_latent = to_torch(latent).permute([0, 3, 1, 2]).to(torch.float32).squeeze(1)
            torch_out = self._cpu_fallback.forward(torch_latent)
            return from_torch_fast(
                torch_out,
                device=self._device,
                dtype=latent.dtype,
                layout=latent.layout,
                shard_dim=-1,
            )

        batch_size, in_height, in_width, _c = latent.shape
        out_height = in_height // self._patch_size
        out_width = in_width // self._patch_size

        latent = self._proj(latent)
        latent = ttnn.reshape(latent, (batch_size, out_height * out_width, -1))
        pos_embed = self._cropped_pos_embed(out_height, out_width)
        return latent + pos_embed

    @property
    def patch_size(self) -> int:
        return self._patch_size

    def _cropped_pos_embed(self, height: int, width: int) -> ttnn.Tensor:
        top = (self._pos_embed_max_size - height) // 2
        left = (self._pos_embed_max_size - width) // 2

        spatial_pos_embed = self._pos_embed.reshape([1, self._pos_embed_max_size, self._pos_embed_max_size, -1])
        spatial_pos_embed = spatial_pos_embed[:, top : top + height, left : left + width, :]
        return spatial_pos_embed.reshape([1, -1, spatial_pos_embed.shape[-1]])
