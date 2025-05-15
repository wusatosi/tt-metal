# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import ttnn

from .utils import from_torch_fast, to_torch


@dataclass
class TtPatchEmbeddingConv2dParameters:
    weight: ttnn.Tensor
    bias: ttnn.Tensor | None
    out_channels: int
    kernel_size: tuple[int, int]

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        hidden_dim_padding: int,
        out_channels: int,
        device: ttnn.MeshDevice,
        use_cpu_fallback: bool = False,
    ) -> TtPatchEmbeddingConv2dParameters | dict[str, torch.Tensor]:
        if use_cpu_fallback:
            return state

        weight = state["weight"]
        out_channels, in_c, kh, kw = weight.shape
        weight = torch.permute(weight, (2, 3, 1, 0))
        weight = torch.reshape(weight, (kh * kw * in_c, out_channels))

        if "bias" in state:
            bias = state["bias"].unsqueeze(0)
        else:
            bias = None

        if os.environ["FAKE_DEVICE"] == "T3K":
            weight = torch.nn.functional.pad(weight, pad=(0, hidden_dim_padding), mode="constant", value=0)
            if not bias == None:
                bias = torch.nn.functional.pad(bias, pad=(0, hidden_dim_padding), mode="constant", value=0)

        return cls(
            weight=from_torch_fast(
                weight,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                shard_dim=-1,
            ),
            bias=(
                from_torch_fast(
                    bias.reshape((1, 1, 1, -1)),
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    shard_dim=-1,
                )
                if "bias" in state
                else None
            ),
            out_channels=out_channels,
            kernel_size=(kh, kw),
        )


class TtPatchEmbeddingConv2d:
    def __init__(
        self, parameters: TtPatchEmbeddingConv2dParameters | dict[str, torch.Tensor], device: ttnn.MeshDevice
    ) -> None:
        self._device = device

        if isinstance(parameters, dict):
            with torch.device("meta"):
                self._cpu_fallback = torch.nn.Conv2d(
                    in_channels=parameters["weight"].shape[1],
                    out_channels=parameters["weight"].shape[0],
                    kernel_size=2,
                    stride=2,
                )
            self._cpu_fallback.load_state_dict(parameters, assign=True)
            self._cpu_fallback.eval()
            self._cpu_fallback.to(torch.float32)
        else:
            self._cpu_fallback = None

            self._weight = parameters.weight
            self._bias = parameters.bias
            self._unfold = torch.nn.Unfold(kernel_size=parameters.kernel_size, stride=parameters.kernel_size)

            self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if self._cpu_fallback is not None:
            torch_x = to_torch(x).to(torch.float32).permute(0, 3, 1, 2)
            torch_out = self._cpu_fallback.forward(torch_x).permute([0, 2, 3, 1])
            return from_torch_fast(
                torch_out,
                device=self._device,
                dtype=x.dtype,
                layout=x.layout,
                shard_dim=-1,
            )

        batch_size, img_h, img_w, img_c = x.shape  # permuted input NHWC
        patch_size = 2
        stride_h = patch_size
        stride_w = 1
        patches_h = img_h // patch_size
        patches_w = img_w // patch_size

        x = ttnn.reshape(x, (batch_size, patches_h, patch_size, patches_w, patch_size, img_c))
        x = ttnn.permute(x, (0, 1, 3, 2, 4, 5))
        x = ttnn.reshape(x, (batch_size, patches_h * patches_w, patch_size * patch_size * img_c))
        out = ttnn.linear(
            x,
            self._weight,
            bias=self._bias,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=ttnn.CoreGrid(y=8, x=8),
        )
        out = ttnn.reshape(out, (batch_size, patches_h, patches_w, -1))
        return out
