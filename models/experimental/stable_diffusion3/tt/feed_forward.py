# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import ttnn

from ..reference.feed_forward import FeedForward
from .linear import TtLinear, TtLinearParameters
from .substate import substate
from .utils import from_torch_fast, to_torch

if TYPE_CHECKING:
    import torch


@dataclass
class TtFeedForwardParameters:
    in_proj: TtLinearParameters
    out_proj: TtLinearParameters
    distributed: bool

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.MeshDevice,
        use_cpu_fallback: bool = False,
    ) -> TtFeedForwardParameters | dict[str, torch.Tensor]:
        if use_cpu_fallback:
            return state

        return cls(
            in_proj=TtLinearParameters.from_torch(
                substate(state, "net.0.proj"), dtype=dtype, device=device, shard_dim=-1
            ),
            out_proj=TtLinearParameters.from_torch(substate(state, "net.2"), dtype=dtype, device=device, shard_dim=-2),
            distributed=device.get_num_devices() > 1,
        )


class TtFeedForward:
    def __init__(
        self,
        parameters: TtFeedForwardParameters | dict[str, torch.Tensor],
        *,
        device: ttnn.MeshDevice,
    ) -> None:
        super().__init__()

        if isinstance(parameters, dict):
            with torch.device("meta"):
                self._cpu_fallback = FeedForward(
                    parameters["net.0.proj.weight"].shape[1], parameters["net.2.weight"].shape[0]
                )
            self._cpu_fallback.load_state_dict(parameters, assign=True)
            self._cpu_fallback.eval()
            self._cpu_fallback.to(torch.float32)
        else:
            self._cpu_fallback = None

            self.in_proj = TtLinear(parameters.in_proj)
            self.out_proj = TtLinear(parameters.out_proj)
            self._distributed = parameters.distributed

        self._device = device

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if self._cpu_fallback is not None:
            torch_x = to_torch(x).to(torch.float32)
            torch_out = self._cpu_fallback.forward(torch_x)
            return from_torch_fast(
                torch_out,
                device=self._device,
                dtype=x.dtype,
                layout=x.layout,
                shard_dim=-1,
            )

        grid_size = x.device().compute_with_storage_grid_size()
        core_grid = ttnn.CoreGrid(x=grid_size.x, y=grid_size.y)

        x = self.in_proj(x, core_grid=core_grid, highest_quality=False)  # not enough L1 for highest quality
        # Turning on fast_and_approximate_mode leads to big changes in the generated image.
        # The image quality might still be okay.
        x = ttnn.gelu(x, fast_and_approximate_mode=False)

        result = self.out_proj(x, core_grid=core_grid)

        if self._distributed:
            result = ttnn.reduce_scatter(
                result,
                dim=-1,
                math_op=ttnn.ReduceType.Sum,
                num_links=1,
                memory_config=ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
                topology=ttnn.Topology.Ring,
            )

        return result
