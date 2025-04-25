# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List
import torch
import PIL


class QwenVLGenerator:
    def __init__(self):
        pass

    # Note: This function is called by vLLM
    def prefill_forward(
        self,
        tokens: torch.Tensor,
        images: List[PIL.Image.Image],
        page_table: torch.Tensor,
        kv_cache,
        prompt_lens,
    ):
        # TODO
        pass

    # Note: This function is called by vLLM
    def decode_forward(
        self,
        start_pos,
        tokens,
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        read_from_device=True,
    ):
        # TODO
        pass
