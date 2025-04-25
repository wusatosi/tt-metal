# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn

from models.demos.qwen25_vl.tt.generator import QwenVLGenerator
from models.tt_transformers.tt.generator_vllm import initialize_vllm_text_transformer, allocate_vllm_kv_cache
from models.tt_transformers.tt.model_config import DecodersPrecision

from vllm.model_executor.models.interfaces import SupportsMultiModal


class Qwen2_5_VLForConditionalGeneration(QwenVLGenerator, SupportsMultiModal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO
        pass

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, n_layers=None, tt_data_parallel=1):
        tt_model, model_args = initialize_vllm_text_transformer(
            hf_config,
            tt_data_parallel,
            mesh_device,
            max_batch_size,
            max_seq_len=131072,
            n_layers=n_layers,
            dtype=ttnn.bfloat8_b,
            optimizations=DecodersPrecision.performance,
        )

        return cls()  # TODO

    @property
    def cache_path(self):
        return self.model_args[0].model_cache_path

    def allocate_kv_cache(self, *args, **kwargs):
        return allocate_vllm_kv_cache(
            *args, **kwargs, dp_model=self.model, tt_cache_path=self.cache_path
        )  # TODO: replace self.model with text generator
