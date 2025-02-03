# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from transformers.models.t5 import T5PreTrainedModel
from typing import Union, Tuple, Optional
from transformers.models.t5.modeling_t5 import BaseModelOutput
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_t5_stack import (
    ttnn_T5Stack,
)
import ttnn
import copy


class ttnn_T5EncoderModel:
    def __init__(self, config, parameters=None):
        self.config = config  # added
        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = ttnn_T5Stack(encoder_config, parameters["encoder"])

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        parameters=None,
    ) -> Union[Tuple[ttnn.Tensor], BaseModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            parameters=parameters["encoder"],
        )

        return encoder_outputs
