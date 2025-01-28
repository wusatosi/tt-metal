# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_clip_text_transformer import ttnn_CLIPTextTransformer
from typing import Optional, Union, Tuple
from dataclasses import dataclass


@dataclass
class ttnn_CLIPTextModelOutput:
    text_embeds: Optional[ttnn.Tensor] = None
    last_hidden_state: ttnn.Tensor = None
    hidden_states: Optional[Tuple[ttnn.Tensor, ...]] = None
    attentions: Optional[Tuple[ttnn.Tensor, ...]] = None


class ttnn_CLIPTextModelWithProjection:
    def __init__(self, config, parameters):
        self.text_model = ttnn_CLIPTextTransformer(config, parameters=parameters.text_model)

        self.text_projection = ttnn.linear
        self.config = config

    def get_input_embeddings(self):
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    def __call__(
        self,
        input_ids: Optional[ttnn.Tensor] = None,
        attention_mask: Optional[ttnn.Tensor] = None,
        position_ids: Optional[ttnn.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        parameters=None,
    ) -> Union[Tuple, ttnn_CLIPTextModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            parameters=parameters.text_model,
        )

        pooled_output = text_outputs[1]

        text_embeds = self.text_projection(
            pooled_output, parameters["text_projection"]["weight"], memory_config=ttnn.L1_MEMORY_CONFIG
        )

        if not return_dict:
            outputs = (text_embeds, text_outputs[0]) + text_outputs[2:]
            return tuple(output for output in outputs if output is not None)

        return ttnn_CLIPTextModelOutput(
            text_embeds=text_embeds,
            last_hidden_state=text_outputs.last_hidden_state,
            hidden_states=text_outputs.hidden_states,
            attentions=text_outputs.attentions,
        )
