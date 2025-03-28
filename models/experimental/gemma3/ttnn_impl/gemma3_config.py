from models.tt_transformers.tt.model_config import (
    ModelArgs,
    ModelOptimizations,
    standardize_hf_keys,
    convert_hf_to_meta,
)


# Based on Gemma3TextConfig(PretrainedConfig)
class Gemma3TextTtModelArgs(ModelArgs):
    def __init__(
        self,
        model_name: str,
        mesh_device,
        instruct=False,
        dummy_weights=False,
        max_batch_size=1,
        max_seq_len=1024 * 128,
        optimizations=ModelOptimizations.accuracy,
        vocab_size=262_208,
        hidden_size=2304,
        intermediate_size=9216,
        num_hidden_layers=26,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=256,
        max_position_embeddings=131_072,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=True,
        rope_theta=1_000_000.0,
        attention_bias=False,
        attention_dropout=0.0,
        query_pre_attn_scalar=256,
        sliding_window=4096,
        final_logit_softcapping=None,
        attn_logit_softcapping=None,
        rope_scaling=None,
        rope_local_base_freq=10_000.0,
        sliding_window_pattern=6,
        **kwargs,
    ):
        super().__init__(mesh_device, instruct, dummy_weights, max_batch_size, max_seq_len, optimizations)
        self.model_name = model_name
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.sliding_window = sliding_window
        self.final_logit_softcapping = final_logit_softcapping
        self.attn_logit_softcapping = attn_logit_softcapping

        # RoPE
        self.rope_scaling = rope_scaling
        self.rope_local_base_freq = rope_local_base_freq
        self.sliding_window_pattern = sliding_window_pattern

    def _set_params_from_dict(self, params):
        super()._set_params_from_dict(params["text_config"])

    def load_state_dict(self):
        from transformers import Gemma3ForConditionalGeneration

        model = Gemma3ForConditionalGeneration.from_pretrained(self.CKPT_DIR)
        state_dict = model.language_model.state_dict()
        state_dict = standardize_hf_keys(state_dict)
        state_dict = convert_hf_to_meta(state_dict, self.head_dim)
        keys_dict = list(state_dict.keys())[:]
        remv = [f"layers.{i}." for i in list(range(self.n_layers, self.full_model_n_layers))]
        for k in keys_dict:
            if any([r in k for r in remv]):
                state_dict.pop(k)

        return state_dict


class Gemma3TtModelArgs:
    def __init__(self, text_config: Gemma3TextTtModelArgs, vision_config: dict = None):
        self.text_config = text_config
        self.vision_config = vision_config


# _VISION_CONFIG = {
#     "hidden_size": 1152,
#     "intermediate_size": 4304,
#     "num_hidden_layers": 27,
#     "num_attention_heads": 16,
#     "num_channels": 3,
#     "image_size": 896,
#     "patch_size": 14,
#     "hidden_act": "gelu_pytorch_tanh",
#     "layer_norm_eps": 1e-6,
#     "attention_dropout": 0.0,
#     "vision_use_head": False,
# }

# _VARIANT_GEMMA_3_1B = "gemma3_1b"
# _VARIANT_GEMMA_3_4B = "gemma3_4b"
# _VARIANT_GEMMA_3_12B = "gemma3_12b"
# _VARIANT_GEMMA_3_27B = "gemma3_27b"
# _VARIANTS = {
#     _VARIANT_GEMMA_3_1B: Gemma3TtModelArgs(
#         text_config=Gemma3TextTtModelArgs(
#             model_name="gemma-3-1b-it",
#             vocab_size=262_144,
#             hidden_size=1152,
#             intermediate_size=6 * 1152,
#             num_attention_heads=4,
#             num_hidden_layers=26,
#             num_key_value_heads=1,
#             head_dim=256,
#             sliding_window=512,
#             rope_theta=1_000_000,  # used for global RoPE only
#             rope_local_base_freq=10_000,
#             attn_logit_softcapping=None,
#             query_pre_attn_scalar=256,
#             max_position_embeddings=32_768,
#         ),
#         vision_config=None,
#     ),
#     _VARIANT_GEMMA_3_4B: Gemma3TtModelArgs(
#         text_config=Gemma3TextTtModelArgs(
#             model_name="gemma-3-4b-it",
#             vocab_size=262_208,
#             hidden_size=2560,
#             intermediate_size=2560 * 8 // 2,
#             num_attention_heads=8,
#             head_dim=256,
#             num_hidden_layers=34,
#             num_key_value_heads=4,
#             sliding_window=1024,
#             rope_scaling={"rope_type": "linear", "factor": 8.0},  # used for global RoPE only
#             rope_theta=1_000_000,
#             rope_local_base_freq=10_000,
#             attn_logit_softcapping=None,
#             query_pre_attn_scalar=256,
#         ),
#         vision_config=_VISION_CONFIG,
#     ),
#     _VARIANT_GEMMA_3_12B: Gemma3TtModelArgs(
#         text_config=Gemma3TextTtModelArgs(
#             model_name="gemma-3-12b-it",
#             vocab_size=262_208,
#             hidden_size=30 * 128,
#             intermediate_size=30 * 128 * 8 // 2,
#             num_attention_heads=16,
#             head_dim=256,
#             num_hidden_layers=48,
#             num_key_value_heads=8,
#             sliding_window=1024,
#             rope_scaling={"rope_type": "linear", "factor": 8.0},  # used for global RoPE only
#             rope_theta=1_000_000,
#             rope_local_base_freq=10_000,
#             attn_logit_softcapping=None,
#             query_pre_attn_scalar=256,
#         ),
#         vision_config=_VISION_CONFIG,
#     ),
#     _VARIANT_GEMMA_3_27B: Gemma3TtModelArgs(
#         text_config=Gemma3TextTtModelArgs(
#             model_name="gemma-3-27b-it",
#             vocab_size=262_208,
#             hidden_size=42 * 128,
#             intermediate_size=42 * 128 * 8 // 2,
#             num_attention_heads=32,
#             num_hidden_layers=62,
#             num_key_value_heads=16,
#             head_dim=128,
#             sliding_window=1024,
#             rope_scaling={"rope_type": "linear", "factor": 8.0},  # used for global RoPE only
#             rope_theta=1_000_000,
#             rope_local_base_freq=10_000,
#             attn_logit_softcapping=None,
#             query_pre_attn_scalar=(42 * 128 // 32),  # 1 / sqrt(hidden_size // num_attention_heads)
#         ),
#         vision_config=_VISION_CONFIG,
#     ),
# }
