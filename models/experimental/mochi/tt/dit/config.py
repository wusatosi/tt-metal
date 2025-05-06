from dataclasses import dataclass


@dataclass
class MochiConfig:
    """Configuration class for Mochi model and its components.

    Contains all the important parameters required to instantiate Mochi or its modules.
    """

    # Model architecture parameters
    depth: int = 48
    patch_size: int = 2
    num_heads: int = 24
    hidden_size_x: int = 3072
    hidden_size_y: int = 1536
    mlp_ratio_x: float = 4.0
    mlp_ratio_y: float = 4.0
    in_channels: int = 12
    out_channels: int = 12

    # Attention related parameters
    qk_norm: bool = True
    qkv_bias: bool = False
    out_bias: bool = True
    attention_mode: str = "sdpa"
    use_extended_posenc: bool = False
    rope_theta: float = 10000.0

    # Embedding parameters
    patch_embed_bias: bool = True
    timestep_mlp_bias: bool = True
    timestep_scale: float = 1000.0

    # Text encoder parameters
    t5_feat_dim: int = 4096
    t5_token_length: int = 256
