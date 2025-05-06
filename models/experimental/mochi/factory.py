import os
from typing import Optional, Dict, Any

from genmo.mochi_preview.pipelines import ModelFactory, load_to_cpu
from models.experimental.mochi.tt.dit.model import AsymmDiTJoint
from models.experimental.mochi.tt.dit.config import MochiConfig
from models.experimental.mochi.tt.common import get_cache_path, get_mochi_dir
from models.experimental.mochi.tt.vae.decoder import Decoder
from models.experimental.mochi.tt.vae.common import load_decoder_weights


class TtDiTModelFactory(ModelFactory):
    """Factory for creating TensorTorch DiT models."""

    def __init__(
        self,
        mesh_device,
        *,
        model_path: str,
        model_dtype: str,
        lora_path: Optional[str] = None,
        attention_mode: Optional[str] = None,
    ):
        """Initialize the TT DiT model factory.

        Args:
            model_path: Path to model weights
            model_dtype: Data type for model (e.g. "bf16")
            lora_path: Optional path to LoRA weights
            attention_mode: Optional attention implementation mode
        """
        attention_mode = "sdpa"

        super().__init__(
            model_path=model_path,
            lora_path=lora_path,
            model_dtype=model_dtype,
            attention_mode=attention_mode,
        )

        # TODO: parametrize based on inputs to get_model
        self.weight_cache_path = get_cache_path(os.environ.get("MESH_DEVICE"))
        self.weights_path = os.path.join(get_mochi_dir(), "dit.safetensors")
        self.mesh_device = mesh_device
        self.config = MochiConfig()

    def get_model(
        self,
        *,
        local_rank: int,
        device_id: Any,
        world_size: int,
        model_kwargs: Optional[Dict[str, Any]] = None,
        strict_load: bool = True,
        load_checkpoint: bool = True,
        fast_init: bool = True,
    ) -> AsymmDiTJoint:
        """Create and initialize a TT DiT model.

        Args:
            local_rank: Local process rank
            device_id: Device ID (ignored for TT implementation)
            world_size: Total number of processes
            model_kwargs: Optional additional model arguments
            strict_load: Whether to strictly enforce state dict loading
            load_checkpoint: Whether to load weights from checkpoint

        Returns:
            Initialized TT DiT model
        """
        assert load_checkpoint, "Checkpoint loading is required for TT DiT"
        if not model_kwargs:
            model_kwargs = {}

        # Load state dict if needed
        state_dict = {}

        print(f"Loading weights from {self.weights_path}")
        state_dict = load_to_cpu(self.weights_path)

        # Create model using config values
        model = AsymmDiTJoint(
            mesh_device=self.mesh_device,
            state_dict=state_dict,
            weight_cache_path=self.weight_cache_path,
            depth=self.config.depth,
            patch_size=self.config.patch_size,
            num_heads=self.config.num_heads,
            hidden_size_x=self.config.hidden_size_x,
            hidden_size_y=self.config.hidden_size_y,
            mlp_ratio_x=self.config.mlp_ratio_x,
            mlp_ratio_y=self.config.mlp_ratio_y,
            in_channels=self.config.in_channels,
            qk_norm=self.config.qk_norm,
            qkv_bias=self.config.qkv_bias,
            out_bias=self.config.out_bias,
            patch_embed_bias=self.config.patch_embed_bias,
            timestep_mlp_bias=self.config.timestep_mlp_bias,
            timestep_scale=self.config.timestep_scale,
            t5_feat_dim=self.config.t5_feat_dim,
            t5_token_length=self.config.t5_token_length,
            rope_theta=self.config.rope_theta,
            attention_mode=self.kwargs["attention_mode"],
            **model_kwargs,
        )

        return model


class TtDecoderModelFactory(ModelFactory):
    def __init__(self, mesh_device, *, model_path: str):
        super().__init__(model_path=model_path)
        self.mesh_device = mesh_device

    def get_model(self, *, local_rank=0, device_id=0, world_size=1):
        # TODO(ved): Set flag for torch.compile
        # TODO(ved): Use skip_init
        state_dict = load_decoder_weights()

        decoder = Decoder(
            mesh_device=self.mesh_device,
            state_dict=state_dict,
            state_dict_prefix="",
            out_channels=3,
            base_channels=128,
            channel_multipliers=[1, 2, 4, 6],
            temporal_expansions=[1, 2, 3],
            spatial_expansions=[2, 2, 2],
            num_res_blocks=[3, 3, 4, 6, 3],
            latent_dim=12,
            has_attention=[False, False, False, False, False],
            output_norm=False,
            nonlinearity="silu",
            output_nonlinearity="silu",
            causal=True,
        )

        return decoder
