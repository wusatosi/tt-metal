# Stable Diffusion Turbo-XL

This branch contains sub-modules test for the following Sub-modules:

1. GeGlu
2. FeedForward
3. Attention(s)
4. BasicTransformerBlock
5. Transformer2d
6. CrossAttentionDownBlock2d

## Steps to Run the sub-module tests for 1024x1024 resolution:

### Update the latest diffusers package

```
 pip install diffusers transformers accelerate --upgrade
```

### To run the test for Geglu sub-Module:

```
pytest tests/ttnn/integration_tests/stable_diffusion_xl_turbo/test_sd.py::test_geglu
```

### To run the test for FeedForward sub-Module:

```
pytest tests/ttnn/integration_tests/stable_diffusion_xl_turbo/test_sd.py::test_feed_forward
```

### To run the test for Attention sub-module from
- CrossAttentionDownBlock2D

```
pytest tests/ttnn/integration_tests/stable_diffusion_xl_turbo/test_sd.py::test_attention_down_blocks
```

- CrossAttentionUpBlock2D
```
pytest tests/ttnn/integration_tests/stable_diffusion_xl_turbo/test_sd.py::test_attention_up_blocks
 ```

- UNetMidBlock2DCrossAttn

```
pytest tests/ttnn/integration_tests/stable_diffusion_xl_turbo/test_sd.py::test_attention_mid_blocks
```

### To run the test for BasicTransformerBlock sub-module.
```
pytest tests/ttnn/integration_tests/stable_diffusion_xl_turbo/test_sd.py::test_basic_transformer_block
```

### To run the test for Transformer2D sub-module.
```
pytest tests/ttnn/integration_tests/stable_diffusion_xl_turbo/test_sd.py::test_transformer_2d_model
```
### To run the test for CrossAttentionDownBlock2d sub-module.
```
tests/ttnn/integration_tests/stable_diffusion_xl_turbo/test_sd.py::test_cross_attention_downblock2d
```


## Steps to Run the sub-module tests for 512x512 resolution:

### Update the latest diffusers package

```
 pip install diffusers transformers accelerate --upgrade
```

### To run the test for BasicTransformer2D sub-module in 512x512 resolution.
```
pytest tests/ttnn/integration_tests/stable_diffusion_xl_turbo/test_sd_512x512.py::test_basic_transformer_block_512_512
```

### To run the test for Transformer2D sub-module in 512x512 resolution.
```
pytest tests/ttnn/integration_tests/stable_diffusion_xl_turbo/test_sd_512x512.py::test_transformer_2d_model_512_512
```
