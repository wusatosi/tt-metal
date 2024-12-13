# Stable Diffusion Turbo-XL

This branch contains sub-modules test for the following Sub-modules:

1. TimeStepEmbedding
2. Downsample2d
3. Upsample2d
4. DownBlock2d
5. UpBlock2d

## Steps to Run the sub-module tests:

### Update the latest diffusers package

```
 pip install diffusers transformers accelerate --upgrade
 ```

### To run the test for TimeStepEmbedding sub-Module:

```
pytest models/demos/stable_diffusion_xl_turbo/tests/test_timestep_embedding.py
```

### To run the test for Downsample and Upsample sub-module
```
pytest models/demos/stable_diffusion_xl_turbo/tests/test_downsample_2d.py
```
```
pytest models/demos/stable_diffusion_xl_turbo/tests/test_upsample_2d.py
```

### To run the test for DownBlock2d and UpBlock2d sub-module.
```
pytest models/demos/stable_diffusion_xl_turbo/tests/test_downblock_2d.py
```
```
pytest models/demos/stable_diffusion_xl_turbo/tests/test_upblock_2d.py
```
