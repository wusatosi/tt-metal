This unit test is done to check if the following torch code snippet can be converted to ttnn:
```
y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i]
y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
```

Since, the indexing item assignment is not supported in ttnn, we have the following workarounds done:

1. Checked if the torch indexing item assignment can be replaced by torch slice followed by concat operation, getting PCC of 0.947206171395425
2. Checked if the torch indexing item assignment  can be replace by ttnn slice followed by concat operation.
   - On trying to add 1.0 to the ttnn sliced input, facing the following issue:
   ```
    E       RuntimeError: TT_FATAL @ /home/ubuntu/new_repo/tt-metal/ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_op.cpp:122: (output_tensor_shape[-1] % TILE_WIDTH == 0) && (this->slice_start[-1] % TILE_WIDTH == 0)
    E       info:
    E       Can only unpad tilized tensor with full tiles
    ```
