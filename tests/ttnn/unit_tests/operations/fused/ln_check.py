import torch
import ttnn
import numpy as np
import os

USE_RAND = True
ttnn.set_printoptions(profile="short")


class LightweightModule:
    """Lightweight version of PyTorch's nn.Module"""

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class TorchLayerNorm(LightweightModule):
    def __init__(self, device, weight, bias=None, eps=1e-5):
        super().__init__()
        self.device = device
        self.weight = weight
        self.bias = bias
        self.eps = eps

    def forward(self, x):
        torch_x = ttnn.to_torch(x)
        torch_weight = ttnn.to_torch(self.weight)
        torch_bias = ttnn.to_torch(self.bias) if self.bias is not None else None

        normalized_x_torch = torch.nn.functional.layer_norm(
            torch_x, normalized_shape=torch_x.shape[-1:], weight=torch_weight, bias=torch_bias, eps=self.eps
        )

        result_ttnn = ttnn.from_torch(
            normalized_x_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        return result_ttnn


class TTNNLayerNorm(LightweightModule):
    def __init__(self, weight, bias=None, eps=1e-5):
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.eps = eps

    def forward(self, x):
        normalized_x = ttnn.layer_norm(
            x,
            epsilon=self.eps,
            weight=self.weight,
            bias=self.bias,
            memory_config=ttnn.get_memory_config(x),
        )

        if normalized_x.dtype != ttnn.bfloat16:
            normalized_x = ttnn.typecast(normalized_x, dtype=ttnn.bfloat16)

        return normalized_x


def create_random_data():
    np_input = np.random.randn(1, 6, 2048).astype(np.float32)
    np_weight = np.random.randn(2048).astype(np.float32)
    np_bias = np.random.randn(2048).astype(np.float32)

    return np_input, np_weight, np_bias


def run_comparison():
    if USE_RAND:
        np_input, np_weight, np_bias = create_random_data()
    else:
        input_path = "ln_numpy/layernorm_input.npy"
        weight_path = "ln_numpy/layernorm_weight.npy"
        bias_path = "ln_numpy/layernorm_bias.npy"

        for path in [input_path, weight_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required file not found: {path}")

        np_input = np.load(input_path)
        np_weight = np.load(weight_path)
        np_bias = np.load(bias_path) if os.path.exists(bias_path) else None

        if os.path.exists(bias_path):
            try:
                loaded_bias = np.load(bias_path, allow_pickle=True)
                if isinstance(loaded_bias, np.ndarray) and loaded_bias.size > 0:
                    np_bias = loaded_bias
            except Exception as e:
                print(f"Error loading bias file: {e}")

    device_id = 0
    device = ttnn.open_device(device_id=device_id)

    target_dtype = ttnn.bfloat16
    target_layout = ttnn.TILE_LAYOUT

    tt_input = ttnn.from_torch(
        torch.from_numpy(np_input).to(torch.bfloat16),
        dtype=target_dtype,
        layout=target_layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_weight = ttnn.from_torch(
        torch.from_numpy(np_weight).to(torch.bfloat16),
        dtype=target_dtype,
        layout=target_layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_bias = None
    if np_bias is not None:
        tt_bias = ttnn.from_torch(
            torch.from_numpy(np_bias).to(torch.bfloat16),
            dtype=target_dtype,
            layout=target_layout,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    epsilon = 1e-5
    torch_ln_ref = TorchLayerNorm(device, tt_weight, tt_bias, eps=epsilon)
    ttnn_ln_test = TTNNLayerNorm(tt_weight, tt_bias, eps=epsilon)

    print("Running reference implementation (PyTorch)...")
    tt_output_torch_ref = torch_ln_ref(tt_input)

    print("Running TTNN native implementation...")
    tt_output_ttnn_test = ttnn_ln_test(tt_input)

    torch_output_ref = ttnn.to_torch(tt_output_torch_ref).cpu().float()
    torch_output_ttnn = ttnn.to_torch(tt_output_ttnn_test).cpu().float()

    print("\nComparison Results:")

    assert (
        torch_output_ref.shape == torch_output_ttnn.shape
    ), f"Output shapes mismatch! Reference: {torch_output_ref.shape}, TTNN: {torch_output_ttnn.shape}"

    abs_diff = torch.abs(torch_output_ref - torch_output_ttnn)
    mae = torch.mean(abs_diff).item()
    mse = torch.mean(abs_diff**2).item()
    max_abs_diff = torch.max(abs_diff).item()

    rtol = 1e-2
    atol = 1e-3
    allclose_result = torch.allclose(torch_output_ref, torch_output_ttnn, rtol=rtol, atol=atol)

    print(f"torch.allclose(rtol={rtol}, atol={atol}): {allclose_result}")
    print(f"Mean Absolute Error (MAE):           {mae:.8f}")
    print(f"Mean Squared Error (MSE):            {mse:.8f}")
    print(f"Maximum Absolute Difference:         {max_abs_diff:.8f}")

    num_top_diffs = 5
    print(f"\nTop {num_top_diffs} Largest Differences:")

    flat_abs_diff = abs_diff.flatten()
    top_diff_values, top_flat_indices = torch.topk(flat_abs_diff, k=num_top_diffs)
    original_shape = abs_diff.shape

    for i in range(num_top_diffs):
        diff_value = top_diff_values[i].item()
        flat_index = top_flat_indices[i].item()

        multi_dim_index = np.unravel_index(flat_index, original_shape)

        ref_value = torch_output_ref[multi_dim_index].item()
        ttnn_value = torch_output_ttnn[multi_dim_index].item()

        rel_diff_at_index = abs_diff[multi_dim_index].item() / abs(ref_value) if ref_value != 0 else float("inf")

        print(
            f"Index {multi_dim_index}: Ref={ref_value:.6f}, TTNN={ttnn_value:.6f}, Diff={diff_value:.6f}, Rel={rel_diff_at_index:.6f}"
        )

    ttnn.close_device(device)
    print("\nComparison completed.")


if __name__ == "__main__":
    run_comparison()
