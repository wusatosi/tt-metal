import ttnn


def residual_tanh_gated_rmsnorm(x, x_res, gate, eps=1e-6, high_fidelity=False):
    if high_fidelity:
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
    else:
        compute_kernel_config = None
    tanh_gate = ttnn.tanh(gate)
    x_normed = ttnn.rms_norm(x_res, weight=tanh_gate, epsilon=eps, compute_kernel_config=compute_kernel_config)
    output = x + x_normed
    return output


def modulated_rmsnorm(x, scale, eps=1e-6, high_fidelity=False):
    if high_fidelity:
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
    else:
        compute_kernel_config = None
    weight = 1.0 + scale
    x = ttnn.rms_norm(x, weight=weight, epsilon=eps, compute_kernel_config=compute_kernel_config)
    return x
