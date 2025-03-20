import torch
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import compare_pcc, data_gen_with_range
from tests.ttnn.utils_for_testing import check_with_pcc
import math

input_shape = torch.Size([1, 1, 1024, 1024])
# print(help(ttnn.experimental.gelu_bw))

device_id = 0
dispatch_core_type = ttnn.device.DispatchCoreType.ETH
device = ttnn.open_device(
    device_id=device_id, l1_small_size=8192, dispatch_core_config=ttnn.device.DispatchCoreConfig(dispatch_core_type)
)
ttnn.enable_program_cache(device)

try:
    inp0_torch, inp0_ttnn = data_gen_with_range(input_shape, -1, 1, device)
    inp1_torch, inp1_ttnn = data_gen_with_range(input_shape, -5, 5, device, True)

    tt_output_tensor_on_device_exp = [ttnn.experimental.test_op(inp0_ttnn, inp1_ttnn)]
    golden_function_exp = ttnn.get_golden_function(ttnn.experimental.gelu_bw)

    x = inp1_torch
    y = inp0_torch
    # x_square = x * x
    # x_cube = x_square * x
    # sqrt_2_over_pi = math.sqrt(2 / math.pi)
    # tanh = torch.tanh((x_cube * 0.044715 + x) * sqrt_2_over_pi)
    # cdf_term = 0.5 * (1.0 + tanh)
    # pdf_term = 0.5 * sqrt_2_over_pi * (1 + 0.134145 * x_square) * (1 - tanh * tanh)
    # golden_tensor_exp = inp0_torch * (cdf_term + x * pdf_term)
    # golden_tensor_exp = pdf_term
    golden_tensor_exp = x * y

    tt_output_tensor_on_device_exp_torch = ttnn.to_torch(tt_output_tensor_on_device_exp[0])
    # golden_pass = compare_pcc([tt_output_tensor_on_device_exp], [golden_tensor_exp])
    # composite_pass = compare_pcc([tt_output_tensor_on_device_exp], tt_output_tensor_on_device_ref)

    exp_res = check_with_pcc(golden_tensor_exp, tt_output_tensor_on_device_exp_torch, 0.999)
    print(f"[test_op pcc] {exp_res}")

finally:
    ttnn.close_device(device)
