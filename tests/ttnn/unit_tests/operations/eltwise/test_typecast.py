import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def create_full_range_tensor(input_shapes, dtype):
    num_elements = torch.prod(torch.tensor(input_shapes)).item()

    # Create tensors covering different important parts of bfloat16 range
    large_negatives = torch.linspace(-3.3e38, -1e30, steps=num_elements // 5, dtype=dtype)
    medium_negatives = torch.linspace(-1e5, -1e-3, steps=num_elements // 5, dtype=dtype)
    near_zero = torch.linspace(-1e-5, 1e-5, steps=num_elements // 5, dtype=dtype)
    medium_positives = torch.linspace(1e-3, 1e5, steps=num_elements // 5, dtype=dtype)
    large_positives = torch.linspace(1e30, 3.3e38, steps=num_elements // 5, dtype=dtype)

    # Concatenate
    in_data = torch.cat([large_negatives, medium_negatives, near_zero, medium_positives, large_positives])

    corner_cases = torch.tensor([0.0], dtype=dtype)
    in_data = torch.cat([in_data, corner_cases])

    # Ensure correct number of elements
    in_data = in_data[:num_elements]
    in_data = in_data.reshape(input_shapes)

    return in_data


@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([1, 1, 32, 32]),
    ],
)
@pytest.mark.parametrize(
    "output_dtype",
    [
        ttnn.float32,
        ttnn.bfloat8_b,
        # ttnn.bfloat4_b,  (mismatches)
        ttnn.int32,
        ttnn.uint32,
        ttnn.uint16,
        # ttnn.uint8, (fails)
    ],
)
def test_typecast_from_bf16_ttnn(input_shapes, output_dtype, device):
    num_elements = torch.prod(torch.tensor(input_shapes)).item()
    if output_dtype in (ttnn.float32, ttnn.bfloat8_b):
        input_shapes = torch.Size([1, 1, 64, 64])
        uniform_values = create_full_range_tensor(input_shapes, torch.bfloat16)
        corner_cases = torch.tensor([], dtype=torch.bfloat16)
    elif output_dtype == ttnn.bfloat4_b:
        uniform_values = torch.linspace(-100, 100, num_elements, dtype=torch.bfloat16)
        corner_cases = torch.tensor([0.0, 1.0], dtype=torch.bfloat16)
    elif output_dtype == ttnn.int32:
        uniform_values = torch.linspace(-2147483647, 2139095040, num_elements, dtype=torch.bfloat16)  # torch limitation
        corner_cases = torch.tensor([0.0, 1.0, 2139095040.0], dtype=torch.bfloat16)
    elif output_dtype == ttnn.uint32:
        uniform_values = torch.linspace(0, 2139095040.0, num_elements, dtype=torch.bfloat16)  # torch limitation
        corner_cases = torch.tensor([0.0, 1.0, 2139095040.0], dtype=torch.bfloat16)
    elif output_dtype == ttnn.uint16:
        uniform_values = torch.linspace(0, 32700, num_elements, dtype=torch.bfloat16)  # torch limitation
        corner_cases = torch.tensor(
            [
                0.0,
                1.0,
            ],
            dtype=torch.bfloat16,
        )
    elif output_dtype == ttnn.uint8:
        uniform_values = torch.linspace(0, 255, num_elements, dtype=torch.bfloat16)
        corner_cases = torch.tensor([0.0, 1.0, 255.0], dtype=torch.bfloat16)

    in_data = torch.cat([uniform_values, corner_cases])

    in_data = in_data[-num_elements:].reshape(input_shapes)

    input_tensor = ttnn.from_torch(in_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.typecast(input_tensor, dtype=output_dtype)

    output = ttnn.to_torch(output_tensor)

    if output_dtype in (ttnn.int32, ttnn.uint32, ttnn.uint16, ttnn.uint8):
        in_data = ttnn.to_torch(input_tensor, dtype=torch.int32)
    else:
        in_data = ttnn.to_torch(input_tensor)

    pcc = ttnn.pearson_correlation_coefficient(in_data, output)
    assert pcc >= 0.999


@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([1, 1, 32, 32]),
    ],
)
@pytest.mark.parametrize(
    "output_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        # ttnn.bfloat4_b,  (mismatches)
        ttnn.int32,
        ttnn.uint32,
        ttnn.uint16,
        # ttnn.uint8, (fails)
    ],
)
def test_typecast_from_fp32_ttnn(input_shapes, output_dtype, device):
    num_elements = torch.prod(torch.tensor(input_shapes)).item()
    if output_dtype in (ttnn.bfloat16, ttnn.bfloat8_b):
        input_shapes = torch.Size([1, 1, 64, 64])
        uniform_values = create_full_range_tensor(input_shapes, torch.float32)
        corner_cases = torch.tensor([], dtype=torch.float32)
    elif output_dtype == ttnn.bfloat4_b:
        uniform_values = torch.linspace(-100, 100, num_elements, dtype=torch.float32)
        corner_cases = torch.tensor([0.0, 1.0], dtype=torch.float32)
    elif output_dtype == ttnn.int32:
        uniform_values = torch.linspace(-2147483647, 2139095040, num_elements, dtype=torch.float32)  # torch limitation
        corner_cases = torch.tensor([0.0, 1.0, 2139095040.0], dtype=torch.float32)
    elif output_dtype == ttnn.uint32:
        uniform_values = torch.linspace(0, 2139095040.0, num_elements, dtype=torch.float32)  # torch limitation
        corner_cases = torch.tensor([0.0, 1.0, 2139095040.0], dtype=torch.float32)
    elif output_dtype == ttnn.uint16:
        uniform_values = torch.linspace(0, 32700, num_elements, dtype=torch.float32)  # torch limitation
        corner_cases = torch.tensor(
            [
                0.0,
                1.0,
            ],
            dtype=torch.float32,
        )
    elif output_dtype == ttnn.uint8:
        uniform_values = torch.linspace(0, 255, num_elements, dtype=torch.float32)
        corner_cases = torch.tensor([0.0, 1.0, 255.0], dtype=torch.float32)

    in_data = torch.cat([uniform_values, corner_cases])

    in_data = in_data[-num_elements:].reshape(input_shapes)

    input_tensor = ttnn.from_torch(in_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.typecast(input_tensor, dtype=output_dtype)

    output = ttnn.to_torch(output_tensor)

    if output_dtype in (ttnn.int32, ttnn.uint32, ttnn.uint16, ttnn.uint8):
        in_data = ttnn.to_torch(input_tensor, dtype=torch.int32)
    else:
        in_data = ttnn.to_torch(input_tensor)

    pcc = ttnn.pearson_correlation_coefficient(in_data, output)
    assert pcc >= 0.999


@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([1, 1, 32, 32]),
    ],
)
@pytest.mark.parametrize(
    "output_dtype",
    [
        ttnn.bfloat16,
        # ttnn.bfloat4_b, (mismatches)
        ttnn.float32,
        ttnn.int32,
        ttnn.uint32,
        ttnn.uint16,
        # ttnn.uint8, (fails)
    ],
)
def test_typecast_from_bf8b_ttnn(input_shapes, output_dtype, device):
    num_elements = torch.prod(torch.tensor(input_shapes)).item()
    if output_dtype in (ttnn.bfloat16, ttnn.float32):
        input_shapes = torch.Size([1, 1, 64, 64])
        uniform_values = create_full_range_tensor(input_shapes, torch.bfloat16)
        corner_cases = torch.tensor([], dtype=torch.bfloat16)
    elif output_dtype == ttnn.bfloat4_b:
        uniform_values = torch.linspace(-100, 100, num_elements, dtype=torch.bfloat16)
        corner_cases = torch.tensor([0.0, 1.0], dtype=torch.bfloat16)
    elif output_dtype == ttnn.int32:
        uniform_values = torch.linspace(-2147483647, 2139095040, num_elements, dtype=torch.bfloat16)  # torch limitation
        corner_cases = torch.tensor([0.0, 1.0, 2139095040.0], dtype=torch.bfloat16)
    elif output_dtype == ttnn.uint32:
        uniform_values = torch.linspace(0, 2139095040.0, num_elements, dtype=torch.bfloat16)  # torch limitation
        corner_cases = torch.tensor([0.0, 1.0, 2139095040.0], dtype=torch.bfloat16)
    elif output_dtype == ttnn.uint16:
        uniform_values = torch.linspace(0, 32700, num_elements, dtype=torch.bfloat16)  # torch limitation
        corner_cases = torch.tensor(
            [
                0.0,
                1.0,
            ],
            dtype=torch.bfloat16,
        )
    elif output_dtype == ttnn.uint8:
        uniform_values = torch.linspace(0, 255, num_elements, dtype=torch.bfloat16)
        corner_cases = torch.tensor([0.0, 1.0, 255.0], dtype=torch.bfloat16)

    in_data = torch.cat([uniform_values, corner_cases])

    in_data = in_data[-num_elements:].reshape(input_shapes)

    input_tensor = ttnn.from_torch(in_data, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.typecast(input_tensor, dtype=output_dtype)

    output = ttnn.to_torch(output_tensor)

    if output_dtype in (ttnn.int32, ttnn.uint32, ttnn.uint16, ttnn.uint8):
        in_data = ttnn.to_torch(input_tensor, dtype=torch.int32)
    else:
        in_data = ttnn.to_torch(input_tensor)

    pcc = ttnn.pearson_correlation_coefficient(in_data, output)
    assert pcc >= 0.999


@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([1, 1, 32, 32]),
    ],
)
@pytest.mark.parametrize(
    "output_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        ttnn.float32,
        ttnn.int32,
        ttnn.uint32,
        ttnn.uint16,
        # ttnn.uint8, (fails)
    ],
)
def test_typecast_from_bf4b_ttnn(input_shapes, output_dtype, device):
    num_elements = torch.prod(torch.tensor(input_shapes)).item()
    if output_dtype in (ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b):
        input_shapes = torch.Size([1, 1, 64, 64])
        uniform_values = create_full_range_tensor(input_shapes, torch.bfloat16)
        corner_cases = torch.tensor([], dtype=torch.bfloat16)
    elif output_dtype == ttnn.int32:
        uniform_values = torch.linspace(-2147483647, 2139095040, num_elements, dtype=torch.bfloat16)  # torch limitation
        corner_cases = torch.tensor([0.0, 1.0, 2139095040.0], dtype=torch.bfloat16)
    elif output_dtype == ttnn.uint32:
        uniform_values = torch.linspace(0, 2139095040.0, num_elements, dtype=torch.bfloat16)  # torch limitation
        corner_cases = torch.tensor([0.0, 1.0, 2139095040.0], dtype=torch.bfloat16)
    elif output_dtype == ttnn.uint16:
        uniform_values = torch.linspace(0, 32700, num_elements, dtype=torch.bfloat16)  # torch limitation
        corner_cases = torch.tensor(
            [
                0.0,
                1.0,
            ],
            dtype=torch.bfloat16,
        )
    elif output_dtype == ttnn.uint8:
        uniform_values = torch.linspace(0, 255, num_elements, dtype=torch.bfloat16)
        corner_cases = torch.tensor([0.0, 1.0, 255.0], dtype=torch.bfloat16)

    in_data = torch.cat([uniform_values, corner_cases])

    in_data = in_data[-num_elements:].reshape(input_shapes)

    input_tensor = ttnn.from_torch(in_data, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.typecast(input_tensor, dtype=output_dtype)

    output = ttnn.to_torch(output_tensor)

    if output_dtype in (ttnn.int32, ttnn.uint32, ttnn.uint16, ttnn.uint8):
        in_data = ttnn.to_torch(input_tensor, dtype=torch.int32)
    else:
        in_data = ttnn.to_torch(input_tensor)

    pcc = ttnn.pearson_correlation_coefficient(in_data, output)
    assert pcc >= 0.999


@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([1, 1, 32, 32]),
    ],
)
@pytest.mark.parametrize(
    "output_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        ttnn.bfloat4_b,
        ttnn.float32,
        ttnn.uint32,
        # ttnn.uint16, (fails)
        ttnn.uint8,
    ],
)
def test_typecast_from_int32_ttnn(input_shapes, output_dtype, device):
    num_elements = torch.prod(torch.tensor(input_shapes)).item()
    if output_dtype == ttnn.uint32:
        uniform_values = torch.linspace(0, 2147483647, num_elements, dtype=torch.int32)
    elif output_dtype == ttnn.uint16:
        uniform_values = torch.linspace(0, 32700, num_elements, dtype=torch.float32)
    elif output_dtype == ttnn.uint8:
        uniform_values = torch.linspace(0, 255, num_elements, dtype=torch.int32)
    else:
        uniform_values = torch.linspace(-2147483647, 2147483647, num_elements, dtype=torch.int32)

    corner_cases = torch.tensor(
        [
            0,
            1,
            -1,
        ],
        dtype=torch.int32,
    )
    in_data = torch.cat([uniform_values, corner_cases])

    in_data = in_data[-num_elements:].reshape(input_shapes)

    input_tensor = ttnn.from_torch(in_data, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.typecast(input_tensor, dtype=output_dtype)

    output = ttnn.to_torch(output_tensor)

    if output_dtype in (ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.float32):
        in_data = in_data.to(torch.bfloat16)
    else:
        in_data = ttnn.to_torch(input_tensor)

    pcc = ttnn.pearson_correlation_coefficient(in_data, output)
    assert pcc >= 0.99


@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([1, 1, 32, 32]),
    ],
)
@pytest.mark.parametrize(
    "output_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        ttnn.bfloat4_b,
        ttnn.float32,
        ttnn.int32,
        # ttnn.uint16,   #(fails)
        ttnn.uint8,
    ],
)
def test_typecast_from_uint32_ttnn(input_shapes, output_dtype, device):
    num_elements = torch.prod(torch.tensor(input_shapes)).item()
    if output_dtype == ttnn.uint16:
        uniform_values = torch.linspace(0, 32700, num_elements, dtype=torch.float32)
    elif output_dtype == ttnn.uint8:
        uniform_values = torch.linspace(0, 255, num_elements, dtype=torch.int32)
    else:
        uniform_values = torch.linspace(0, 2147483647, num_elements, dtype=torch.int32)

    corner_cases = torch.tensor(
        [
            0,
            1,
        ],
        dtype=torch.int32,
    )
    in_data = torch.cat([uniform_values, corner_cases])

    in_data = in_data[-num_elements:].reshape(input_shapes)

    input_tensor = ttnn.from_torch(in_data, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.typecast(input_tensor, dtype=output_dtype)

    output = ttnn.to_torch(output_tensor)

    if output_dtype in (ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.float32):
        in_data = ttnn.to_torch(input_tensor, dtype=torch.bfloat16)
    else:
        ttnn.to_torch(input_tensor)

    pcc = ttnn.pearson_correlation_coefficient(in_data, output)
    assert pcc >= 0.99


@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([1, 1, 32, 32]),
    ],
)
@pytest.mark.parametrize(
    "output_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        ttnn.bfloat4_b,
        ttnn.float32,
        # ttnn.int32, (fails)
        ttnn.uint32,
        # ttnn.uint8, (fails)
    ],
)
def test_typecast_from_uint16_ttnn(input_shapes, output_dtype, device):
    num_elements = torch.prod(torch.tensor(input_shapes)).item()
    if output_dtype == ttnn.uint8:
        uniform_values = torch.linspace(0, 255, num_elements, dtype=torch.int32)
    else:
        uniform_values = torch.linspace(0, 65535, num_elements, dtype=torch.int32)

    corner_cases = torch.tensor(
        [
            0,
            1,
        ],
        dtype=torch.int32,
    )
    in_data = torch.cat([uniform_values, corner_cases])

    in_data = in_data[-num_elements:].reshape(input_shapes)
    input_tensor = ttnn.from_torch(in_data, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.typecast(input_tensor, dtype=output_dtype)

    output = ttnn.to_torch(output_tensor)

    if output_dtype in (ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.float32):
        in_data = in_data.to(torch.bfloat16)
    else:
        ttnn.to_torch(input_tensor)

    pcc = ttnn.pearson_correlation_coefficient(in_data, output)
    assert pcc >= 0.99


@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([1, 1, 32, 32]),
    ],
)
@pytest.mark.parametrize(
    "output_dtype",
    [
        # ttnn.bfloat16,    (fails)
        # ttnn.bfloat8_b,   (fails)
        # ttnn.bfloat4_b,   (fails)
        # ttnn.float32,     (fails)
        ttnn.int32,
        ttnn.uint32,
        # ttnn.uint16,        (fails)
    ],
)
def test_typecast_from_uint8_ttnn(input_shapes, output_dtype, device):
    num_elements = torch.prod(torch.tensor(input_shapes)).item()
    uniform_values = torch.linspace(0, 255, num_elements, dtype=torch.int32)

    corner_cases = torch.tensor(
        [
            0,
            1,
        ],
        dtype=torch.int32,
    )
    in_data = torch.cat([uniform_values, corner_cases])

    in_data = in_data[-num_elements:].reshape(input_shapes)

    input_tensor = ttnn.from_torch(in_data, dtype=ttnn.uint8, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.typecast(input_tensor, dtype=output_dtype)

    output = ttnn.to_torch(output_tensor)

    if output_dtype in (ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.float32):
        in_data = in_data.to(torch.bfloat16)

    pcc = ttnn.pearson_correlation_coefficient(in_data, output)
    assert pcc >= 0.999
