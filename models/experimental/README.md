Convs are failing with Out of Memory issue:
"""
E       RuntimeError: TT_THROW @ ../tt_metal/impl/allocator/allocator.cpp:145: tt::exception
E       info:
E       Out of Memory: Not enough space to allocate 67108864 B L1 buffer across 8 banks, where each bank needs to store 8388608 B
"""

Run the command to test: `pytest tests/ttnn/unit_tests/operations/test_new_conv2d.py::test_sd35_vae_512`
