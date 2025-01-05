Silu ops are failing with Out of Memory issue:
"""
E       RuntimeError: TT_THROW @ ../tt_metal/impl/allocator/allocator.cpp:145: tt::exception
E       info:
E       Out of Memory: Not enough space to allocate 35651584 B L1 buffer across 64 banks, where each bank needs to store 557056 B
"""

Run the command to test: `pytest tests/ttnn/unit_tests/operations/test_silu.py::test_sd35_vae_512`
