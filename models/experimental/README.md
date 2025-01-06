Commands to run the unit tests:
- Conv2d: `pytest tests/ttnn/unit_tests/operations/test_new_conv2d.py::test_sd35_vae_512`
- GroupNorm: `pytest tests/ttnn/unit_tests/operations/test_group_norm.py::test_sd35_vae_512`
- Silu: pytest `pytest tests/ttnn/unit_tests/operations/test_silu.py::test_sd35_vae_512`
- Upsample: `pytest tests/ttnn/unit_tests/operations/test_upsample.py::test_sd35_vae_512`
- Linear: `pytest tests/ttnn/unit_tests/operations/test_linear.py::test_sd35_vae_512`

Conv2d: 3 out of 12 cases Passed, 9 cases fails with OOM issue
GroupNorm: 4 out of 7 cases fails with OOM, remaining 3 cases fails with shape mismatch error
Silu: 2 out of 6 cases Passed in Bfloat8_b and Bfloat16. 1 case passed in Bfloat8_b and failed in Bfloat16. Remaining 3 cases fails with OOM issue.
Upsample: 3 out of 3 cases Passed.
Linear: 1 out of 1 case Passed.
