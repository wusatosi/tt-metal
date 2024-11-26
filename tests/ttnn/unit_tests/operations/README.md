For the following test case ttnn.reshape is failing in ConvNet Mnist model.

To recreate the issue run the command:
`pytest tests/ttnn/unit_tests/operations/test_reshape_mesh.py`

For input shape [128, 64, 6, 6]

Reshaping a tensor to from 4D to 2D shape results in a low PCC in data parallel setup. PCC = 0.000422
