This commit has end to end pipeline for the vovnet model.
The pcc of the model is low(~0.0) as the fused conv fails to achieve pcc > 0.99 with the pretrained weights.

To reproduce the results, run the following command:
Checkout to the branch
`pytest models/experimental/vovnet_fused_conv/tests/test_tt_vovnet.py`
