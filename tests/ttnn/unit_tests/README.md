This commit includes a unit test for the fused Conv2D operation (where fused conv = Conv2D + BatchNorm2D + optional activation).

The PCC of the fused conv drops significantly when using pre-trained weights in the unit test. The test is designed to initialize the Conv and BatchNorm layers with real (pre-trained) weights, and the weights are fused using the method described in the PyTorch documentation [here](<https://pytorch.org/tutorials/intermediate/fx_conv_bn_fuser.html#:~:text=def%20fuse_conv_bn_eval(,(conv_b)>)
