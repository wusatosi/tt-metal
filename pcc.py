import numpy as np
import torch
from tests.ttnn.utils_for_testing import assert_with_pcc


print("---noise pred ---")
for i in range(40):
    # orig = torch.from_numpy(np.load("../../sd35_512_unopt/tt-metal/models/experimental/functional_stable_diffusion3_5/demo/first_npy/demo_unoptimized_512x512__noise_pred_"+ str(i)+ ".npy"))
    orig = torch.from_numpy(
        np.load(
            "../../sd35_512_unopt/tt-metal/models/experimental/functional_stable_diffusion3_5/demo/demo_unoptimized_512x512__noise_pred_"
            + str(i)
            + ".npy"
        )
    )
    new = torch.from_numpy(
        np.load(
            "models/experimental/functional_stable_diffusion3_5/demo/demo_optimGuidedd_512x512__noise_pred_"
            + str(i)
            + ".npy"
        )
    )
    print(assert_with_pcc(orig, new, pcc=-100))

    # new = torch.from_numpy(np.load("../../sd35_512_unopt/tt-metal/models/experimental/functional_stable_diffusion3_5/demo/first_npy/demo_unoptimized_512x512__pooled_proj_"+ str(i)+ ".npy"))
    # print( assert_with_pcc(orig, new, pcc=-100))


print()
print("---noise pred post guidance ---")
for i in range(40):
    orig = torch.from_numpy(
        np.load(
            "../../sd35_512_unopt/tt-metal/models/experimental/functional_stable_diffusion3_5/demo/demo_unoptimized_512x512__noise_pred_ii_"
            + str(i)
            + ".npy"
        )
    )
    new = torch.from_numpy(
        np.load(
            "models/experimental/functional_stable_diffusion3_5/demo/demo_optimGuidedd_512x512__noise_pred_ii_"
            + str(i)
            + ".npy"
        )
    )
    print(assert_with_pcc(orig, new, pcc=-100))


print()
print("---time step ---")
for i in range(40):
    orig = torch.from_numpy(
        np.load(
            "../../sd35_512_unopt/tt-metal/models/experimental/functional_stable_diffusion3_5/demo/demo_unoptimizedd_512x512__ttt_i_"
            + str(i)
            + ".npy"
        )
    )
    new = torch.from_numpy(
        np.load(
            "models/experimental/functional_stable_diffusion3_5/demo/demo_optimGuidedd_512x512__ttt_i_"
            + str(i)
            + ".npy"
        )
    )
    print(assert_with_pcc(orig, new, pcc=-100))


print()
print("---latents-prev-----")

for i in range(40):
    orig = torch.from_numpy(
        np.load(
            "../../sd35_512_unopt/tt-metal/models/experimental/functional_stable_diffusion3_5/demo/demo_unoptimized_512x512__latents_old_"
            + str(i)
            + ".npy"
        )
    )
    new = torch.from_numpy(
        np.load(
            "models/experimental/functional_stable_diffusion3_5/demo/demo_optimGuidedd_512x512__latents_old_"
            + str(i)
            + ".npy"
        )
    )
    print(assert_with_pcc(orig, new, pcc=-100))


print()
print("---latents-i-----")

for i in range(40):
    orig = torch.from_numpy(
        np.load(
            "../../sd35_512_unopt/tt-metal/models/experimental/functional_stable_diffusion3_5/demo/demo_unoptimized_512x512__latents_i_"
            + str(i)
            + ".npy"
        )
    )
    new = torch.from_numpy(
        np.load(
            "models/experimental/functional_stable_diffusion3_5/demo/demo_optimGuidedd_512x512__latents_i_"
            + str(i)
            + ".npy"
        )
    )
    print(assert_with_pcc(orig, new, pcc=-100))


print()
print("---latents------")

for i in range(40):
    orig = torch.from_numpy(
        np.load(
            "../../sd35_512_unopt/tt-metal/models/experimental/functional_stable_diffusion3_5/demo/demo_unoptimized_512x512__latents_"
            + str(i)
            + ".npy"
        )
    )
    new = torch.from_numpy(
        np.load(
            "models/experimental/functional_stable_diffusion3_5/demo/demo_optimGuidedd__512x512__latents_"
            + str(i)
            + ".npy"
        )
    )
    print(assert_with_pcc(orig, new, pcc=-100))
