
# DEIT Tiny Wormhole Demo

## DEIT Model Overview

The Data-efficient Image Transformer (DEIT) model is a variant of the Vision Transformer (ViT) that focuses on training efficiency and performance. It was introduced in the paper "Training data-efficient image transformers & distillation through attention." DEIT achieves competitive results on ImageNet while requiring fewer resources and training data.

For more details, refer to the official documentation: [DEIT Model Documentation](https://huggingface.co/docs/transformers/en/model_doc/deit)

## How to Run

To run the demo for the DEIT model, follow these instructions:


- For the inference overall runtime (end-2-end), use the following command:

  ```sh
  pytest --disable-warnings models/demos/wormhole/deit_tiny/demo/demo_deit_ttnn_inference_perf_e2e_2cq_trace.py
  ```

- For running the inference device OPs analysis, use the following command:

  ```sh
  pytest --disable-warnings models/demos/wormhole/deit_tiny/demo/demo_deit_ttnn_inference_device_OPs.py
  ```

- For running the demo and benchmarking, use the following command:

  ```sh
  pytest --disable-warnings models/demos/wormhole/deit_tiny/demo/demo_deit_ttnn_cifar10_inference.py
  ```

- For running the prediction on images, use the following command:

  ```sh
  pytest --disable-warnings models/demos/wormhole/deit_tiny/demo/demo_deit_ttnn_predict.py
  ```

  > **Note:** Ensure that the images to be used for predictions are stored in the `images` folder. When the `demo_deit_ttnn_predict.py` script is run, the predictions will be saved in the `results` folder.

