import pathlib

import numpy as np
import tensorflow as tf
import torch
import torchvision

from fp32_model import SimpleNet


def quantize():
    fp32_model = SimpleNet()
    checkpoint = torch.load("fp32_model.pth")
    fp32_model.load_state_dict(checkpoint)
    fp32_model.eval()
    parameters = list(fp32_model.parameters())
    parameters_np = [p.detach().cpu().numpy() for p in parameters]

    model_keras = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(28, 28)),
            tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Permute((3, 1, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10),
        ]
    )

    # load the trained weights from the pytorch model
    conv_weights = np.zeros((3, 3, 1, 12), np.float32)
    for i in range(12):
        conv_weights[:, :, 0, i] = parameters_np[0][i, 0]
    conv_biases = parameters_np[1]

    fc_weights = np.zeros((2028, 10), np.float32)
    for i in range(10):
        fc_weights[:, i] = parameters_np[2][i]
    fc_biases = parameters_np[3]

    model_keras.set_weights([conv_weights, conv_biases, fc_weights, fc_biases])

    # calibration dataset
    calibration_dataset = torchvision.datasets.MNIST(
        "mnist",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    def representative_data_gen():
        for idx in range(500):
            input_image = calibration_dataset[idx][0].numpy()
            yield [input_image]

    # quantize
    converter = tf.lite.TFLiteConverter.from_keras_model(model_keras)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    quantized_model_tflite = converter.convert()

    tflite_models_dir = pathlib.Path(".")
    tflite_model_quant_file = tflite_models_dir / "quantized_model.tflite"
    tflite_model_quant_file.write_bytes(quantized_model_tflite)


if __name__ == "__main__":
    quantize()
