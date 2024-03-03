import pathlib

import numpy as np
import torch
import torchvision

from fp32_model import SimpleNet
from manual_quantization import get_quantized_model_manual
from pytorch_quantization import load_quantized_model_pytorch
from solution import dequantize, quantize

try:
    import tensorflow as tf
except ImportError as exc:
    tf = None


def compare_models():
    fp32_model = SimpleNet()
    checkpoint = torch.load("fp32_model.pth")
    fp32_model.load_state_dict(checkpoint)
    fp32_model.eval()

    use_tflite_quant = False
    if pathlib.Path("quantized_model.tflite").is_file() and tf is not None:
        use_tflite_quant = True

    use_pytorch_quant = False
    if pathlib.Path("quantized_model.pth").is_file():
        use_pytorch_quant = True

    quantized_model_manual = get_quantized_model_manual(fp32_model)

    if use_pytorch_quant:
        quantized_model_pytorch = load_quantized_model_pytorch()

    # Comparison
    counter_qmodel_manual = 0
    counter_qmodel_tflite = 0
    counter_qmodel_pytorch = 0
    counter_fp32 = 0

    NUM_INPUTS = 1000  # set to 10000 to test on the whole test dataset

    test_dataset = torchvision.datasets.MNIST(
        "mnist",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    for idx in range(NUM_INPUTS):
        image = test_dataset[idx][0]
        label = test_dataset[idx][1]

        # manual quantized model inference
        input_quantized = quantize(
            image.numpy()[0],
            quantized_model_manual.qparams_input,
        )
        output = quantized_model_manual(input_quantized).astype(np.float32)
        output_dequantized = dequantize(
            output,
            quantized_model_manual.qparams_output,
        )
        counter_qmodel_manual += output_dequantized.argmax() == label

        # tflite quantized model inference
        if use_tflite_quant:
            interpreter = tf.lite.Interpreter(model_path="quantized_model.tflite")
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()[0]
            output_details = interpreter.get_output_details()[0]
            interpreter.set_tensor(
                input_details["index"],
                np.expand_dims(input_quantized, 0),
            )
            interpreter.invoke()
            output_tflite = interpreter.get_tensor(output_details["index"])[0]
            counter_qmodel_tflite += output_tflite.argmax().item() == label

        # pytorch quantized model inference
        if use_pytorch_quant:
            output_pytorch = quantized_model_pytorch(image.unsqueeze(0))
            counter_qmodel_pytorch += output_pytorch.argmax().item() == label

        # fp32 model inference
        output_fp32 = fp32_model(image.unsqueeze(0))
        counter_fp32 += output_fp32.argmax().item() == label

        print(
            f"Evaluation progress: {(idx+1) * 100 / NUM_INPUTS}%",
            end="\r",
            flush=True,
        )

    print("\n\nModel accuracies:")
    print(
        f"manually quantized: {counter_qmodel_manual}/{NUM_INPUTS} "
        f"({100*counter_qmodel_manual/NUM_INPUTS:.2f}%)"
    )
    if use_tflite_quant:
        print(
            f"tflite quantized:   {counter_qmodel_tflite}/{NUM_INPUTS} "
            f"({100*counter_qmodel_tflite/NUM_INPUTS:.2f}%)"
        )
    if use_pytorch_quant:
        print(
            f"pytorch quantized:  {counter_qmodel_pytorch}/{NUM_INPUTS} "
            f"({100*counter_qmodel_pytorch/NUM_INPUTS:.2f}%)"
        )
    print(
        f"fp32:               {counter_fp32}/{NUM_INPUTS} "
        f"({100*counter_fp32/NUM_INPUTS:.2f}%)"
    )


if __name__ == "__main__":
    compare_models()
