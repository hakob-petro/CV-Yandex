import torch
import torchvision
from torch.ao.quantization import get_default_qconfig_mapping, quantize_fx

from fp32_model import SimpleNet


def quantize():
    fp32_model = SimpleNet()
    checkpoint = torch.load("fp32_model.pth")
    fp32_model.load_state_dict(checkpoint)
    fp32_model.eval()

    # prepare (observers are inserted)
    qconfig_mapping = get_default_qconfig_mapping("x86")
    example_inputs = (torch.randn(1, 1, 28, 28),)
    model_prepared = quantize_fx.prepare_fx(
        fp32_model,
        qconfig_mapping,
        example_inputs,
    )

    # calibrate
    calibration_dataset = torchvision.datasets.MNIST(
        "mnist",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    with torch.inference_mode():
        for image_idx in range(500):
            model_prepared(calibration_dataset[image_idx][0].unsqueeze(0))

    # quantize
    quantized_model_pytorch = quantize_fx.convert_fx(model_prepared)
    quantized_model_pytorch.eval()
    torch.save(quantized_model_pytorch.state_dict(), "quantized_model.pth")


def load_quantized_model_pytorch():
    fp32_model = SimpleNet()

    # prepare (observers are inserted)
    qconfig_mapping = get_default_qconfig_mapping("x86")
    example_inputs = (torch.randn(1, 1, 28, 28),)
    model_prepared = quantize_fx.prepare_fx(
        fp32_model,
        qconfig_mapping,
        example_inputs,
    )

    # load weights
    quantized_model_pytorch = quantize_fx.convert_fx(model_prepared)
    quantized_model_pytorch.load_state_dict(torch.load("quantized_model.pth"))
    return quantized_model_pytorch


if __name__ == "__main__":
    quantize()
