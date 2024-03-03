import numpy as np
import torch
import torch.nn as nn
import torchvision

from solution import (
    MinMaxObserver,
    QuantizationParameters,
    compute_quantization_params,
    multiply_by_quantized_multiplier,
    quantize_bias,
    quantize_multiplier,
    quantize_weights_per_channel,
    quantize_weights_per_tensor,
)


class SimpleNetWithObservers(nn.Module):
    def __init__(self):
        super(SimpleNetWithObservers, self).__init__()
        self.conv = nn.Conv2d(1, 12, 3)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2028, 10)

        self.observer_input = MinMaxObserver()
        self.observer_conv_relu = MinMaxObserver()
        self.observer_fc = MinMaxObserver()

    def forward(self, x):
        self.observer_input(x)

        y = self.conv(x)
        y = self.relu(y)
        self.observer_conv_relu(y)

        y = self.maxpool(y)
        y = self.flatten(y)

        y = self.fc(y)
        self.observer_fc(y)
        return y

    def get_minmax_observations(self):
        minmax_observations = {
            "input": self.observer_input,
            "conv_relu": self.observer_conv_relu,
            "fc": self.observer_fc,
        }
        return minmax_observations


class QuantizedConvReLU:
    def __init__(
        self,
        w: np.ndarray,
        b: np.ndarray,
        qp_in: QuantizationParameters,
        qp_out: QuantizationParameters,
    ):
        # quantize parameters
        self.weights, qps_w = quantize_weights_per_channel(w)
        self.biases = [
            quantize_bias(b[ch], qps_w[ch].scale, qp_in.scale)
            for ch in range(w.shape[0])
        ]

        # quantize multipliers
        multipliers = [qp.scale * qp_in.scale / qp_out.scale for qp in qps_w]
        self.quant_multipliers = [quantize_multiplier(m) for m in multipliers]

        # initialize accumulators
        self.accumulators = [
            -qp_in.zero_point * np.sum(self.weights[ch], dtype=np.int32)
            + self.biases[ch]
            for ch in range(w.shape[0])
        ]

        self.qp_out = qp_out

    def __call__(self, input_tensor: np.ndarray) -> np.ndarray:
        channels = self.weights.shape[0]
        height, width = input_tensor.shape
        height -= 2
        width -= 2
        output = np.zeros((channels, height, width), dtype=np.int8)
        for ch in range(channels):
            n, m0 = self.quant_multipliers[ch]
            for row in range(height):
                for col in range(width):
                    x = input_tensor[row : row + 3, col : col + 3]
                    y = (
                        np.sum(
                            np.multiply(self.weights[ch], x, dtype=np.int16),
                            dtype=np.int32,
                        )
                        + self.accumulators[ch]
                    )
                    y = multiply_by_quantized_multiplier(y, n, m0)
                    y = y + self.qp_out.zero_point
                    y = np.clip(y, self.qp_out.q_min, self.qp_out.q_max).astype(np.int8)
                    output[ch, row, col] = y
        return output


class QuantizedLinear:
    def __init__(
        self,
        w: np.ndarray,
        b: np.ndarray,
        qp_in: QuantizationParameters,
        qp_out: QuantizationParameters,
    ):
        # quantize parameters
        self.weights, qp_w = quantize_weights_per_tensor(w)
        self.biases = [
            quantize_bias(b[c], qp_w.scale, qp_in.scale) for c in range(w.shape[0])
        ]

        # quantize multiplier
        multiplier = qp_w.scale * qp_in.scale / qp_out.scale
        self.n, self.m0 = quantize_multiplier(multiplier)

        # initialize accumulators
        self.accumulators = [
            -qp_in.zero_point * np.sum(self.weights[c], dtype=np.int32) + self.biases[c]
            for c in range(w.shape[0])
        ]

        self.qp_out = qp_out
        self.num_classes = w.shape[0]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        output = np.zeros((self.num_classes), dtype=np.int8)
        for c in range(self.num_classes):
            y = (
                np.sum(np.multiply(self.weights[c], x, dtype=np.int16), dtype=np.int32)
                + self.accumulators[c]
            )
            y = multiply_by_quantized_multiplier(y, self.n, self.m0)
            y = y + self.qp_out.zero_point
            y = np.clip(y, self.qp_out.q_min, self.qp_out.q_max).astype(np.int8)
            output[c] = y
        return output


class SimpleNetQuantized:
    def __init__(self, parameters, qparams_input, qparams_conv_relu, qparams_fc):
        self.qparams_input = qparams_input
        self.qparams_output = qparams_fc
        self.conv_relu = QuantizedConvReLU(
            parameters[0], parameters[1], qparams_input, qparams_conv_relu
        )
        self.fc = QuantizedLinear(
            parameters[2], parameters[3], qparams_conv_relu, qparams_fc
        )

    def maxpool(self, x: np.ndarray) -> np.ndarray:
        channels, height, width = x.shape
        height = height // 2
        width = width // 2
        output = np.zeros((channels, height, width), np.int8)
        for c in range(channels):
            for h in range(height):
                for w in range(width):
                    square_2x2 = x[c, h * 2 : h * 2 + 2, w * 2 : w * 2 + 2]
                    output[c, h, w] = np.max(square_2x2)
        return output

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y = self.conv_relu(x)
        y = self.maxpool(y)
        y = y.flatten()
        y = self.fc(y)
        return y


def get_quantized_model_manual(fp32_model):
    # Calibrate SimpleNet
    # create SimpleNet model with the same weights and with observers
    fp32_model_with_observers = SimpleNetWithObservers()
    fp32_model_with_observers.load_state_dict(fp32_model.state_dict())
    fp32_model_with_observers.eval()

    # calibrate
    train_dataset = torchvision.datasets.MNIST(
        "mnist",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    with torch.no_grad():
        for image_idx in range(500):
            fp32_model_with_observers(train_dataset[image_idx][0].unsqueeze(0))

    observations = fp32_model_with_observers.get_minmax_observations()

    # print min/max values
    print("min/max observations:")
    for key, val in observations.items():
        print(f"{key}: [{val.min}, {val.max}]")

    # quantize model
    # quantization parameters
    qp_input = compute_quantization_params(
        observations["input"].min,
        observations["input"].max,
        -128,
        127,
    )
    qp_conv_relu = compute_quantization_params(
        observations["conv_relu"].min,
        observations["conv_relu"].max,
        -128,
        127,
    )
    qp_fc = compute_quantization_params(
        observations["fc"].min,
        observations["fc"].max,
        -128,
        127,
    )

    # learned FP32-parameters
    fp32_model.eval()
    parameters = list(fp32_model.parameters())
    parameters_np = [p.detach().cpu().numpy() for p in parameters]

    # quantize
    quantized_model = SimpleNetQuantized(
        parameters_np,
        qp_input,
        qp_conv_relu,
        qp_fc,
    )
    return quantized_model
