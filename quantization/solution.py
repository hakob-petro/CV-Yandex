from typing import List, Tuple
import random

import numpy as np
import torch


# Task 1 (1 point)
class QuantizationParameters:
    def __init__(
        self,
        scale: np.float64,
        zero_point: np.int32,
        q_min: np.int32,
        q_max: np.int32,
    ):
        self.scale = scale
        self.zero_point = zero_point
        self.q_min = q_min
        self.q_max = q_max

    def __repr__(self):
        return f"scale: {self.scale}, zero_point: {self.zero_point}"


def compute_quantization_params(
    r_min: np.float32,
    r_max: np.float32,
    q_min: np.int32,
    q_max: np.int32,
) -> QuantizationParameters:
    # your code goes here \/
    scale = (r_max - r_min) / (q_max - q_min)
    zero_point = np.round((r_max * q_min - r_min * q_max) / (r_max - r_min))
    return QuantizationParameters(scale.astype(np.float64), zero_point.astype(np.int32), q_min, q_max)
    # your code goes here /\


# Task 2 (0.5 + 0.5 = 1 point)
def quantize(r: np.ndarray, qp: QuantizationParameters) -> np.ndarray:
    # your code goes here \/
    return np.clip(np.round(r / qp.scale + qp.zero_point), qp.q_min, qp.q_max).astype(np.int8)
    # your code goes here /\


def dequantize(q: np.ndarray, qp: QuantizationParameters) -> np.ndarray:
    # your code goes here \/
    return (qp.scale * (q.astype(np.int32) - qp.zero_point)).astype(np.float32)
    # your code goes here /\


# Task 3 (1 point)
class MinMaxObserver:
    def __init__(self):
        self.min = np.finfo(np.float32).max
        self.max = np.finfo(np.float32).min

    def __call__(self, x: torch.Tensor):
        # your code goes here \/
        curr_min = torch.min(x.detach()).numpy().astype(np.float32)
        curr_max = torch.max(x.detach()).numpy().astype(np.float32)
        if curr_min < self.min:
            self.min = curr_min
        if curr_max > self.max:
            self.max = curr_max
        # your code goes here /\


# Task 4 (1 + 1 = 2 points)
def quantize_weights_per_tensor(
    weights: np.ndarray,
) -> Tuple[np.array, QuantizationParameters]:
    # your code goes here \/
    weights_min = weights.min()
    weights_max = weights.max()

    weights_min = -weights_max if np.abs(weights_min) < np.abs(weights_max) else weights_min
    weights_max = -weights_min if np.abs(weights_min) >= np.abs(weights_max) else weights_max

    qp = compute_quantization_params(weights_min.astype(np.float32), weights_max.astype(np.float32), np.int32(-127), np.int32(127))
    q = quantize(weights, qp)
    return q, qp
    # your code goes here /\


def quantize_weights_per_channel(
    weights: np.ndarray,
) -> Tuple[np.array, List[QuantizationParameters]]:
    # your code goes here \/

    weights_min = weights.min(axis=(2, 3))
    weights_max = weights.max(axis=(2, 3))

    weights_min = np.where(np.abs(weights_min) < np.abs(weights_max), -weights_max, weights_min).astype(np.float32)
    weights_max = np.where(np.abs(weights_min) >= np.abs(weights_max), -weights_min, weights_max).astype(np.float32)

    qps = []
    for i, channel in enumerate(weights):
        qp = compute_quantization_params(weights_min[i], weights_max[i], np.int32(-127), np.int32(127))
        qps.append(qp)
        if i == 0:
            q = quantize(channel, qp)
        else:
            q = np.stack([q, quantize(channel, qp)], axis=0).astype(np.int8)

    return q, qps
    # your code goes here /\


# Task 5 (1 point)
def quantize_bias(
    bias: np.float32,
    scale_w: np.float64,
    scale_x: np.float64,
) -> np.int32:
    # your code goes here \/
    scale = scale_w * scale_x
    return np.clip(np.round(bias / scale), np.iinfo(np.int32).min, np.iinfo(np.int32).max).astype(np.int32)
    # your code goes here /\


# Task 6 (2 points)
def quantize_multiplier(m: np.float64) -> [np.int32, np.int32]:
    # your code goes here \/
    n = np.ceil(np.log2(0.5 / m)).astype(np.int32)
    m_0 = m * np.power(2, n) if n >= 0 else m / np.power(2, -n)
    m_0 = np.int32(np.round(m_0 * 2 ** 31))
    return n, m_0
    # your code goes here /\


# Task 7 (2 points)
def multiply_by_quantized_multiplier(
    accum: np.int32,
    n: np.int32,
    m0: np.int32,
) -> np.int32:
    # your code goes here \/
    res = np.multiply(accum, m0, dtype=np.int64)
    point_pos_from_left = 33 - n
    point_pos_from_right = 64 - point_pos_from_left

    bin_res = np.binary_repr(res, 64)
    after_point = bin_res[point_pos_from_left]

    res = np.int32(res >> point_pos_from_right)
    res += np.int32(after_point)
    return res
    # your code goes here /\
