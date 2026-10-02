"""
Shared int8 quantization utilities used by export and Python/C++ parity checks.

Quantization matches export_weights.py and the C++/ESP32 runtimes:
  scale = 127 / max(abs(weights))
  q = round(weights * scale) clipped to [-128, 127]
  dequant = q / scale
"""

from __future__ import annotations

import torch

DEFAULT_SCALE_FACTOR = 127.0


def quantize_weights(weights: torch.Tensor, scale_factor: float = DEFAULT_SCALE_FACTOR):
    """
    Quantize FP32 weights to int8.

    Returns:
        (quantized_int8_numpy, scale) where scale is a Python float.
    """
    w_max = torch.abs(weights).max().item()
    scale = scale_factor / w_max if w_max > 0 else 1.0
    quantized = torch.clamp(torch.round(weights * scale), -128, 127).to(torch.int8)
    return quantized.numpy(), float(scale)


def dequantize_weights(quantized, scale: float) -> torch.Tensor:
    """Dequantize int8 weights back to FP32."""
    if not torch.is_tensor(quantized):
        quantized = torch.from_numpy(quantized)
    scale_val = scale if abs(scale) > 1e-9 else 1.0
    return quantized.to(torch.float32) / scale_val


def quantize_linear(layer: torch.nn.Linear):
    """Return (weight_q, weight_scale, bias_q|None, bias_scale|None) for a Linear layer."""
    weight_q, weight_scale = quantize_weights(layer.weight.data)
    if layer.bias is not None:
        bias_q, bias_scale = quantize_weights(layer.bias.data)
    else:
        bias_q, bias_scale = None, 1.0
    return weight_q, weight_scale, bias_q, bias_scale


def quantize_embedding(layer: torch.nn.Embedding):
    weight_q, weight_scale = quantize_weights(layer.weight.data)
    return weight_q, weight_scale


def quantize_layer_norm(layer: torch.nn.LayerNorm):
    weight_q, weight_scale = quantize_weights(layer.weight.data)
    bias_q, bias_scale = quantize_weights(layer.bias.data)
    return weight_q, weight_scale, bias_q, bias_scale
