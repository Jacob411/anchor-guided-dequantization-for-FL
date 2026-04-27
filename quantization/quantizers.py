"""
Quantizers for uniform symmetric quantization.
Supported: INT2, INT4, INT8, FP16 (passthrough with dtype cast).
"""
from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Tuple


BITS_PER_ELEMENT = {
    2:  2,
    3:  3,
    4:  4,
    6:  6,
    8:  8,
    16: 16,
    32: 32,   # full precision sentinel
}


@dataclass
class QuantizationResult:
    quantized: torch.Tensor   # quantized tensor (int or fp16)
    scale: float              # scale factor used
    bits: int                 # bit-width
    original_shape: Tuple     # shape before any flattening


class Quantizer:
    """
    Uniform symmetric quantizer with per-tensor scaling.

    Usage:
        q = Quantizer(bits=4)
        result = q.quantize(weight_delta)
        reconstructed = q.dequantize(result)
    """

    def __init__(self, bits: int) -> None:
        if bits not in BITS_PER_ELEMENT:
            raise ValueError(f"Unsupported bit-width: {bits}. Choose from {list(BITS_PER_ELEMENT)}")
        self.bits = bits

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def quantize(self, tensor: torch.Tensor) -> QuantizationResult:
        """Quantize a floating-point tensor."""
        if self.bits == 32:
            return QuantizationResult(
                quantized=tensor.clone(),
                scale=1.0,
                bits=32,
                original_shape=tuple(tensor.shape),
            )

        if self.bits == 16:
            return QuantizationResult(
                quantized=tensor.to(torch.float16),
                scale=1.0,
                bits=16,
                original_shape=tuple(tensor.shape),
            )

        # Integer quantization (2, 4, 8 bit)
        n_levels = 2 ** (self.bits - 1) - 1   # symmetric: [-n_levels, n_levels]
        abs_max = tensor.abs().max().item()
        scale = abs_max / n_levels if abs_max > 0 else 1.0

        q = torch.clamp(torch.round(tensor / scale), -n_levels, n_levels)
        return QuantizationResult(
            quantized=q.to(torch.int8 if self.bits <= 8 else torch.int16),
            scale=scale,
            bits=self.bits,
            original_shape=tuple(tensor.shape),
        )

    def dequantize(self, result: QuantizationResult) -> torch.Tensor:
        """Reconstruct a float32 tensor from a QuantizationResult."""
        if result.bits == 32:
            return result.quantized.float()
        if result.bits == 16:
            return result.quantized.float()
        return result.quantized.float() * result.scale

    # ------------------------------------------------------------------
    # Communication cost helpers
    # ------------------------------------------------------------------

    def comm_bytes(self, tensor: torch.Tensor) -> int:
        """Estimated uplink bytes to transmit the quantized tensor."""
        num_elements = tensor.numel()
        return (num_elements * BITS_PER_ELEMENT[self.bits]) // 8
