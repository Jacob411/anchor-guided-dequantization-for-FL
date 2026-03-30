"""
Constrained client — quantizes weight deltas before sending.
"""
from __future__ import annotations

import copy
from typing import Dict, Tuple

import torch

from clients.base_client import BaseClient
from quantization.quantizers import Quantizer, QuantizationResult


class ConstrainedClient(BaseClient):
    """
    Resource-constrained client that quantizes its weight delta
    to `bits` bit-width before uploading.

    The quantized delta is dequantized back to FP32 before returning,
    so the server receives a lossy-reconstructed FP32 tensor (as would
    happen after transmission and server-side decompression).
    """

    def __init__(self, bits: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.bits = bits
        self.quantizer = Quantizer(bits=bits)

    @property
    def is_anchor(self) -> bool:
        return False

    def train_one_round(
        self, global_state: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        Returns:
            delta      : {layer: FP32 tensor} — dequantized (lossy) weight delta
            comm_bytes : actual uplink cost after quantization
        """
        self.load_global_model(global_state)
        self._local_train()

        fp_delta = self._compute_delta(global_state)

        # Quantize each layer's delta independently
        reconstructed_delta: Dict[str, torch.Tensor] = {}
        total_comm_bytes = 0

        for layer_name, tensor in fp_delta.items():
            result: QuantizationResult = self.quantizer.quantize(tensor)
            reconstructed         = self.quantizer.dequantize(result)
            reconstructed_delta[layer_name] = reconstructed
            total_comm_bytes     += self.quantizer.comm_bytes(tensor)

        return reconstructed_delta, total_comm_bytes
