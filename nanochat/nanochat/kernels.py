"""Custom kernels and quantization utilities for nanochat."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Tuple

import torch
from torch import Tensor, nn
from torch.autograd import Function

_LOGGED_KERNEL_USAGE = False


def _log_kernel_once(message: str) -> None:
    """Emit a single structured log for kernel activation."""
    global _LOGGED_KERNEL_USAGE
    if _LOGGED_KERNEL_USAGE:
        return
    _LOGGED_KERNEL_USAGE = True
    payload = {
        "filename": __name__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "classname": "TernaryLinear",
        "function": "forward",
        "system_section": "model.quantization",
        "line_num": 0,
        "error": False,
        "db_phase": "none",
        "method": "NONE",
        "message": message,
    }
    print(json.dumps(payload, sort_keys=True))
    print("[Continuous skepticism (Sherlock Protocol)]", message)


@dataclass
class TernaryStats:
    """Statistics returned by the ternary quantizer."""

    weight: Tensor
    mask: Tensor
    scaling: Tensor


def ternary_quantize(weight: Tensor, threshold: float) -> TernaryStats:
    """Quantize the given weight matrix to ternary values {-alpha, 0, +alpha}."""
    if threshold <= 0:
        raise ValueError("Ternary threshold must be positive")
    weight_abs = weight.abs()
    delta = threshold * weight_abs.mean(dim=1, keepdim=True)
    mask = (weight_abs > delta).to(weight.dtype)
    masked_abs = weight_abs * mask
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    scaling = masked_abs.sum(dim=1, keepdim=True) / denom
    scaling = torch.nan_to_num(scaling, nan=0.0, posinf=0.0, neginf=0.0)
    ternary = torch.sign(weight) * mask
    quantized = scaling * ternary
    return TernaryStats(weight=quantized, mask=mask, scaling=scaling)


class _TernaryLinearKernel(Function):
    """Autograd kernel for ternary linear layers with straight-through gradients."""

    @staticmethod
    def forward(ctx, input: Tensor, weight: Tensor, threshold: float) -> Tensor:
        stats = ternary_quantize(weight, threshold)
        ctx.save_for_backward(input, weight, stats.mask)
        ctx.threshold = threshold
        _log_kernel_once("Activated ternary linear kernel")
        output = input.matmul(stats.weight.t())
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, Tensor, None]:
        input, weight, mask = ctx.saved_tensors
        stats = ternary_quantize(weight, ctx.threshold)
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
        input_2d = input.reshape(-1, input.shape[-1])
        grad_input = grad_output_2d.matmul(stats.weight)
        grad_input = grad_input.view_as(input)
        grad_weight = grad_output_2d.t().matmul(input_2d)
        grad_weight = grad_weight * mask
        return grad_input, grad_weight, None


def ternary_linear_kernel(input: Tensor, weight: Tensor, threshold: float) -> Tensor:
    """Functional API for the ternary linear kernel."""
    return _TernaryLinearKernel.apply(input, weight, threshold)


class TernaryLinear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` that uses ternary weights."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False, threshold: float = 0.7) -> None:
        super().__init__()
        if bias:
            raise ValueError("TernaryLinear does not support bias to match nanochat defaults")
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = float(threshold)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input: Tensor) -> Tensor:
        return ternary_linear_kernel(input, self.weight, self.threshold)


__all__ = ["TernaryLinear", "ternary_linear_kernel", "ternary_quantize", "TernaryStats"]
