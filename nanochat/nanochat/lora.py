"""Lightweight LoRA injection utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List

import torch
import torch.nn as nn


@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: int = 16
    dropout: float = 0.0
    target_modules: Iterable[str] = (
        "c_q",
        "c_k",
        "c_v",
        "c_proj",
        "c_fc",
        "c_proj",
    )


class LoRALinear(nn.Module):
    """Wrap a linear module with a trainable LoRA adapter."""

    def __init__(self, base: nn.Module, rank: int, alpha: int, dropout: float = 0.0) -> None:
        super().__init__()
        if not hasattr(base, "in_features") or not hasattr(base, "out_features"):
            raise TypeError("LoRA adapters require modules with in_features/out_features attributes")
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        in_features = int(base.in_features)
        out_features = int(base.out_features)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Linear(in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_features, bias=False)
        self.reset_parameters()
        for param in self.base.parameters():
            param.requires_grad = False

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_in = self.dropout(x)
        update = self.lora_b(self.lora_a(lora_in)) * self.scaling
        return base_out + update.to(base_out.dtype)


def _resolve_parent(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    tokens = module_name.split(".")
    parent = model
    for token in tokens[:-1]:
        parent = getattr(parent, token)
    return parent, tokens[-1]


def inject_lora(model: nn.Module, config: LoRAConfig) -> List[str]:
    """Replace targeted linear projections with LoRA wrapped versions."""

    applied: List[str] = []
    targets = set(config.target_modules)
    for name, module in list(model.named_modules()):
        if name.split(".")[-1] not in targets:
            continue
        parent, attr = _resolve_parent(model, name)
        current = getattr(parent, attr)
        if isinstance(current, LoRALinear):
            continue
        wrapped = LoRALinear(current, config.rank, config.alpha, config.dropout)
        setattr(parent, attr, wrapped)
        applied.append(name)
    if not applied:
        raise ValueError("No modules matched LoRA target list")
    return applied


def lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Collect all parameters that remain trainable after LoRA injection."""

    return [p for p in model.parameters() if p.requires_grad]


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract only the LoRA adapter weights for checkpointing."""

    state: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[f"{name}.lora_a.weight"] = module.lora_a.weight.detach().cpu()
            state[f"{name}.lora_b.weight"] = module.lora_b.weight.detach().cpu()
    return state

