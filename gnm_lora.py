"""LoRA for the frozen Walker used as the GNM encoder (local GPU / small boxes).

The GNM encoder reads Walker's own P blocks (+ one R pass); the design doc's remaining lever is a
small adapter on Walker's early blocks so the owner/label signal becomes linearly available
(docs/GNM.md section 3). Full fine-tuning needs the whole 1B and its optimizer state; a LoRA adds
only A/B low-rank factors (zero-init B => start as an exact no-op) to the attention/MLP projections
of a block range, and leaves every base weight frozen.

    n = attach_lora(model, blocks=range(0, model.r_end), r=16)
    params = lora_parameters(model)          # only these get gradient

Works on both the trainer's QLinear (quantized master weight) and plain nn.Linear.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def _io_dims(module: nn.Module) -> tuple[int, int]:
    """(in_features, out_features) for QLinear (master), nn.Linear (weight) or a bare module."""
    if hasattr(module, "master"):                       # trainer QLinear
        return int(module.master.shape[1]), int(module.master.shape[0])
    if type(module).__name__ == "Conv1D" and hasattr(module, "weight"):   # HF GPT-2: (in, out)
        return int(module.weight.shape[0]), int(module.weight.shape[1])
    if hasattr(module, "weight") and module.weight is not None:
        return int(module.weight.shape[1]), int(module.weight.shape[0])
    if hasattr(module, "in_features"):
        return int(module.in_features), int(module.out_features)
    raise TypeError(f"cannot infer in/out dims for {type(module).__name__}")


class LoRALinear(nn.Module):
    """Frozen base layer + a trainable low-rank update: base(x) + scale * (x @ A^T) @ B^T.

    `B` is zero-initialized so the adapter is an exact no-op at attach time (the encoder's output is
    bitwise the frozen walker until training moves B away from zero)."""

    def __init__(self, base: nn.Module, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        d_in, d_out = _io_dims(base)
        ref = next(base.parameters(), None)
        device = ref.device if ref is not None else None
        self.r = int(r)
        self.alpha = float(alpha)
        self.scale = self.alpha / max(1, self.r)
        self.A = nn.Parameter(torch.empty(self.r, d_in, device=device))
        self.B = nn.Parameter(torch.zeros(d_out, self.r, device=device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        out = self.base(x)
        dtype, device = out.dtype, out.device
        a = self.A.to(device=device, dtype=dtype)
        b = self.B.to(device=device, dtype=dtype)
        rank = (x.to(device=device, dtype=dtype) @ a.t()) @ b.t()
        return out + self.scale * rank

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha:g}, scale={self.scale:g}"


DEFAULT_TARGETS = ("attn.qkv", "attn.proj", "mlp.up", "mlp.down", "mlp.gate")


def attach_lora(model: nn.Module, blocks=None, r: int = 8, alpha: float = 16.0,
                targets=DEFAULT_TARGETS) -> int:
    """Wrap the named projection submodules inside `model.blocks[i]` (i in `blocks`; None = all) with
    LoRALinear. `targets` are EXACT dotted paths within the block (attention qkv/proj, FFN up/down/gate)
    so walk/sidecar submodules (e.g. walk.gate_mlp) are never touched. Returns the wrapped count."""
    wrapped = 0
    block_list = list(range(len(model.blocks))) if blocks is None else list(blocks)
    for index in block_list:
        block = model.blocks[index]
        for name, child in list(block.named_modules()):
            if name not in targets:
                continue
            if not (isinstance(child, nn.Linear) or hasattr(child, "master")
                    or (hasattr(child, "weight") and getattr(child.weight, "ndim", 0) == 2)):
                continue
            parent = block
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], LoRALinear(child, r=r, alpha=alpha))
            wrapped += 1
    return wrapped


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for name, p in model.named_parameters()
            if p.requires_grad and (name.endswith(".A") or name.endswith(".B"))]


def mark_only_lora_trainable(model: nn.Module) -> None:
    """Freeze everything, then re-enable only LoRA A/B (safety when the base was not frozen)."""
    for p in model.parameters():
        p.requires_grad_(False)
    for name, p in model.named_parameters():
        if name.endswith(".A") or name.endswith(".B"):
            p.requires_grad_(True)
