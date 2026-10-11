"""Conditional linear projections used by the CIFAR-10 ViT experiments."""

import math

import torch
from torch import nn

from condlr import BasisMixLowRank, DPLRCond, ExactCondLowRank


VARIANTS = ("plain", "diagonal", "exact", "basis", "dplr", "static_lora_diagonal")


class ConditionedLinear(nn.Module):
    """Replace a projection while preserving its base weights and bias.

    Diagonal variants use input scale *and* shift. DPLR retains the reference
    layer's additive low-rank branch and adds the same input shift as diagonal.
    Every conditional branch starts at zero; coefficient heads stay nonzero.
    """

    def __init__(self, base, variant, rank=2, n_basis=16, chunk=2048, gain=1.0,
                 factor_offsets=False):
        super().__init__()
        if variant not in VARIANTS or variant == "plain":
            raise ValueError(f"Unsupported conditional variant: {variant}")
        if rank < 1 or n_basis < 1 or chunk < 1:
            raise ValueError("rank, n_basis and chunk must be positive")
        self.variant = variant
        if type(factor_offsets) is not bool or (factor_offsets and variant not in ("basis", "dplr")):
            raise ValueError("Factor offsets require a basis/DPLR variant and a boolean setting")
        self.factor_offsets = factor_offsets
        if not math.isfinite(gain) or gain <= 0:
            raise ValueError("gain must be finite and positive")
        if variant in ("exact", "diagonal") and gain != 1:
            raise ValueError("Non-unit gain is supported for basis/DPLR/static low-rank updates")
        self.gain = gain
        self.d_in, self.d_out = base.in_features, base.out_features
        self.rank, self.n_basis = rank, n_basis
        d = self.d_in
        if variant == "exact":
            self.transform = ExactCondLowRank(d, self.d_out, d, rank, chunk)
            self.transform.base = base
        elif variant in ("basis", "dplr"):
            cls = DPLRCond if variant == "dplr" else BasisMixLowRank
            self.transform = cls(d, self.d_out, d, rank, n_basis, shift=True, gain=gain,
                                 factor_offsets=factor_offsets) if variant == "dplr" else cls(
                                     d, self.d_out, d, rank, n_basis, gain=gain, factor_offsets=factor_offsets)
            if variant == "dplr":
                self.transform.lowrank.base = base
            else:
                self.transform.base = base
        else:
            self.base = base
            self.to_affine = nn.Linear(d, 2 * d)
            nn.init.zeros_(self.to_affine.weight)
            nn.init.zeros_(self.to_affine.bias)
            if variant == "static_lora_diagonal":
                self.U = nn.Parameter(torch.randn(d, rank) / math.sqrt(d))
                self.V = nn.Parameter(torch.zeros(rank, self.d_out))

    def forward(self, x, condition):
        if self.variant in ("exact", "basis", "dplr"):
            return self.transform(x, condition)
        scale, shift = self.to_affine(condition).chunk(2, dim=-1)
        y = self.base(x * (1 + scale) + shift)
        if self.variant == "static_lora_diagonal":
            y = y + self.gain * ((x @ self.U) @ self.V)
        return y

    def additional_matmul_flops_per_token(self):
        """Leading matrix-multiply FLOPs, excluding elementwise ops/reductions."""
        d, o, r, k = self.d_in, self.d_out, self.rank, self.n_basis
        if self.variant == "exact":
            return 2 * d * r * (d + o)
        basis = 4 * d * k + 2 * k * r * (d + o)
        if self.factor_offsets:
            basis += 2 * r * (d + o)
        if self.variant == "basis":
            return basis
        if self.variant == "dplr":
            return basis + 4 * d * d
        diagonal = 4 * d * d
        return diagonal + (2 * r * (d + o) if self.variant == "static_lora_diagonal" else 0)


def project(layer, x, condition):
    return layer(x, condition) if isinstance(layer, ConditionedLinear) else layer(x)
