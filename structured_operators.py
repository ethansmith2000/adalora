"""Broadcast conditional square operators; no dense matrix in the Hadamard path.

The learned conditional Hadamard chain is inspired by SHGΠHB in Fastfood
(https://proceedings.mlr.press/v28/le13.pdf). It is not a random-feature kernel
approximation: diagonals are generated from c, H is orthonormal, G starts at
one, and the output diagonal starts at zero for a zero residual.
"""
import math

import torch
from torch import nn

from dit_cifar10 import BroadcastLowRank


OPERATORS = ("none", "lowrank", "diagonal", "block", "hadamard", "tt")


def validate_operator(kind, width, block_size, *, variant, adapter_input,
                      factor_offsets, projector_groups, depth, decoder_groups,
                      decoder_cp_width, decoder_tucker_width, decoder_channel_mode,
                      coefficient_hidden, coefficient_activation, tt_modes="4,4,4,4", tt_bond_rank=6):
    if kind not in OPERATORS or type(block_size) is not int or block_size < 1:
        raise ValueError("Invalid square operator or block size")
    if kind == "none":
        return
    if (variant != "dplr" or adapter_input != "normalized" or not factor_offsets
            or projector_groups not in (0, depth) or decoder_groups
            or decoder_cp_width or decoder_tucker_width or decoder_channel_mode != "none"
            or coefficient_hidden or coefficient_activation != "linear"):
        raise ValueError("Square operators require independent linear DPLR heads, offsets, normalized input and no other decoder modes")
    if kind == "tt":
        from conditional_tt import tt_spec
        tt_spec(width, tt_modes, tt_bond_rank)
    if kind == "block" and width % block_size:
        raise ValueError("Block size must divide model width")
    if kind == "hadamard" and (width < 2 or width & (width - 1)):
        raise ValueError("Hadamard width must be a power of two, at least two")


def hadamard(x):
    """Orthonormal Walsh-Hadamard on the last axis, FP32 butterfly accumulation.

    Static power-of-two width; eager PyTorch/compile reference, not a claim
    about fused kernels or measured speed. No [d,d] tensor is constructed.
    """
    d = x.shape[-1]
    if d < 2 or d & (d-1):
        raise ValueError("Hadamard width must be a power of two, at least two")
    dtype = x.dtype
    y = x.float() if dtype in (torch.bfloat16, torch.float16) else x
    stride = 1
    while stride < d:
        pairs = y.reshape(*y.shape[:-1], d // (2*stride), 2, stride)
        left, right = pairs.unbind(-2)
        y = torch.stack((left + right, left - right), dim=-2).flatten(-3)
        stride *= 2
    return (y / math.sqrt(d)).to(dtype)


class SquareLowRank(BroadcastLowRank):
    is_square = True


class ConditionalStructured(nn.Module):
    is_square = True
    variant = "dplr"

    def __init__(self, width, condition_width, n_basis, kind, block_size,
                 gain, initialization_seed):
        super().__init__()
        if kind not in ("diagonal", "block", "hadamard") or min(width, condition_width, n_basis) < 1:
            raise ValueError("Invalid conditional operator")
        if kind == "block" and (block_size < 1 or width % block_size):
            raise ValueError("Block size must divide width")
        if kind == "hadamard" and (width < 2 or width & (width-1)):
            raise ValueError("Hadamard width must be a power of two")
        if not math.isfinite(gain) or gain <= 0:
            raise ValueError("Operator gain must be finite and positive")
        self.kind, self.d_in, self.d_out = kind, width, width
        self.d_condition, self.n_basis, self.block_size = condition_width, n_basis, block_size
        self.gain, self.factor_offsets = float(gain), True
        self.rank = width  # Compatibility field; explicit rank bound in model statistics.
        count = 3 if kind == "hadamard" else 1
        self.coefficients = nn.Linear(condition_width, count*n_basis)
        size = width*block_size if kind == "block" else width
        self.output_basis = nn.Parameter(torch.zeros(n_basis, size))
        self.output_offset = nn.Parameter(torch.zeros(size))
        if kind == "hadamard":
            self.input_basis = nn.Parameter(torch.zeros(n_basis, width))
            self.input_offset = nn.Parameter(torch.zeros(width))
            self.middle_basis = nn.Parameter(torch.zeros(n_basis, width))
            self.middle_offset = nn.Parameter(torch.zeros(width))
            generator = torch.Generator().manual_seed(initialization_seed)
            self.register_buffer("signs", (torch.randint(2, (width,), generator=generator)*2-1).float())
            self.register_buffer("permutation", torch.randperm(width, generator=generator))

    def decoder_named_parameters(self):
        names = ("output_basis", "output_offset")
        if self.kind == "hadamard":
            names += ("input_basis", "input_offset", "middle_basis", "middle_offset")
        return [(name, getattr(self, name)) for name in names]

    def values(self, condition):
        if condition.ndim != 2 or condition.shape[-1] != self.d_condition:
            raise ValueError("Expected broadcast condition [B,q]")
        coefficients = self.coefficients(condition)
        if self.kind == "hadamard":
            a, b, c = coefficients.chunk(3, -1)
            left = a @ self.input_basis
            middle = b @ self.middle_basis
            out = c @ self.output_basis
            return ((1 + left + self.input_offset.to(left.dtype))*self.signs.to(left.dtype),
                    1 + middle + self.middle_offset.to(middle.dtype),
                    out + self.output_offset.to(out.dtype))
        out = coefficients @ self.output_basis
        return out + self.output_offset.to(out.dtype)

    def forward(self, x, condition, factors=None):
        if x.ndim != 3 or x.shape[0] != condition.shape[0] or x.shape[-1] != self.d_in:
            raise ValueError("Expected x=[B,T,d] and condition=[B,q]")
        if factors is not None:
            raise ValueError("Shared factor caching is not implemented for these operators")
        values = self.values(condition)
        if self.kind == "hadamard":
            left, middle, out = values
            y = hadamard(x.to(left.dtype) * left[:, None])
            y = y.index_select(-1, self.permutation) * middle[:, None]
            delta = hadamard(y) * out[:, None]
        elif self.kind == "diagonal":
            delta = x.to(values.dtype) * values[:, None]
        else:
            b, t, d = x.shape
            matrices = values.reshape(b, d//self.block_size, self.block_size, self.block_size)
            grouped = x.reshape(b, t, d//self.block_size, self.block_size).transpose(1, 2)
            delta = torch.matmul(grouped, matrices).transpose(1, 2).flatten(2)
        return delta if self.gain == 1 else delta*self.gain

    def forward_matmul_flops(self, tokens):
        d, q, s = self.d_in, self.d_condition, self.n_basis
        if self.kind == "hadamard":
            return 6*s*(q+d)
        return 2*q*s + 2*s*d*(self.block_size if self.kind == "block" else 1) + (2*tokens*d*self.block_size if self.kind == "block" else 0)

    @torch.no_grad()
    def operator_diagnostics(self, condition):
        identity = torch.eye(self.d_in, device=condition.device, dtype=condition.dtype)
        matrix = self(identity[None].expand(condition.shape[0], -1, -1), condition)
        result = dict(operator=self.kind, transform_frobenius_mean=matrix.float().square().sum((-2,-1)).sqrt().mean().item(),
                      parameter_rms={n:p.float().square().mean().sqrt().item() for n,p in self.decoder_named_parameters()},
                      matrix_materialization_scope="untimed diagnostic only; square feature units")
        if self.kind == "hadamard":
            left, middle, out = self.values(condition)
            result.update(input_diagonal_rms=left.float().square().mean().sqrt().item(),
                          middle_diagonal_rms=middle.float().square().mean().sqrt().item(),
                          output_diagonal_rms=out.float().square().mean().sqrt().item())
        return result
