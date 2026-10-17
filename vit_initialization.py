"""MLP-only normal initialization with explicit GELU and residual gain anchoring."""
from functools import lru_cache
import math

import numpy as np
import torch
from torch.nn import functional as F

MLP_INITIALIZERS = ('legacy', 'reference_fanin', 'fanin_gelu', 'xavier_gelu', 'mixed_gelu')


@lru_cache(maxsize=None)
def gelu_second_moment(std):
    """E[GELU(z)^2], z normal(0,std^2), deterministic 96-point quadrature."""
    nodes, weights = np.polynomial.hermite.hermgauss(96)
    z = torch.from_numpy(nodes * (math.sqrt(2) * std))
    return float(np.dot(weights, F.gelu(z).square().numpy()) / math.sqrt(math.pi))


def mlp_initialization_stds(dim, hidden, mode):
    if mode not in MLP_INITIALIZERS or dim < 1 or hidden < 1:
        raise ValueError('Invalid MLP initialization mode or dimensions')
    reference = 4 * dim
    if mode == 'legacy':return .02, .02
    if mode == 'reference_fanin':return .02, .02 * math.sqrt(reference / hidden)
    # The first projection has gain1, except the mixed recipe's explicit sqrt2
    # approximation before GELU. The second projection has an explicit
    # gain chosen to match the legacy narrow MLP's expected output second moment.
    def raw_stds(h):
        fanin = (1 / math.sqrt(dim), 1 / math.sqrt(h))
        xavier = (math.sqrt(2 / (dim + h)), math.sqrt(2 / (h + dim)))
        return fanin if mode == 'fanin_gelu' else xavier if mode == 'xavier_gelu' else (math.sqrt(2)*fanin[0], xavier[1])
    ref_in, ref_out = raw_stds(reference)
    target = reference * .02**2 * gelu_second_moment(.02 * math.sqrt(dim))
    raw_reference = reference * ref_out**2 * gelu_second_moment(ref_in * math.sqrt(dim))
    gain_out = math.sqrt(target / raw_reference)
    actual_in, actual_out = raw_stds(hidden)
    return actual_in, actual_out * gain_out


@torch.no_grad()
def initialize_mlp(mlp, dim, mode):
    """Rescale the existing legacy normal draws; consumes no random numbers.

    The original truncated normal uses absolute bounds ±2 at std0.02; these
    are ±100 std. Rescaling retains that truncation and preserves every draw.
    Fan-aware modes implement the stated std rules rather than replacing draws.
    """
    hidden = mlp.fc1.out_features
    std_in, std_out = mlp_initialization_stds(dim, hidden, mode)
    mlp.fc1.weight.mul_(std_in / .02)
    mlp.fc2.weight.mul_(std_out / .02)
