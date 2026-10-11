"""
condlr.py — reference implementations for condition-generated low-rank transforms.

Contents
  BilinearCondIn / BilinearCondOut   exact full-hypernetwork path, chunked custom autograd;
                                     never stores U(c_t) / V(c_t) for the whole sequence
  ExactCondLowRank                   module: y = W x + x U(c_t) V(c_t)  (exact, per-token c)
  NaiveCondLowRank                   same function, materializes U/V — correctness oracle only
  BasisMixLowRank                    factorized hypernet (rank-K basis mixing), per-sample or per-token c
  DPLRCond                           diagonal + low-rank conditional transform (basis-mixed low-rank part)

Layout convention for the exact path:
  A : (d_in,  d_c * r)   with A[i, j*r + k]  ->  U(c)[i, k] = Σ_j A[i, j, k] c_j
  B : (d_c * r, d_out)   with B[j*r + k, o]  ->  V(c)[k, o] = Σ_j B[j, k, o] c_j

Run `python condlr.py` to execute the self-tests (gradcheck + oracle comparisons).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------------------
# Exact path: chunked bilinear contractions (inputs are 2-D: tokens flattened)
# ----------------------------------------------------------------------------------------

def _contraction_dtype(*tensors):
    device_type = tensors[0].device.type
    if torch.is_autocast_enabled(device_type):
        return torch.get_autocast_dtype(device_type)
    dtype = tensors[0].dtype
    for tensor in tensors[1:]:
        dtype = torch.promote_types(dtype, tensor.dtype)
    return dtype


def _accumulator_dtype(dtype):
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


class BilinearCondIn(torch.autograd.Function):
    """Exact input contraction; saves x, c, A, with chunk-bounded intermediates.

    Capture the forward compute dtype explicitly: backward generally runs outside
    autocast, and casting only activations there would mismatch fp32 parameters.
    Reductions and accumulation across chunks use at least fp32.
    """

    @staticmethod
    def forward(ctx, x, c, A, r: int, chunk: int = 1024):
        if chunk < 1 or r < 1:
            raise ValueError("chunk and rank must be positive")
        T, d_c = c.shape
        dtype = _contraction_dtype(x, c, A)
        acc_dtype = _accumulator_dtype(dtype)
        with torch.autocast(x.device.type, enabled=False):
            weight = A.to(dtype)
            z = torch.empty(T, r, device=x.device, dtype=dtype)
            for s in range(0, T, chunk):
                xs, cs = x[s:s + chunk].to(dtype), c[s:s + chunk].to(dtype)
                Y = (xs @ weight).view(-1, d_c, r)
                z[s:s + chunk] = (Y.to(acc_dtype) * cs.to(acc_dtype).unsqueeze(-1)).sum(1)
        ctx.save_for_backward(x, c, A)
        ctx.r, ctx.chunk, ctx.compute_dtype = r, chunk, dtype
        return z

    @staticmethod
    def backward(ctx, dz):
        x, c, A = ctx.saved_tensors
        r, chunk, dtype = ctx.r, ctx.chunk, ctx.compute_dtype
        acc_dtype = _accumulator_dtype(dtype)
        d_c = c.shape[1]
        dx, dcond = torch.empty_like(x), torch.empty_like(c)
        dA = torch.zeros_like(A, dtype=_accumulator_dtype(A.dtype))
        with torch.autocast(x.device.type, enabled=False):
            weight = A.to(dtype)
            for s in range(0, x.shape[0], chunk):
                xs, cs, dzs = (t[s:s + chunk].to(dtype) for t in (x, c, dz))
                G = (cs.unsqueeze(-1) * dzs.unsqueeze(1)).flatten(1)
                dx[s:s + chunk] = G @ weight.T
                dA.add_((xs.T @ G).to(dA.dtype))
                Y = (xs @ weight).view(-1, d_c, r)
                dcond[s:s + chunk] = (Y.to(acc_dtype) * dzs.to(acc_dtype).unsqueeze(1)).sum(-1)
        return dx, dcond, dA.to(A.dtype), None, None


class BilinearCondOut(torch.autograd.Function):
    """Exact output contraction; saves z, c, B and recomputes chunk tiles."""

    @staticmethod
    def forward(ctx, z, c, B, chunk: int = 1024):
        if chunk < 1:
            raise ValueError("chunk must be positive")
        dtype = _contraction_dtype(z, c, B)
        with torch.autocast(z.device.type, enabled=False):
            weight = B.to(dtype)
            y = torch.empty(z.shape[0], B.shape[1], device=z.device, dtype=dtype)
            for s in range(0, z.shape[0], chunk):
                zs, cs = z[s:s + chunk].to(dtype), c[s:s + chunk].to(dtype)
                G = (cs.unsqueeze(-1) * zs.unsqueeze(1)).flatten(1)
                y[s:s + chunk] = G @ weight
        ctx.save_for_backward(z, c, B)
        ctx.chunk, ctx.compute_dtype = chunk, dtype
        return y

    @staticmethod
    def backward(ctx, dy):
        z, c, B = ctx.saved_tensors
        chunk, dtype = ctx.chunk, ctx.compute_dtype
        acc_dtype = _accumulator_dtype(dtype)
        d_c, r = c.shape[1], z.shape[1]
        dz, dcond = torch.empty_like(z), torch.empty_like(c)
        dB = torch.zeros_like(B, dtype=_accumulator_dtype(B.dtype))
        with torch.autocast(z.device.type, enabled=False):
            weight = B.to(dtype)
            for s in range(0, z.shape[0], chunk):
                zs, cs, dys = (t[s:s + chunk].to(dtype) for t in (z, c, dy))
                G = (cs.unsqueeze(-1) * zs.unsqueeze(1)).flatten(1)
                dB.add_((G.T @ dys).to(dB.dtype))
                H = (dys @ weight.T).view(-1, d_c, r).to(acc_dtype)
                dz[s:s + chunk] = (H * cs.to(acc_dtype).unsqueeze(-1)).sum(1)
                dcond[s:s + chunk] = (H * zs.to(acc_dtype).unsqueeze(1)).sum(-1)
        return dz, dcond, dB.to(B.dtype), None


class ExactCondLowRank(nn.Module):
    """y = W x + x U(c_t) V(c_t), U/V linear in c (put an MLP before c if you want nonlinearity).
    Exact full hypernetwork; memory per token saved for backward: x, c, z (r floats)."""

    def __init__(self, d_in, d_out, d_cond, rank=2, chunk=1024):
        super().__init__()
        if rank < 1 or chunk < 1:
            raise ValueError("rank and chunk must be positive")
        self.r, self.chunk = rank, chunk
        self.base = nn.Linear(d_in, d_out)
        self.A = nn.Parameter(torch.randn(d_in, d_cond * rank) / math.sqrt(d_in * d_cond))
        self.B = nn.Parameter(torch.zeros(d_cond * rank, d_out))     # starts at base

    def forward(self, x, c):                       # x: (..., d_in), c: (..., d_cond)
        lead = x.shape[:-1]
        x2, c2 = x.reshape(-1, x.shape[-1]), c.reshape(-1, c.shape[-1])
        z = BilinearCondIn.apply(x2, c2, self.A, self.r, self.chunk)
        dy = BilinearCondOut.apply(z, c2, self.B, self.chunk)
        return self.base(x) + dy.view(*lead, -1)


class NaiveCondLowRank(nn.Module):
    """Same function as ExactCondLowRank but materializes U(c_t), V(c_t). Oracle only."""

    def __init__(self, exact: ExactCondLowRank):
        super().__init__()
        self.m = exact

    def forward(self, x, c):
        m = self.m
        d_in, d_out, d_c, r = x.shape[-1], m.B.shape[1], c.shape[-1], m.r
        U = torch.einsum("ijk,...j->...ik", m.A.view(d_in, d_c, r), c)    # (..., d_in, r)
        V = torch.einsum("jko,...j->...ko", m.B.view(d_c, r, d_out), c)   # (..., r, d_out)
        z = torch.einsum("...i,...ik->...k", x, U)
        return m.base(x) + torch.einsum("...k,...ko->...o", z, V)


# ----------------------------------------------------------------------------------------
# Factorized path: low-rank hypernetwork via basis mixing
# ----------------------------------------------------------------------------------------

class BasisMixLowRank(nn.Module):
    """y = W x + x U(c) V(c),  U(c)=Σ_k a_k(c) U_k,  V(c)=Σ_k b_k(c) V_k.

    Equivalent to a full hypernetwork whose projection W_u, W_v has rank n_basis.
    c may be (B, d_cond) [per-sample] or (B, T, d_cond) [per-token].
    mix='activations' : never builds U(c)            (per-token c, or short sequences)
    mix='weights'     : builds U(c), V(c) per sample (per-sample c, long sequences: K× fewer FLOPs)
    """

    def __init__(self, d_in, d_out, d_cond, rank=8, n_basis=16, mix="activations", gain=1.0,
                 factor_offsets=False):
        super().__init__()
        if rank < 1 or n_basis < 1:
            raise ValueError("rank and n_basis must be positive")
        if mix not in ("activations", "weights"):
            raise ValueError("mix must be activations or weights")
        self.K, self.r, self.mix = n_basis, rank, mix
        if not math.isfinite(gain) or gain <= 0:
            raise ValueError("gain must be finite and positive")
        self.gain = gain
        if type(factor_offsets) is not bool:
            raise ValueError("factor_offsets must be boolean")
        self.factor_offsets = factor_offsets
        self.base = nn.Linear(d_in, d_out)
        self.U = nn.Parameter(torch.randn(d_in, n_basis * rank) / math.sqrt(d_in))
        self.V = nn.Parameter(torch.zeros(n_basis * rank, d_out))
        self.to_coef = nn.Linear(d_cond, 2 * n_basis)
        self.U_offset = nn.Parameter(torch.zeros(d_in, rank)) if factor_offsets else None
        self.V_offset = nn.Parameter(torch.zeros(rank, d_out)) if factor_offsets else None

    def forward(self, x, c):                       # x: (B,T,d_in)
        return self.base(x) + self.delta(x, c)

    def delta(self, x, c):
        """Conditional update; activation mixing never builds per-token factors."""
        if c.dim() == 2:
            c = c.unsqueeze(1)                     # (B,1,d_cond) broadcasts over T
        a, b = self.to_coef(c).chunk(2, -1)        # (B,T|1,K)
        if self.mix == "weights":
            assert c.shape[1] == 1, "mix='weights' needs per-sample c"
            Uc = torch.einsum("bk,ikr->bir", a[:, 0], self.U.view(-1, self.K, self.r))
            Vc = torch.einsum("bk,kro->bro", b[:, 0], self.V.view(self.K, self.r, -1))
            if self.factor_offsets:
                Uc = Uc + self.U_offset.to(dtype=Uc.dtype)
                Vc = Vc + self.V_offset.to(dtype=Vc.dtype)
            delta = (x @ Uc) @ Vc
            return delta if self.gain == 1 else delta * self.gain
        h = (x @ self.U).unflatten(-1, (self.K, self.r))                 # (B,T,K,r)
        z = (h * a.unsqueeze(-1)).sum(-2)                                 # (B,T,r)  == x U(c)
        if self.factor_offsets:
            z = z + x @ self.U_offset
        g = (z.unsqueeze(-2) * b.unsqueeze(-1)).flatten(-2)               # (B,T,K*r)
        delta = g @ self.V
        if self.factor_offsets:
            delta = delta + z @ self.V_offset
        return delta if self.gain == 1 else delta * self.gain


class DPLRCond(nn.Module):
    """Diagonal + low-rank:  y = W(x ⊙ (1 + s(c))) + x U(c) V(c),  per-token or per-sample c."""

    def __init__(self, d_in, d_out, d_cond, rank=8, n_basis=16, shift=False, gain=1.0,
                 factor_offsets=False):
        super().__init__()
        self.lowrank = BasisMixLowRank(d_in, d_out, d_cond, rank, n_basis, gain=gain,
                                     factor_offsets=factor_offsets)
        self.to_scale = nn.Linear(d_cond, d_in)
        nn.init.zeros_(self.to_scale.weight)
        nn.init.zeros_(self.to_scale.bias)
        self.to_shift = nn.Linear(d_cond, d_in) if shift else None
        if self.to_shift is not None:
            nn.init.zeros_(self.to_shift.weight)
            nn.init.zeros_(self.to_shift.bias)
        # Note: only V and the scale head are zero-initialized. Zeroing to_coef as well
        # would leave both U-side and V-side with zero gradient forever.

    def forward(self, x, c):
        lr = self.lowrank
        cc = c.unsqueeze(1) if c.dim() == 2 else c
        s = self.to_scale(cc)
        scaled = x * (1 + s)
        if self.to_shift is not None:
            scaled = scaled + self.to_shift(cc)
        return lr.base(scaled) + lr.delta(x, c)


# ----------------------------------------------------------------------------------------
# Self-tests
# ----------------------------------------------------------------------------------------

def _tests():
    torch.manual_seed(0)
    dt = torch.float64

    # 1) gradcheck of the two custom Functions (chunk smaller than T to exercise the loop)
    T, d_in, d_c, r, d_out = 7, 5, 4, 3, 6
    x = torch.randn(T, d_in, dtype=dt, requires_grad=True)
    c = torch.randn(T, d_c, dtype=dt, requires_grad=True)
    A = torch.randn(d_in, d_c * r, dtype=dt, requires_grad=True)
    z = torch.randn(T, r, dtype=dt, requires_grad=True)
    B = torch.randn(d_c * r, d_out, dtype=dt, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x, c, A: BilinearCondIn.apply(x, c, A, r, 3), (x, c, A))
    assert torch.autograd.gradcheck(lambda z, c, B: BilinearCondOut.apply(z, c, B, 3), (z, c, B))

    # 2) exact module vs. materializing oracle (values + all grads), random nonzero B
    m = ExactCondLowRank(d_in, d_out, d_c, rank=r, chunk=4).to(dt)
    with torch.no_grad():
        m.B.normal_()
    naive = NaiveCondLowRank(m)
    xb = torch.randn(2, 5, d_in, dtype=dt, requires_grad=True)
    cb = torch.randn(2, 5, d_c, dtype=dt, requires_grad=True)
    y1 = m(xb, cb); g1 = torch.autograd.grad(y1.square().sum(), [xb, cb, m.A, m.B])
    y2 = naive(xb, cb); g2 = torch.autograd.grad(y2.square().sum(), [xb, cb, m.A, m.B])
    assert torch.allclose(y1, y2)
    assert all(torch.allclose(a, b) for a, b in zip(g1, g2))

    # 3) basis mixing: activation-mixing == weight-mixing for per-sample c, and == explicit U(c)V(c)
    bm = BasisMixLowRank(d_in, d_out, d_c, rank=r, n_basis=4).to(dt)
    with torch.no_grad():
        bm.V.normal_()
    xs, cs = torch.randn(3, 6, d_in, dtype=dt), torch.randn(3, d_c, dtype=dt)
    ya = bm(xs, cs)
    bm.mix = "weights"; yw = bm(xs, cs); bm.mix = "activations"
    assert torch.allclose(ya, yw)
    a, b = bm.to_coef(cs).chunk(2, -1)
    Uc = torch.einsum("bk,ikr->bir", a, bm.U.view(d_in, 4, r))
    Vc = torch.einsum("bk,kro->bro", b, bm.V.view(4, r, d_out))
    assert torch.allclose(ya, bm.base(xs) + xs @ Uc @ Vc)

    # 4) per-token c runs; DPLR reduces to base at init
    ct = torch.randn(3, 6, d_c, dtype=dt)
    assert bm(xs, ct).shape == (3, 6, d_out)
    dp = DPLRCond(d_in, d_out, d_c, rank=r, n_basis=4).to(dt)
    assert torch.allclose(dp(xs, ct), dp.lowrank.base(xs))

    print("all tests passed")


if __name__ == "__main__":
    _tests()
