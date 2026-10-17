"""Small pixel-space DiT with sample-wise class/timestep conditioning.

AdaLN-Zero block layout follows the public DiT architecture. Extra transforms
are added only at MLP input; all variants keep the same AdaLN heads and gates.
This is a CIFAR-10 epsilon predictor, not a pretrained or latent-space DiT.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


VARIANTS = ("adaln_zero", "static_lora", "exact", "dplr", "conditional_bottleneck",
            "conditional_core", "conditional_dplr_core")
ADAPTER_INPUTS = ("normalized", "modulated")
COEFFICIENT_ACTIVATIONS = ("linear", "silu")
CHANNEL_MODES = ("none", "static", "modulated")


class BroadcastLowRank(nn.Module):
    """Generate factors once per sample, then reuse them across image tokens."""

    def __init__(self, d_in, d_out, d_condition, variant, rank=8, n_basis=16, gain=1.0, core_rank=2,
                 coefficient_activation="linear", coefficient_hidden=0, factor_offsets=False):
        super().__init__()
        if variant not in VARIANTS[1:] or min(rank, n_basis) < 1:
            raise ValueError("Invalid adapter variant or dimensions")
        self.variant, self.rank, self.n_basis = variant, rank, n_basis
        if coefficient_activation not in COEFFICIENT_ACTIVATIONS or (variant != "dplr" and coefficient_activation != "linear"):
            raise ValueError("Coefficient activation requires a valid DPLR head")
        self.coefficient_activation = coefficient_activation
        if (not isinstance(coefficient_hidden, int) or isinstance(coefficient_hidden, bool)
                or coefficient_hidden < 0 or (variant != "dplr" and coefficient_hidden)):
            raise ValueError("Coefficient hidden width requires DPLR and a nonnegative integer")
        self.coefficient_hidden = coefficient_hidden
        if type(factor_offsets) is not bool or (factor_offsets and variant != "dplr"):
            raise ValueError("Factor offsets require DPLR and a boolean setting")
        self.factor_offsets = factor_offsets
        if core_rank < 1 or (variant == "conditional_dplr_core" and core_rank > rank):
            raise ValueError("Core factor rank must be positive and no larger than the outer rank")
        self.core_rank = core_rank
        self.d_in, self.d_out, self.d_condition = d_in, d_out, d_condition
        if not math.isfinite(gain) or gain <= 0:
            raise ValueError("Adapter gain must be finite and positive")
        self.gain = float(gain)  # fixed, nonzero; identity comes from the zero factor/head
        if variant == "exact":
            self.A = nn.Parameter(torch.randn(d_in, d_condition * rank) / math.sqrt(d_in * d_condition))
            self.B = nn.Parameter(torch.zeros(d_condition * rank, d_out))
        elif variant in ("conditional_bottleneck", "conditional_core", "conditional_dplr_core"):
            self.U = nn.Parameter(torch.randn(d_in, rank) / math.sqrt(d_in))
            self.V = nn.Parameter(torch.randn(rank, d_out) / math.sqrt(rank))
            if variant == "conditional_core":
                self.core = nn.Linear(d_condition, rank * rank)
                nn.init.zeros_(self.core.weight)
                nn.init.zeros_(self.core.bias)
            else:
                self.gains = nn.Linear(d_condition, rank)
                nn.init.zeros_(self.gains.weight)
                nn.init.zeros_(self.gains.bias)
                if variant == "conditional_dplr_core":
                    self.core_left = nn.Linear(d_condition, rank * core_rank)
                    self.core_right = nn.Linear(d_condition, core_rank * rank)
                    # Nonzero L + zero R opens R gradients; both-zero L/R would stall.
                    nn.init.zeros_(self.core_right.weight)
                    nn.init.zeros_(self.core_right.bias)
        else:
            size = rank if variant == "static_lora" else n_basis * rank
            self.U = nn.Parameter(torch.randn(d_in, size) / math.sqrt(d_in))
            self.V = nn.Parameter(torch.zeros(size, d_out))
            if variant == "dplr":
                if coefficient_hidden:
                    self.coefficients = nn.Sequential(
                        nn.Linear(d_condition, coefficient_hidden),
                        nn.SiLU() if coefficient_activation == "silu" else nn.Identity(),
                        nn.Linear(coefficient_hidden, 2 * n_basis))
                else:
                    self.coefficients = nn.Linear(d_condition, 2 * n_basis)
                self.U_offset = nn.Parameter(torch.zeros(d_in, rank)) if factor_offsets else None
                self.V_offset = nn.Parameter(torch.zeros(rank, d_out)) if factor_offsets else None

    def coefficient_values(self, condition):
        values = self.coefficients(condition)
        return F.silu(values) if self.coefficient_activation == "silu" and not self.coefficient_hidden else values

    def decoder_named_parameters(self):
        names = ("U", "V", "U_offset", "V_offset") if self.factor_offsets else ("U", "V")
        return [(name, getattr(self, name)) for name in names]

    def factors(self, condition):
        if condition.ndim != 2 or condition.shape[-1] != self.d_condition:
            raise ValueError("Expected one condition vector per sample: (B, d_condition)")
        if self.variant == "exact":
            u = torch.einsum("idr,bd->bir", self.A.view(self.d_in, self.d_condition, self.rank), condition)
            v = torch.einsum("dro,bd->bro", self.B.view(self.d_condition, self.rank, self.d_out), condition)
        elif self.variant == "dplr":
            a, b = self.coefficient_values(condition).chunk(2, dim=-1)
            u = torch.einsum("bk,ikr->bir", a, self.U.view(self.d_in, self.n_basis, self.rank))
            v = torch.einsum("bk,kro->bro", b, self.V.view(self.n_basis, self.rank, self.d_out))
            if self.factor_offsets:
                u = u + self.U_offset.to(dtype=u.dtype)
                v = v + self.V_offset.to(dtype=v.dtype)
        else:
            raise ValueError("This adapter does not generate sample-dependent outer factors")
        return u, v

    def core_components(self, condition):
        """Keep leading sample/token axes; do not materialize a DPLR core matrix."""
        if self.variant == "conditional_core":
            return self.core(condition).unflatten(-1, (self.rank, self.rank))
        if self.variant == "conditional_dplr_core":
            return (self.gains(condition),
                    self.core_left(condition).unflatten(-1, (self.rank, self.core_rank)),
                    self.core_right(condition).unflatten(-1, (self.core_rank, self.rank)))
        raise ValueError("Only conditional-core adapters have core components")

    def forward(self, x, condition, factors=None):
        if x.ndim != 3 or condition.ndim != 2 or x.shape[0] != condition.shape[0]:
            raise ValueError("Expected x=(B,T,d_in) and condition=(B,d_condition)")
        if self.variant == "static_lora":
            delta = (x @ self.U) @ self.V
        elif self.variant == "conditional_bottleneck":
            delta = ((x @ self.U) * self.gains(condition).unsqueeze(1)) @ self.V
        elif self.variant == "conditional_core":
            delta = torch.bmm(x @ self.U, self.core_components(condition)) @ self.V
        elif self.variant == "conditional_dplr_core":
            gains, left, right = self.core_components(condition)
            z = x @ self.U
            mixed = z * gains.unsqueeze(1) + torch.bmm(torch.bmm(z, left), right)
            delta = mixed @ self.V
        else:
            u, v = self.factors(condition) if factors is None else factors
            delta = torch.bmm(torch.bmm(x, u), v)
        return delta if self.gain == 1.0 else delta * self.gain

    def forward_matmul_flops(self, tokens):
        d, o, c, r, k = self.d_in, self.d_out, self.d_condition, self.rank, self.n_basis
        apply = 2 * tokens * r * (d + o)
        if self.variant == "static_lora":
            return apply
        if self.variant == "conditional_bottleneck":
            return apply + 2 * c * r
        if self.variant == "conditional_core":
            return apply + 2 * c * r * r + 2 * tokens * r * r
        if self.variant == "conditional_dplr_core":
            p = self.core_rank
            return apply + 2 * c * (r + 2 * r * p) + 4 * tokens * r * p
        if self.variant == "exact":
            return apply + 2 * c * r * (d + o)
        head = 2 * self.coefficient_hidden * (c + 2*k) if self.coefficient_hidden else 4*c*k
        return apply + head + 2 * k * r * (d + o)


class CPBroadcastLowRank(BroadcastLowRank):
    """CP-compressed factor decoders; transform rank remains `rank`.

    U[i,s,k] = sum_p U_channel[i,p] * U_basis[s,p] * U_rank[p,k].
    V[s,k,o] has the analogous three factors. Only sample-wise U/V are
    materialized during forward, never the dense decoder basis tensors.
    """

    def __init__(self, *args, cp_width, initialization_seed, **kwargs):
        if type(cp_width) is not int or cp_width < 1:
            raise ValueError("CP width must be a positive integer")
        super().__init__(*args, **kwargs)
        if self.variant != "dplr":
            raise ValueError("CP decoders require DPLR")
        self.cp_width = cp_width
        # Reuse the dense initialization's head and RNG trajectory. These two
        # temporary CPU tensors are discarded before moving the model to CUDA.
        del self.U, self.V
        generator = torch.Generator().manual_seed(initialization_seed)
        rand = lambda *shape: torch.randn(*shape, generator=generator)
        p, s, r, d, o = cp_width, self.n_basis, self.rank, self.d_in, self.d_out
        self.U_channel = nn.Parameter(rand(d, p) / math.sqrt(d))
        self.U_basis = nn.Parameter(rand(s, p))
        self.U_rank = nn.Parameter(rand(p, r) / math.sqrt(p))
        self.V_basis = nn.Parameter(rand(s, p))
        self.V_rank = nn.Parameter(rand(p, r))
        self.V_channel = nn.Parameter(torch.zeros(p, o))
        # E[U[i,s,k]^2]=1/d under independent initialization draws, matching
        # dense U's entry variance. Zero V_channel preserves the zero delta;
        # its two companion factors are nonzero so it can receive gradients.

    def decoder_named_parameters(self):
        names = ("U_channel", "U_basis", "U_rank", "V_basis", "V_rank", "V_channel")
        if self.factor_offsets:
            names += ("U_offset", "V_offset")
        return [(name, getattr(self, name)) for name in names]

    def factors(self, condition):
        if condition.ndim != 2 or condition.shape[-1] != self.d_condition:
            raise ValueError("Expected one condition vector per sample: (B, d_condition)")
        a, b = self.coefficient_values(condition).chunk(2, dim=-1)
        u_coeff = a @ self.U_basis
        v_coeff = b @ self.V_basis
        u_weights = u_coeff.unsqueeze(-1) * self.U_rank.to(dtype=u_coeff.dtype)
        v_weights = v_coeff.unsqueeze(-1) * self.V_rank.to(dtype=v_coeff.dtype)
        u = torch.matmul(self.U_channel, u_weights)
        v = torch.matmul(v_weights.transpose(-1, -2), self.V_channel)
        if self.factor_offsets:
            u = u + self.U_offset.to(dtype=u.dtype)
            v = v + self.V_offset.to(dtype=v.dtype)
        return u, v

    def forward_matmul_flops(self, tokens):
        d, o, c, r, s, p = self.d_in, self.d_out, self.d_condition, self.rank, self.n_basis, self.cp_width
        head = 2 * self.coefficient_hidden * (c + 2*s) if self.coefficient_hidden else 4*c*s
        return 2*tokens*r*(d+o) + head + 4*s*p + 2*p*r*(d+o)


class TuckerBroadcastLowRank(CPBroadcastLowRank):
    """Factor only the channel mode; learn unrestricted [s,p,r] cores.

    Cores start as expanded CP products to pair initial channel bases and
    condition-to-factor functions with CP (up to arithmetic reassociation).
    This is a Tucker-style channel-mode factorization, not a conditional r×r
    core between static outer factors.
    """

    def __init__(self, *args, tucker_width, initialization_seed, **kwargs):
        super().__init__(*args, cp_width=tucker_width, initialization_seed=initialization_seed, **kwargs)
        self.tucker_width = tucker_width
        self.U_core = nn.Parameter((self.U_basis.unsqueeze(-1) * self.U_rank).detach())
        self.V_core = nn.Parameter((self.V_basis.unsqueeze(-1) * self.V_rank).detach())
        del self.U_basis, self.U_rank, self.V_basis, self.V_rank, self.cp_width

    def decoder_named_parameters(self):
        names = ("U_channel", "U_core", "V_core", "V_channel")
        if self.factor_offsets:
            names += ("U_offset", "V_offset")
        return [(name, getattr(self, name)) for name in names]

    def factors(self, condition):
        if condition.ndim != 2 or condition.shape[-1] != self.d_condition:
            raise ValueError("Expected one condition vector per sample: (B, d_condition)")
        a, b = self.coefficient_values(condition).chunk(2, dim=-1)
        u_weights = (a @ self.U_core.flatten(1)).unflatten(-1, (self.tucker_width, self.rank))
        v_weights = (b @ self.V_core.flatten(1)).unflatten(-1, (self.tucker_width, self.rank))
        u = torch.matmul(self.U_channel, u_weights)
        v = torch.matmul(v_weights.transpose(-1, -2), self.V_channel)
        if self.factor_offsets:
            u = u + self.U_offset.to(dtype=u.dtype)
            v = v + self.V_offset.to(dtype=v.dtype)
        return u, v

    def forward_matmul_flops(self, tokens):
        d, o, c, r, s, p = self.d_in, self.d_out, self.d_condition, self.rank, self.n_basis, self.tucker_width
        head = 2*self.coefficient_hidden*(c+2*s) if self.coefficient_hidden else 4*c*s
        return 2*tokens*r*(d+o) + head + 4*s*p*r + 2*p*r*(d+o)


class ChannelModulatedLowRank(BroadcastLowRank):
    """Condition-driven row/column scales around static rank-r factors.

    Static control uses the same factor/offset slots but removes all unused
    condition-head parameters. The model's common AdaLN remains conditional.
    """

    def __init__(self, *args, channel_mode, initialization_seed, **kwargs):
        if channel_mode not in ("static", "modulated"):
            raise ValueError("Expected static or modulated channel mode")
        super().__init__(*args, **kwargs)
        if self.variant != "dplr":
            raise ValueError("Channel factors require DPLR")
        self.channel_mode = channel_mode
        del self.U, self.V
        generator = torch.Generator().manual_seed(initialization_seed)
        self.U_static = nn.Parameter(torch.randn(self.d_in,self.rank,generator=generator)/math.sqrt(self.d_in))
        self.V_static = nn.Parameter(torch.zeros(self.rank,self.d_out))
        if channel_mode == "modulated":
            self.U_scale_basis = nn.Parameter(torch.zeros(self.n_basis,self.d_in))
            self.V_scale_basis = nn.Parameter(torch.zeros(self.n_basis,self.d_out))
        else:
            del self.coefficients

    def decoder_named_parameters(self):
        names = ("U_static", "V_static")
        if self.channel_mode == "modulated":
            names += ("U_scale_basis", "V_scale_basis")
        if self.factor_offsets:
            names += ("U_offset", "V_offset")
        return [(name,getattr(self,name)) for name in names]

    def channel_scales(self, condition):
        if self.channel_mode != "modulated":
            raise ValueError("Static control has no condition heads")
        a,b = self.coefficient_values(condition).chunk(2,-1)
        return 1 + a @ self.U_scale_basis, 1 + b @ self.V_scale_basis

    def factors(self, condition):
        if condition.ndim != 2 or condition.shape[-1] != self.d_condition:
            raise ValueError("Expected one condition vector per sample: (B, d_condition)")
        if self.channel_mode == "modulated":
            gu,gv = self.channel_scales(condition)
            u = self.U_static.to(dtype=gu.dtype).unsqueeze(0) * gu.unsqueeze(-1)
            v = self.V_static.to(dtype=gv.dtype).unsqueeze(0) * gv.unsqueeze(1)
        else:
            dtype = torch.get_autocast_dtype(condition.device.type) if torch.is_autocast_enabled(condition.device.type) else condition.dtype
            u = self.U_static.to(dtype=dtype).unsqueeze(0).expand(condition.shape[0],-1,-1)
            v = self.V_static.to(dtype=dtype).unsqueeze(0).expand(condition.shape[0],-1,-1)
        if self.factor_offsets:
            u = u + self.U_offset.to(dtype=u.dtype)
            v = v + self.V_offset.to(dtype=v.dtype)
        return u,v

    def forward_matmul_flops(self, tokens):
        d,o,c,r,s = self.d_in,self.d_out,self.d_condition,self.rank,self.n_basis
        apply = 2*tokens*r*(d+o)
        if self.channel_mode == "static":return apply
        head = 2*self.coefficient_hidden*(c+2*s) if self.coefficient_hidden else 4*c*s
        return apply + head + 2*s*(d+o)


def sinusoidal(values, width):
    half = width // 2
    frequency = torch.exp(-math.log(10000) * torch.arange(half, device=values.device, dtype=torch.float32) / half)
    phase = values.float().unsqueeze(-1) * frequency
    return torch.cat((phase.cos(), phase.sin()), dim=-1)


class DiTBlock(nn.Module):
    def __init__(self, width, heads, mlp_ratio, adapter_input="normalized"):
        super().__init__()
        if adapter_input not in ADAPTER_INPUTS:
            raise ValueError("Invalid adapter input")
        self.adapter_input = adapter_input
        self.heads = heads
        self.norm_attention = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.norm_mlp = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(width, 3 * width)
        self.attention_out = nn.Linear(width, width)
        self.mlp_in = nn.Linear(width, int(width * mlp_ratio))
        self.mlp_out = nn.Linear(int(width * mlp_ratio), width)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
        self.adapter = None
        # Layer-local observation point even when several blocks use one adapter.
        self.adapter_delta = nn.Identity()

    def forward(self, tokens, condition, adapter_factors=None):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(condition).unsqueeze(1).chunk(6, dim=-1)
        h = self.norm_attention(tokens) * (1 + scale_a) + shift_a
        b, t, d = h.shape
        q, k, v = self.qkv(h).reshape(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        attention = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, t, d)
        tokens = tokens + gate_a * self.attention_out(attention)
        h = self.norm_mlp(tokens)
        h_mod = h * (1 + scale_m) + shift_m
        projected = self.mlp_in(h_mod)
        if self.adapter is not None:
            adapter_input = h if self.adapter_input == "normalized" else h_mod
            projected = projected + self.adapter_delta(self.adapter(adapter_input, condition, adapter_factors))
        return tokens + gate_m * self.mlp_out(F.gelu(projected, approximate="tanh"))


class CIFAR10DiT(nn.Module):
    def __init__(self, width=192, depth=6, heads=3, patch_size=4, mlp_ratio=4,
                 variant="adaln_zero", rank=8, n_basis=16, image_size=32,
                 adapter_input="normalized", adapter_gain=1.0, core_rank=2,
                 projector_groups=0, coefficient_activation="linear", decoder_groups=0,
                 coefficient_hidden=0, enable_unconditional=False, factor_offsets=False, decoder_cp_width=0,
                 decoder_tucker_width=0, decoder_channel_mode="none"):
        super().__init__()
        if variant not in VARIANTS or adapter_input not in ADAPTER_INPUTS or width % heads or width % 4 or image_size % patch_size:
            raise ValueError("Invalid DiT variant or dimensions")
        self.width, self.patch_size, self.image_size = width, patch_size, image_size
        self.num_classes = 10
        self.enable_unconditional = bool(enable_unconditional)
        self.variant, self.depth, self.mlp_ratio = variant, depth, mlp_ratio
        self.core_rank = core_rank
        self.adapter_input = adapter_input
        if (not isinstance(projector_groups, int) or isinstance(projector_groups, bool)
                or projector_groups < 0 or projector_groups > depth
                or (projector_groups and depth % projector_groups)
                or coefficient_activation not in COEFFICIENT_ACTIVATIONS
                or (variant != "dplr" and (projector_groups or coefficient_activation != "linear"))):
            raise ValueError("Projector groups must divide depth and shared/nonlinear heads require DPLR")
        self.projector_groups = (projector_groups or depth) if variant != "adaln_zero" else 0
        self.coefficient_activation = coefficient_activation
        if (not isinstance(decoder_groups, int) or isinstance(decoder_groups, bool)
                or decoder_groups < 0 or decoder_groups > depth
                or (decoder_groups and depth % decoder_groups)
                or (decoder_groups and projector_groups not in (0, depth))
                or not isinstance(coefficient_hidden, int) or isinstance(coefficient_hidden, bool)
                or coefficient_hidden < 0
                or (variant != "dplr" and (decoder_groups or coefficient_hidden))):
            raise ValueError("Decoder groups must divide depth with independent heads; hidden heads require DPLR")
        self.decoder_groups = decoder_groups
        self.coefficient_hidden = coefficient_hidden
        if type(factor_offsets) is not bool or (factor_offsets and variant != "dplr"):
            raise ValueError("Factor offsets require DPLR and a boolean setting")
        self.factor_offsets = factor_offsets
        if (type(decoder_cp_width) is not int or decoder_cp_width < 0
                or (decoder_cp_width and (variant != "dplr" or decoder_groups))):
            raise ValueError("CP decoder width requires DPLR without decoder sharing and a nonnegative integer")
        self.decoder_cp_width = decoder_cp_width
        if (type(decoder_tucker_width) is not int or decoder_tucker_width < 0
                or (decoder_tucker_width and (variant != "dplr" or decoder_groups or decoder_cp_width))):
            raise ValueError("Tucker decoder width requires DPLR, no decoder sharing/CP width, and a nonnegative integer")
        self.decoder_tucker_width = decoder_tucker_width
        if (decoder_channel_mode not in CHANNEL_MODES or
                (decoder_channel_mode != "none" and (variant != "dplr" or decoder_groups or decoder_cp_width or decoder_tucker_width))):
            raise ValueError("Channel mode requires DPLR without other decoder parameterizations or sharing")
        self.decoder_channel_mode = decoder_channel_mode
        if not math.isfinite(adapter_gain) or adapter_gain <= 0:
            raise ValueError("Adapter gain must be finite and positive")
        self.adapter_gain = float(adapter_gain)
        self.time_embedding = nn.Sequential(nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width))
        self.class_embedding = nn.Embedding(10, width)
        self.condition_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.patch_embedding = nn.Conv2d(3, width, patch_size, stride=patch_size)
        side = image_size // patch_size
        yy, xx = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
        position = torch.cat((sinusoidal(yy.flatten(), width // 2), sinusoidal(xx.flatten(), width // 2)), dim=-1)
        self.register_buffer("position", position.unsqueeze(0))
        self.blocks = nn.ModuleList(DiTBlock(width, heads, mlp_ratio, adapter_input) for _ in range(depth))
        self.final_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.output = nn.Linear(width, patch_size * patch_size * 3)
        self.apply(self._initialize)
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        nn.init.normal_(self.class_embedding.weight, std=0.02)
        for layer in (self.time_embedding[0], self.time_embedding[2]):
            nn.init.normal_(layer.weight, std=0.02)
        for layer in [b.modulation[-1] for b in self.blocks] + [self.final_modulation[-1], self.output]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        # Preserve identical base, AdaLN and embedding initialization across variants.
        if variant != "adaln_zero":
            adapter_class = ChannelModulatedLowRank if decoder_channel_mode != "none" else TuckerBroadcastLowRank if decoder_tucker_width else CPBroadcastLowRank if decoder_cp_width else BroadcastLowRank
            structured_width = decoder_tucker_width or decoder_cp_width
            width_key = "tucker_width" if decoder_tucker_width else "cp_width"
            adapters = [adapter_class(width, int(width * mlp_ratio), width, variant, rank, n_basis,
                        adapter_gain, core_rank, coefficient_activation, coefficient_hidden, factor_offsets,
                        **({"channel_mode": decoder_channel_mode,"initialization_seed": torch.initial_seed()+100000+i}
                           if decoder_channel_mode != "none" else {width_key: structured_width,"initialization_seed": torch.initial_seed()+100000+i}
                           if structured_width else {})) for i in range(self.projector_groups)]
            for index, block in enumerate(self.blocks):
                # References to one module per group; named_parameters deduplicates aliases.
                block.adapter = adapters[index // (depth // self.projector_groups)]
            if decoder_groups:
                for index, block in enumerate(self.blocks):
                    source = adapters[(index // (depth // decoder_groups)) * (depth // decoder_groups)]
                    block.adapter.U, block.adapter.V = source.U, source.V
                    if factor_offsets:
                        block.adapter.U_offset, block.adapter.V_offset = source.U_offset, source.V_offset
        # Create the null token last so enabling CFG preserves every existing tensor.
        if self.enable_unconditional:
            generator = torch.Generator().manual_seed(torch.initial_seed() + 40000)
            self.null_class_embedding = nn.Parameter(torch.randn(width, generator=generator) * 0.02)

    def grouped_adapters(self):
        groups = []
        for index, block in enumerate(self.blocks):
            if block.adapter is None:
                continue
            if not groups or groups[-1][1] is not block.adapter:
                groups.append(([index], block.adapter))
            else:
                groups[-1][0].append(index)
        return groups

    def grouped_decoders(self):
        """Describe shared U/V tensors separately from complete adapter modules."""
        groups = []
        for index, block in enumerate(self.blocks):
            adapter = block.adapter
            if adapter is None or adapter.variant != "dplr":
                continue
            identity = tuple(id(p) for _, p in adapter.decoder_named_parameters())
            if groups and tuple(id(p) for _, p in groups[-1][1].decoder_named_parameters()) == identity:
                groups[-1][0].append(index)
            else:
                groups.append(([index], adapter))
        return groups

    @staticmethod
    def _initialize(layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.Conv2d):
            nn.init.zeros_(layer.bias)

    def forward(self, images, timesteps, class_labels):
        b = images.shape[0]
        timesteps = torch.as_tensor(timesteps, device=images.device).expand(b)
        if class_labels.shape != (b,) or images.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError("Invalid image or class-label shape")
        if self.enable_unconditional:
            null_mask = class_labels == self.num_classes
            class_condition = torch.where(null_mask.unsqueeze(-1),
                                          self.null_class_embedding,
                                          self.class_embedding(class_labels.masked_fill(null_mask,0)))
        else:
            class_condition = self.class_embedding(class_labels)
        condition = self.condition_norm(self.time_embedding(sinusoidal(timesteps, 256)) + class_condition)
        tokens = self.patch_embedding(images).flatten(2).transpose(1, 2) + self.position
        shared = self.projector_groups and self.projector_groups < self.depth
        factors = [adapter.factors(condition) for _, adapter in self.grouped_adapters()] if shared else None
        for index, block in enumerate(self.blocks):
            pair = factors[index // (self.depth // self.projector_groups)] if shared else None
            tokens = block(tokens, condition, pair)
        shift, scale = self.final_modulation(condition).unsqueeze(1).chunk(2, dim=-1)
        patches = self.output(self.final_norm(tokens) * (1 + scale) + shift)
        p, side = self.patch_size, self.image_size // self.patch_size
        return patches.reshape(b, side, side, p, p, 3).permute(0, 5, 1, 3, 2, 4).reshape(b, 3, self.image_size, self.image_size)

    def statistics(self):
        d, t, h = self.width, (self.image_size // self.patch_size) ** 2, int(self.width * self.mlp_ratio)
        adapters = [adapter for _, adapter in self.grouped_adapters()]
        unique_parameters = {id(p):p for a in adapters for p in a.parameters()}
        adapter_params = sum(p.numel() for p in unique_parameters.values())
        base_flops = (2 * t * 3 * self.patch_size**2 * d + 2 * 256 * d + 2 * d * d
                      + self.depth * (8 * t * d * d + 4 * t * t * d + 4 * t * d * h + 12 * d * d)
                      + 4 * d * d + 2 * t * d * self.patch_size**2 * 3)
        apply = 2 * t * adapters[0].rank * (d+h) if adapters else 0
        adapter_flops = sum(a.forward_matmul_flops(t) for a in adapters) + (self.depth-len(adapters))*apply if adapters else 0
        stats = {"parameters": sum(p.numel() for p in self.parameters()), "adapter_parameters": adapter_params,
                "tokens_per_image": t, "forward_matmul_flops_per_image_estimate": base_flops + adapter_flops,
                "flop_estimate_excludes": "elementwise operations, reductions, normalization and backward",
                "adapter_input": self.adapter_input,
                "adapter_gain": self.adapter_gain,
                "core_factor_rank": self.core_rank if self.variant == "conditional_dplr_core" else None,
                "projector_groups": self.projector_groups,
                "projector_group_layers": [indices for indices, _ in self.grouped_adapters()],
                "coefficient_activation": self.coefficient_activation,
                "coefficient_hidden": self.coefficient_hidden,
                "factor_offsets": self.factor_offsets,
                "decoder_groups": len(self.grouped_decoders()),
                "decoder_group_layers": [indices for indices, _ in self.grouped_decoders()],
                "adapter_parameter_tensors": len(unique_parameters),
                "unconditional_class_embedding": self.enable_unconditional,
                "coefficient_parameters": sum(p.numel() for a in adapters if hasattr(a,"coefficients") for p in a.coefficients.parameters()) if self.variant == "dplr" else None,
                "decoder_parameters": sum(p.numel() for _,a in self.grouped_decoders()
                                          for _,p in a.decoder_named_parameters()) if self.variant == "dplr" else None,
                "factor_generation": ("static factors; rank gains generated once per sample" if self.variant == "conditional_bottleneck"
                                      else "static outer factors; dense core generated once per sample" if self.variant == "conditional_core"
                                      else "static outer factors; diagonal and low-rank core components generated once per sample" if self.variant == "conditional_dplr_core"
                                      else "once per sample and projector group; reused across group layers and image tokens" if self.projector_groups < self.depth
                                      else "once per sample; factors reused across image tokens"),
                "conditioning": "normalized learned class-plus-timestep vector; common AdaLN heads and residual gates"}
        if self.decoder_cp_width:
            stats.update(decoder_parameterization="cp", decoder_cp_width=self.decoder_cp_width,
                         decoder_dense_bases_materialized=False)
        if self.decoder_tucker_width:
            stats.update(decoder_parameterization="tucker_channel", decoder_tucker_width=self.decoder_tucker_width,
                         decoder_dense_bases_materialized=False)
        if self.decoder_channel_mode != "none":
            stats.update(decoder_parameterization="channel_"+self.decoder_channel_mode,
                         decoder_channel_mode=self.decoder_channel_mode,decoder_dense_bases_materialized=False)
            if self.decoder_channel_mode == "static":
                stats["factor_generation"] = "static factors and offsets; common AdaLN remains conditional"
        return stats
