# Conditional transforms: design and optimization notes

Updated 2026-10-08 from the design discussion. This supplements the original
[research brief](RESEARCH_BRIEF.md); it distinguishes the tested implementation
from proposed comparisons. New explanations use PyTorch expressions and tensor
shapes because rendered equations are unreliable in the conversation viewer.

## Original setup context, reconciled 2026-10-07

The user supplied the broader conversation that led to the original brief and
`condlr.py`. Its reported starting point was a full condition-to-factor projection
with interesting earlier performance, but rank often limited to 2 by activation
memory, generator parameters, and speed. Those earlier results are motivation,
not additional audited results in this repository: their task/configuration and
metrics were not supplied in the conversation.

The original research questions remain:

- Factorize or sparsify the projection that generates transform parameters.
- Independently change the produced transform: low-rank, Kronecker, Hadamard/
  Fastfood-style, FFT/circulant, tensor-train, or a single block-sparse transform
  rather than a two-stage Monarch-style construction.
- Preserve an exact, memory-bounded reference to distinguish execution changes
  from changes to the function class.
- Prefer PyTorch/compile where it works well; justify custom Triton work using
  measured bottlenecks rather than assuming fusion is necessary.

The supplied conversation mostly explains the existing brief, rather than adding
an unseen task requirement. It fixes no dataset, width, or sequence-conditioning
benchmark. CIFAR-10 and the current model sizes are later prototype choices.
The subsequent local discussion adds explicit axes for adapter input, placement,
identity initialization, gain, learning rates, and factorization gauges.

Several original statements need narrower interpretation:

| Original statement | Reconciled interpretation |
| --- | --- |
| The layer is bilinear / second-order under self-conditioning | Each input contraction is bilinear in token and condition. The full two-condition-generated-factor delta has higher degree; see below. |
| Exact saved memory does not grow with rank | Full factors are not retained, but the rank bottleneck grows with r. Original rounded CPU figures hid that contribution. |
| Nothing factor-sized reaches HBM | Whole-sequence factors are avoided. Eager chunked PyTorch still writes transient tiles to device memory; stronger fusion is a kernel goal. |
| Compile cannot fuse this reduction | Fusion and memory scheduling depend on version, shape, backend, and settings. Inspect generated execution; do not treat the old prediction as a universal restriction. |
| Healthy GEMM shapes imply modest overhead | They are a useful layout starting point, not a roofline or speed result. Backward, launch overhead, traffic, and compilation need measurement. |
| Structured full-rank transforms cost about as much as AdaLN | Some generate O(d) parameters, but application, scratch, backward, and kernel costs differ. Compact generation is not equal wall-clock cost. |

The original CPU table was a saved-activation illustration, not whole-model GPU
peak memory. Local profiling later measured 6.03/6.13 MiB saved storage for exact
rank 2/8 on a 192-to-768 layer with 4,096 tokens, versus 36.31/127.25 MiB for the
materializing oracle; see the [engineering record](reports/cifar10-engineering.json)
and [README](README.md#first-engineering-screen). Shapes, storage accounting, and
source versions differ from the original CPU setup, so do not merge the numbers
into one interchangeable benchmark.

## Working model and independent axes

```text
condition c -> generator -> conditional parameters
token h + conditional parameters -> conditional channel transform
base projection + conditional delta -> activation -> output projection -> gate
```

There are separate choices here. A low-rank transform does not imply a cheap
generator, small activation storage, or fast execution.

| Axis | Tested diffusion choice | Proposed comparisons |
| --- | --- | --- |
| Condition | Normalized class-plus-timestep vector, broadcast per image | Per-token self-condition; sequence-valued context with explicit retrieval |
| Generator | Linear coefficient head plus learned factor bases | Conditional bottleneck gains; full hypernetwork; CP/Tucker; sparse or shared generator |
| Transform | Rank-r rectangular additive delta | Square additive diagonal + low-rank + shift; structured full-rank transforms |
| Adapter input | Unmodulated normalized token | Shared conditional scale/shift input; private static affine input |
| Placement | MLP input projection, before GELU | Square pre-MLP modulation; QKV; output projections |
| Application order | Parallel base and adapter branches | Residual composition, with ordering controlled explicitly |
| Initialization and gain | Zero output-side bases; nonzero coefficient head | Basis/rank scaling; alternative zeroing mechanism; factor normalization |
| Optimization | One adapter LR multiplier | Separate rates for input bases, output bases, and generator |
| Execution | Generate factors once per image and reuse | Activation mixing per token; chunked exact contraction; compilation/fusion |

`r` bounds the rank of the generated weight delta, not the base-plus-delta
projection. `K` is the basis count per factor generator; `s` below denotes a
generic generator bottleneck width. For the current separate factor generators,
`s=K`. Neither should be confused with model width or sequence length.

## Current implementation and normalization

The tested diffusion model has width 192, six blocks, three attention heads,
64 image tokens, and an MLP expansion from 192 to 768. The condition has width
192. The main DPLR configuration uses `r=8, K=16`.

Both pre-attention and pre-MLP normalizations are:

```python
nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
```

`norm_mlp` is the LayerNorm before the MLP, not another learned MLP. Conditional
heads supply the scale and shift, so omitting static LayerNorm affine parameters
is the clean default for this AdaLN recipe. The earlier self-conditioned ViT
retained ordinary affine LayerNorm; it also did not use diffusion's conditional
residual gates. These are different recipes, not interchangeable results.

The following is the current MLP part after the attention residual has been
applied; it uses actual module names from [dit_cifar10.py](dit_cifar10.py):

```python
import torch.nn.functional as F

def current_mlp_path(tokens, condition, block):
    # tokens: [B, T, d]; condition: [B, d]
    _, _, _, shift, scale, gate = (
        block.modulation(condition).unsqueeze(1).chunk(6, dim=-1)
    )                                      # each [B, 1, d]
    h = block.norm_mlp(tokens)             # [B, T, d]
    h_mod = h * (1 + scale) + shift
    base = block.mlp_in(h_mod)             # [B, T, o], o = 4*d
    delta = block.adapter(h, condition)    # [B, T, o], DPLR variant
    out = block.mlp_out(F.gelu(base + delta, approximate="tanh"))
    return tokens + gate * out
```

The adapter receives `h`, not `h_mod`. The delta is added before GELU. Base and
adapter then share the MLP output projection and residual gate. AdaLN supplies
the diagonal component; there is no duplicate diagonal head in the diffusion
adapter. The base projection is condition-independent but trainable, not frozen.
No full `[B, d, o]` weight delta is materialized.

## Adapter input and additive versus composed placement

Keep the same `c`, base weights, and common AdaLN heads when comparing inputs.
In the sketches below, modulation helpers return tensors already broadcastable
with `h`; the complete current-path example above shows the explicit sample axis.

```python
h = norm(x)
h_mod = h * (1 + scale(c)) + shift(c)
base = mlp_in(h_mod)

# Choose one adapter input:
adapter_input = h_mod                       # shared conditional modulation
adapter_input = h                           # current normalized split
adapter_input = h * (1 + static_scale) + static_shift
# static_scale/static_shift are learned [d] tensors, initially zero.

pre_act = base + lowrank(adapter_input, c)   # rectangular d -> o delta
```

These are alternatives, not three successive assignments in an implementation.
Private static affine calibration costs `2*d` parameters per block. Once it
learns, that adapter input is no longer strictly normalized. Using `h_mod` couples
adapter magnitude to both the AdaLN head and the factor generator. Using `h`
separates those sources of gain; optimization superiority is still a hypothesis.

A square additive modulation is another candidate:

```python
h_adapted = (
    h * (1 + scale(c))
    + lowrank_square(h, c)                  # d -> d delta
    + shift(c)
)
pre_act = mlp_in(h_adapted)
```

Diagonal and low-rank contributions here are added in parallel. In contrast,
applying the square delta to `h_mod` composes the conditional operations:

```python
h_transformed = h_mod + lowrank_square(h_mod, c)
pre_act = mlp_in(h_transformed)

# Equivalent expansion by linearity; no additional nonlinear depth:
pre_act = mlp_in(h_mod) + F.linear(
    lowrank_square(h_mod, c), mlp_in.weight, bias=None
)
```

The identity residual is essential. A pure rank-r map does not preserve the
base behavior and discards directions when `r < d`. Count the base bias only
once in the expansion. Passing the square delta through `mlp_in.weight` couples
the update to that weight and restricts its output directions. The rectangular
delta can choose independent directions in the expanded MLP space.

Shift before versus after the square transform is another choice: transforming
a shifted input also transforms the shift. Input shift and independently
generated output bias are different parameterizations. Additive placement is
the working preference; composed variants remain untested alternatives.

## What basis mixing factorizes

Use `q` for condition width and `N=B*T` for flattened tokens. A direct generator
predicts every factor entry:

```python
# c: [N, q], h: [N, d]
U = to_U(c).reshape(N, d, r)       # to_U: Linear(q, d*r)
V = to_V(c).reshape(N, r, o)       # to_V: Linear(q, r*o)
delta = torch.bmm(torch.bmm(h[:, None, :], U), V)[:, 0, :]
```

Basis mixing factorizes those generator projections through a small intermediate.
The ordinary two-linear-layer view makes the learned bases explicit:

```python
# Broadcast case: c: [B, q], h: [B, T, d]; s=K per factor generator.
# Learned weights: A_U/A_V: [q, s], D_U: [s, d*r], D_V: [s, r*o].
a = c @ A_U + bias_U                  # [B, s]
b = c @ A_V + bias_V                  # [B, s]
U = (a @ D_U).reshape(B, d, r)        # [B, d, r]
V = (b @ D_V).reshape(B, r, o)        # [B, r, o]
delta = torch.bmm(torch.bmm(h, U), V) # [B, T, o]
```

Each row of `D_U`, reshaped to `[d,r]`, is one input-side basis matrix.
Each row of `D_V`, reshaped to `[r,o]`, is one output-side basis matrix.
These decoder weights are learned and shared across samples/tokens; the
condition-dependent `a,b` select signed combinations of them. The bases are
the second projection's parameters, not an additional free resource. With
linear coefficient heads, the effective condition-to-factor projection has
rank at most `s`; the produced update still has rank at most `r`.

The current code packs the two coefficient encoders into `Linear(q,2*K)` and
stores the input bases in a different axis order:

```python
# Static learned parameters:
# U_basis: [d, K, r]; V_basis: [K, r, o]
# coef: Linear(q, 2*K)

a, b = coef(c).chunk(2, dim=-1)    # broadcast case: each [B, K]
U = torch.einsum("bk,dkr->bdr", a, U_basis)  # [B, d, r]
V = torch.einsum("bk,kro->bro", b, V_basis)  # [B, r, o]
delta = torch.bmm(torch.bmm(h, U), V)       # h: [B, T, d]
```

The flat decoder correspondence is
`D_U = U_basis.permute(1,0,2).reshape(K,d*r)` and
`D_V = V_basis.reshape(K,r*o)`. Two independent linear bottleneck generators
with `s=K` reproduce current basis mixing exactly; renaming the bases does not
create a new method. A shared `c -> s -> concat(U,V)` bottleneck is different:
both factors then depend on the same `s` condition features, whereas separate
heads can use up to `2*s` joint features. Its bias-free cost is
`q*s + s*r*(d+o)`, compared with `2*q*s + s*r*(d+o)` for separate heads.

Coefficients are affine linear outputs, without softmax normalization. The
outer additive arrangement supplies the base-plus-delta behavior. Basis mixing
itself is a learned dictionary for the factors; it is not an explicit static
factor plus conditional offset. `U_static + U_offset(c)` would be another design.

Mixing factors independently keeps the final delta rank at most `r` and
includes cross-products between input and output bases. Mixing `K` complete
rank-r update matrices instead could give rank up to `K*r`; it is a different
expressivity/cost comparison.

For per-token conditions, [condlr.py](condlr.py) mixes activations instead:

```python
# h: [N, d], c: [N, q]
a, b = coef(c).chunk(2, dim=-1)              # each [N, K]
p = (h @ U_basis.reshape(d, K*r)).reshape(N, K, r)
z = (p * a[..., None]).sum(dim=1)           # [N, r]
g = z[:, None, :] * b[..., None]            # [N, K, r]
delta = g.reshape(N, K*r) @ V_basis.reshape(K*r, o)
```

This computes the same delta without constructing `[N,d,r]` and `[N,r,o]`
factors. Large intermediates have shape `[N,K,r]`. For per-token conditions,
the leading matmul FLOPs are comparable to explicit factor generation/application:

```python
# Forward matmul FLOPs; multiply-add counts as 2. Excludes reductions/pointwise.
per_token_weight_mix = 4*N*q*K + 2*N*(K+1)*r*(d+o)
per_token_activation_mix = 4*N*q*K + 2*N*K*r*(d+o)
broadcast_weight_mix = 4*B*q*K + 2*B*K*r*(d+o) + 2*B*T*r*(d+o)
broadcast_activation_mix = 4*B*q*K + 2*B*T*K*r*(d+o)
```

Activation mixing reduces factor materialization without inherently adding
leading matmul work in the per-token case. For broadcast conditioning, weight
mixing amortizes generation across `T` tokens, while activation mixing applies
the `K*r` static projections on every token. Shapes, reductions, saved tensors,
backward and kernels determine actual speed/peak memory; these counts are not
throughput guarantees.

## Parameter and activation ledger

```python
# Direct generator, ignoring biases:
full_generator_params = q * r * (d + o)

# Current basis generator, including its coefficient-head bias:
basis_generator_params = K*r*(d + o) + 2*K*(q + 1)

# Naively materialized per-token factors, in elements:
factor_elements = N*r*(d + o)
# Broadcast factors instead cost B*r*(d + o).
```

At `d=q=192, o=768, K=16`, the current rectangular adapter costs:

| r | Parameters/block | Parameters/six blocks | Broadcast U, V, z BF16 tensor sizes/block |
| ---: | ---: | ---: | ---: |
| 2 | 36,896 | 221,376 | 0.5 MiB |
| 4 | 67,616 | 405,696 | 1 MiB |
| 8 | 129,056 | 774,336 | 2 MiB |
| 16 | 251,936 | 1,511,616 | 4 MiB |

Tensor sizes assume `B=128,T=64` and include the rank bottleneck `z`; they are
not a measurement of total backward or peak training memory. Coefficients,
outputs, other saved tensors, parameters, gradients, optimizer, and EMA add cost.
The 4,179,696-parameter baseline grows by about 19% at rank 8 and 36% at rank 16.

Naive per-token factors use `2*r` times the input token storage for square
`d -> d` updates, or `5*r` times for `d -> 4*d`: rank 8 means 16x or 40x,
respectively. Per-token activation mixing uses multiple `N*K*r` intermediates;
at `d=192,K=16,r=8`, each such tensor is about 0.67x the input token tensor.
Neither ledger is an exact autograd footprint. The exact chunked full-generator
path also avoids retaining full factors, but keeps generator parameter/compute
cost and bounded transient scratch; its small rank bottleneck still grows with r.

A square basis adapter at rank 8 costs 55,328 parameters/block before any new
diagonal/shift heads. It can reuse the common AdaLN heads.

### Fractional generator widths and spending the savings

The proposed sweep uses `s` near `q/4`, `q/16`, and `q/32`, with integer rounding
declared in each config. Use condition width `q`, which need not always equal
token width `d`. These are candidate budgets, not fixed requirements or proven
scaling rules. The current choice remains `s=K=16` per factor, or 32 predicted
coefficients in total; `s=8` was an illustrative alternative.

| Generator width | q=192 | q=1024 |
| --- | ---: | ---: |
| q/4 | 48 | 256 |
| q/16 | 12 | 64 |
| q/32 | 6 | 32 |

At `d=q=192,o=768,r=8`, separate factor-generator costs ignoring biases are:

| Generator | Parameters/block |
| --- | ---: |
| Dense condition-to-factor projection | 1,474,560 |
| Factorized, s=48 | 387,072 |
| Factorized, s=16 (current width) | 129,024 |
| Factorized, s=12 | 96,768 |
| Factorized, s=6 | 48,384 |

For current `s=16,r=8`, decoder bases account for 122,880 parameters and the
coefficient head for 6,176 including bias. Compressing only the small head has
limited impact; the bases dominate. At `q=d=1024,o=4096,r=8,s=q/4`, the same
separate-head design costs 11,010,048 parameters/block before biases. Fractional
widths must still be checked against the whole-model budget.

For a concrete reallocation, `r=16,s=6` costs 94,476 parameters/block including
coefficient biases, versus 129,056 for `r=8,s=16`. The rank bound doubles while
the generator uses fewer condition features. Broadcast factor storage and
application FLOPs double, while a per-token `[N,s,r]` intermediate shrinks from
128 to 96 values/token. This trades expressivity along different axes; it is
an untested proposal, not a cost-free improvement.

## Generator alternatives and diagonal structure (2026-10-08)

Treat the generator and generated transform as independent choices. A nonlinear
encoder such as `a = SiLU(c @ A_U + bias_U)` changes condition dependence while
retaining a learned decoder `a @ D_U`. It does not inherently reduce parameters
or activation storage, and each decoded factor still lies in its `s`-basis
span. The current factor heads are affine in their input condition; the shared
class/timestep condition encoder already contains a nonlinearity.

Diagonal plus low-rank is especially natural when replacing a square map:

```python
# x: [..., q], diagonal: [q], L: [q, p], R: [p, q]
y = x * diagonal + (x @ L) @ R
```

This can retain full rank and direct coordinate channels despite small `p`.
For an additive delta beside an existing full projection, full rank is already
available through the base path; a diagonal delta remains a modeling choice.
The current diffusion diagonal is supplied by the common AdaLN heads.

The direct factor generator maps `q -> r*(d+o)`, a rectangular output of
flattened factor entries. A rectangular diagonal, block correspondence or
structured skip must specify which coordinates it connects; there is no
canonical channel-wise identity for this map. In particular, a linear square
diagonal-plus-low-rank encoder followed by compression to `s` does not restore
the discarded generator directions: the complete linear projection still has
rank at most `s`, and an unrestricted compression can absorb the preceding
square map. A skip around compression or an intervening nonlinearity changes
that argument.

| Alternative | Primary potential saving | Expressivity/execution consequence | Status |
| --- | --- | --- | --- |
| Retune s/K at fixed r, or spend on r | Generator parameters and per-token s*r intermediates | Fewer condition features versus a higher transform-rank bound | Included in the 2026-10-08 engineering screen; quality sweep unrun |
| Shared bottleneck for U and V | Removes one coefficient encoder | Couples condition features; decoder usually remains dominant | Unimplemented |
| Nonlinear bottleneck generator | Richer condition dependence at similar parameter count | Added activation work; no automatic memory/speed saving | Unimplemented factor-head variant |
| CP/Tucker or sparse/structured decoder bases | Compresses the dominant decoder weights | Restrictions depend on decomposed axes; kernel efficiency must be measured | Unimplemented |
| Fixed/frozen decoder bases | Trainable parameters, gradients and optimizer states | Basis buffers and projection work remain; loses learned direction alignment | Unimplemented control |
| Static factors with input/output channel gains | Generates d+o values rather than r*(d+o) | Separable channel rescaling rather than arbitrary factor changes | Unimplemented |
| Static factors with conditional dense or diagonal-plus-low-rank core | Generates r*r or r+2*r*p values | Conditional mixing inside fixed outer subspaces | Implemented; engineering and calibrated three-seed quality confirmation completed below |

Channel modulation has the reference form:

```python
# U0: [d,r], V0: [r,o], both shared across conditions.
# channel_head(c): [B,d+o]; the head itself may be factorized.
g_in, g_out = channel_head(c).split([d, o], dim=-1)
z = (h * (1 + g_in[:, None, :])) @ U0
delta = (z @ V0) * (1 + g_out[:, None, :])
```

Unlike conditional rank gains, input/output channel gains can move the
effective outer subspaces through channel rescaling. Their flexibility is
still restricted. A shared linear bottleneck of width `s` costs
`q*s + s*(d+o) + r*(d+o)` before biases, including the static factors.
For `q=d=192,o=768,r=8,s=12`, that is 21,504 parameters/block. Static GEMMs and
pointwise operations are promising for execution, but no timing result exists.
For zero delta with the `1+gain` form, zero `V0` and keep the input path nonzero;
zeroing gain heads alone would leave a nonzero static delta when both factors
are nonzero. Gauge and gradient wake-up checks remain necessary.

Changing the produced operator to block-diagonal, Kronecker, Hadamard/FFT-style
or tensor-train structure is another axis, covered in the original brief.
Compact generation alone does not guarantee cheap application or backward.

## Per-dimension mixing and a simpler generator control

Global basis coefficients have shape `[...,K]`. Per-rank coefficients
`[...,K,r]` let rank components select different bases. Per-channel coefficients
`[...,d,K]` can be more flexible, but their generator and per-token storage may
recreate the original expense. Channel input/output gains are a cheaper separable
alternative, with substantial overlap with the diagonal branch.

A control now implemented in `BroadcastLowRank` as `conditional_bottleneck`
uses a static bottleneck and conditional rank gains. The following is equivalent
reference notation; the engineering suite exercises the trainer implementation:

```python
import torch
from torch import nn

class ConditionalBottleneck(nn.Module):
    """Reference notation for the implemented conditional bottleneck control."""
    def __init__(self, d, o, q, r):
        super().__init__()
        self.down = nn.Linear(d, r, bias=False)
        self.up = nn.Linear(r, o, bias=False)
        self.to_gain = nn.Linear(q, r)
        nn.init.normal_(self.down.weight, std=d**-0.5)
        nn.init.normal_(self.up.weight, std=r**-0.5)
        nn.init.zeros_(self.to_gain.weight)
        nn.init.zeros_(self.to_gain.bias)

    def forward(self, h, c):
        gain = self.to_gain(c)
        if h.ndim == 3 and gain.ndim == 2:
            gain = gain[:, None, :]
        return self.up(self.down(h) * gain)
```

Nonzero down/up weights allow the zero gain head to learn. The fan-in scaling
is a candidate reference initialization, not a variance-preservation proof
for correlated conditional activations. This costs `r*(d+o)+r*(q+1)` parameters:
9,224/block at the rank-8 rectangular setting. It uses fixed input/output
subspaces with conditional gains, unlike basis-generated factors whose subspaces
can depend on the condition.

## Initialization, gain, gauges, and learning rates

Current diffusion initialization zeros AdaLN scale/shift/gates, final output,
and the adapter output bases. Input bases and coefficient heads stay nonzero.
Scale zero means multiplier one because the application uses `1+scale`.
For an isolated adapter, output bases receive gradients first; other factor
parameters receive gradients once the output side opens. The common zero gates
and final zero output delay upstream learning signals further. Existing tests
verify eventual activation of every adapter parameter.

Do not zero both a dedicated adapter gain and its output factor: neither can
learn. The common residual gate is different because it can open through the
nonzero base MLP. The conditional bottleneck above instead zeros its gain head
and keeps both static projections nonzero.

Zero delta preserves baseline behavior at initialization; it does not guarantee
variance preservation after learning. Both factors depend on the condition:
bias-free heads give quadratic scaling with condition magnitude; affine heads
also introduce linear and constant terms. Condition normalization limits input
scale but does not bound learned operator gain. Basis count and rank can change
signal/update magnitudes independently of expressivity.

```text
variance(base + beta*delta)
  = variance(base) + beta^2*variance(delta)
    + 2*beta*covariance(base, delta)
```

Shared token inputs, especially self-conditioning, invalidate an automatic
independence assumption. Target controlled signal and functional-update
magnitudes rather than requiring every learned transform to preserve variance.
General low-rank mixing is not constrained to be orthogonal or norm-preserving.

There are two distinct learned-coordinate gauges, beyond scalar rescaling:

```python
# Generator gauge: Q is invertible [s,s]. Include the encoder bias in a_new.
a_new = a @ Q
D_U_new = torch.linalg.solve(Q, D_U)
# a_new @ D_U_new == a @ D_U; same generated input factor.

# Rank-coordinate gauge: S is invertible [r,r].
U_new = U @ S
V_new = torch.linalg.solve(S, V)
# U_new @ V_new == U @ V; same conditional transform.
```

The latter is realizable by right-multiplying every input basis by `S` and
left-multiplying every output basis by `S.inverse()`. Scaling, rotations and
permutations give nonunique basis representations. Parallel placement preserves
the base path but does not remove these internal gauges. Equivalent forward
functions can receive different SGD/Adam steps and weight-decay penalties
because their parameter coordinates differ; factor norms alone can therefore
misrepresent gain or optimization balance. A nonlinear encoder can remove some
linear generator symmetries, but does not remove the U/V rank gauge.

Start with balanced initialization, a normalized condition, and a fixed
nonzero branch gain; monitor both sides' norms/gradients together with resulting
function updates, branch RMS/correlation and operator magnitudes. Candidate
remedies include basis/factor normalization, constraints, gain normalization
across K/r, and distinct optimizer groups. Orthogonal initialization alone
does not fix the gauge throughout training. Normalization may change the
function class/runtime, can retain other rescaling freedoms, and must be
compatible with the current zero output bases rather than blindly dividing by
their norms. These remedies remain proposed comparisons.

The existing adapter LR multiplier scales all adapter parameters together.
Separate input-base, output-base, and generator rates remain untested. muP-style
reasoning requires declaring which of `d,q,K,r` scale and which stay fixed, plus
the optimizer and zero-init convention. Forward variance and optimizer-induced
functional updates are separate checks. LoRA scaling rules motivate candidates
but do not automatically transfer to a condition-generated two-factor map.

## Interaction order and compiler strategy

The exact input contraction is bilinear in `h` and `c`. The complete delta uses
the condition twice when both factors are generated:

```python
U = linear_U(c)                       # bias-free for this illustration
V = linear_V(c)                       # bias-free for this illustration
z = h @ U                            # bilinear in h and c
delta = z @ V                        # linear in h, quadratic in c

# If c == h and we scale that normalized representation together:
# delta(a*h, a*h) == a**3 * delta(h, h)
```

Thus the full self-conditioned delta is cubic in `h` in this bias-free linear
case, not merely second-order. Affine factor generators add lower-degree terms;
nonlinear generators and normalization prevent a general polynomial statement
in the raw pre-normalization activation. The conditional bottleneck control
uses the condition only once and has a different interaction order. The
square version fed by `h_mod` can add further conditional products.

The working engineering preference is correctness in ordinary PyTorch first,
then compile, then a custom kernel only if measured limitations justify it.
For the exact path, compile reusable per-chunk bodies while keeping the Python
loop eager as an initial strategy; do not claim that whole-function compilation
is inherently impossible. Control ragged-tail specialization and check backward
as well as forward. A successful compiled smoke run is not a compiled throughput
or peak-memory comparison. Such exact-path per-chunk optimization is still open.

Chunking bounds transient tensor dimensions, not all allocator/autograd overhead.
The input scratch is `[chunk,q,r]` from `h_chunk @ A_flat`, not a literal tensor
of generated `[chunk,d,r]` factors. Eager scratch reaches GPU memory. A fused
implementation could reduce that traffic, but tiling, registers/shared memory,
and weight-gradient reductions across token tiles require explicit design.
The reference loop bodies are not a proven one-program-per-chunk Triton mapping.

Compiler fusion and save/recompute decisions are version-dependent. On this
instance, PyTorch is `2.11.0+cu128`, with the private
`torch._functorch.config.activation_memory_budget` setting present; this is not
a stable portable API or an explicit chunk-memory guarantee. Check the installed
version before using such controls. Current upstream docs can describe newer
versions than this environment. Sources: PyTorch's
[checkpointing/compiler discussion](https://pytorch.org/blog/activation-checkpointing-techniques/)
and [compiler programming model](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.html).

## Proposed comparisons and required measurements

The normalized/modulated input choice and conditional bottleneck now have
implementation support and a bounded engineering workflow, described below.
The original comparison list follows. Later sections record completed
engineering and quality studies; uncovered comparisons remain proposals:

1. Compare normalized versus conditionally modulated adapter input at fixed
   generator, rank, placement, condition, and common initialization.
2. Compare conditional bottleneck gains against basis-generated factors at
   small ranks 2/4 first. Report both parameter and runtime budgets.
3. Compare rectangular delta against square additive modulation, reusing common
   diagonal/shift heads. Keep composed ordering as an explicit ablation.
4. Run short width/rank/basis-count scaling checks before large sweeps. Examine
   basis/rank gain normalization and separate parameter-group rates.
5. Advance to structured transforms, sharing, or finer-grained coefficients
   only when simpler variants establish a useful quality/cost tradeoff.

Record branch RMS and mean, delta/base RMS ratio, base/delta correlation,
gated-residual/input RMS ratio, parameter-group gradients, and changes to the
delta on a fixed diagnostic batch after optimizer steps. Inspect initialization
and early learning as well as trained states. Compare eager/compiled execution,
saved activations, peak allocation, complete forward/backward time, and quality
at equal wall-clock budgets. Small-rank matmuls, contractions, and launches can
lose speed despite favorable FLOP estimates.

No additional weight retention is required for live diagnostics. Any diagnostic
snapshot or retained weights need a stated purpose, storage bound, and retention
before launch, following the workspace artifact policy.

## Instrumented design engineering workflow

`train_dit_cifar10.py` accepts `--adapter-input normalized|modulated` and
`--variant conditional_bottleneck`, alongside the original variants. The default
input remains normalized. No square placement, private static affine input,
factor-normalization rule, or asymmetric LR schedule has been implemented yet.

The [benchmark](benchmark_dit_design.py) and [launcher](launch_dit_design.py)
define eight configurations: AdaLN; current rank-8/K=16 DPLR; rank-2/4 basis
mixing with each adapter input; and normalized-input rank-2/4 conditional
bottlenecks. All are width 192, depth 6, with common base initialization and
identical selection, flip, DDPM timestep, and noise streams. Paired basis-input
variants also have identical adapter initialization. Configurations run in a
predeclared randomized order on one lifetime-claimed GPU under private supervisor.

Each case trains for 500 steps at batch 128, using a fixed 4,096-image bank from
the train side of the original split, sampled with replacement. Validation uses
256 held-out images. LR is shared at 0.0004, warmup 50, cosine floor 1%, no weight
decay, global gradient clip 1, and no EMA. This is an engineering subset protocol,
not a rerun of the 30k-step full-data quality confirmations.

Untimed diagnostics measure every block's input/base/delta/scale/shift/gate RMS,
base/delta correlation, gated residual ratios, parameter-group gradients, and
FP32 delta changes on fixed tokens/conditions around optimizer steps. Gradient
wake-up is observed at listed diagnostic steps, not claimed as an exact onset
step. Timed training uses eager execution with per-step synchronization and
includes cached CPU batch selection/flip, H2D, noise, forward/backward, and AdamW.
Diagnostics and validation are excluded from the timer; total wall time is larger.

Live trained first-block weights then feed eager/compiled forward-backward probes
of the MLP input projection, including non-affine normalization and the common
six-output modulation calculation. A separate synthetic per-token extension uses
activation mixing with the same weights. It measures execution/memory, not quality
of a trained token-conditioned model. Compile cold time, three steady-state timing
repeats, numerical output/gradient parity, and allocator peaks are recorded.
Eager saved-storage accounting includes input storage and autocast weight copies
but excludes original parameters; it is distinct from compiled memory scheduling.

The workflow declares a 64 MiB evidence bound before launch and saves no model
weights/checkpoints. [The independent audit](audit_dit_design.py) verifies source
hashes, paired initialization, streams, budgets, finite metrics, gradient wake-up,
compiled parity, and artifact policy. Results must remain engineering diagnostics:
no quality winner is selected from these short single-seed runs. CPU numerical
and integration checks passed 33 tests before the GPU launch.

The completed [engineering screen](README.md#completed-design-engineering-screen)
passed all eight cases and 16 compiled projection probes. The smaller bottlenecks
reduced adapter parameters by 93–94% at matched rank, but their final delta/base
RMS ratios were 0.068/0.084 versus 0.321/0.476 for normalized-input basis rank 2/4
and 0.690 for rank 8. This motivates gain and functional-update controls; it does
not establish a quality advantage for either generator. Compilation benefits
varied, and a reliable speed ordering is still open. Raw validation MSE is a
short engineering diagnostic rather than a selection metric. See the
[readable report](reports/dit-design-engineering.md) for shapes/scopes and timings.

## Fixed gain and functional-update calibration

`train_dit_cifar10.py` now accepts a fixed positive `--adapter-gain` (default 1),
alongside the existing `--adapter-lr-multiplier`. The gain multiplies the entire
adapter delta before addition to `mlp_in(h_mod)`. It is not a trainable gate:
identity initialization still comes from zero `V` for basis mixing and zero
condition-to-rank gains for the conditional bottleneck. Default gain 1 takes the
original arithmetic path and draws no extra random numbers. Sample-wise and
per-token contractions apply the same gain.

```python
h = norm_mlp(tokens)                         # (B, T, d), affine=False
h_mod = h * (1 + scale[:, None]) + shift[:, None]
base = mlp_in(h_mod)                         # (B, T, 4*d)
delta = adapter_gain * conditional_lowrank(h, condition)
pre_act = base + delta
tokens = tokens + gate[:, None] * mlp_out(gelu(pre_act))
```

The [gain workflow](benchmark_dit_gain.py) tests normalized-input basis ranks
2/4/8 and conditional bottleneck ranks 2/4 with `adapter_gain = 1/sqrt(rank)`.
This is an empirical candidate, not a variance-preserving or muP-derived rule.
The reference is the scaled rank-8 basis model at adapter LR multiplier 1.
Calibration measures the arithmetic mean delta-function update RMS on fixed
independent normalized tokens/conditions across six blocks at steps 10/25/50.
These probes use FP32 tensors with `float32_matmul_precision="highest"` and no
autocast; they restore the training matmul setting afterward. This keeps small
late-stage function differences from inheriting the faster training precision.
Base LR stays 0.0004; all three adapter parameter groups share one multiplier.

Every family first runs at multiplier 1. If its update differs from the target
by more than 25%, a correction uses `current_rate * target / observed`, clipped
to `[1/16, 16]`. At most two corrections are allowed. The closest absolute log
update ratio wins; neither training loss nor validation selects the rate. The
protocol is bounded to 13 calibration runs of 50 steps, then six confirmations
of 500 steps: the five chosen families and the original raw-gain rank-8 anchor.
The chosen rates are recorded before confirmation begins.

Calibration and confirmation deliberately share their early initialization,
batch/noise stream, and 50-step warmup. The [audit](audit_dit_gain.py) checks the
replayed prefix and function-update metric. Later diagnostic steps measure
transfer over time rather than independent-seed generalization. Fixed-input
adapter movement, live image/condition branch magnitudes, and gated residual
magnitudes are distinct observables; a single rate is not expected to match all
three throughout training. Whole-model eager timing is recorded, while the
previous compile/memory microbenchmarks are not repeated here.

The same train-only bank, held-out training-split validation, source snapshots,
managed lifetime GPU claim, and 64 MiB compact-evidence bound apply. There are no
weights/checkpoints, EMA, FID, or official-test evaluation. Separate factor LR
ratios, factor normalization, generator initialization changes, and wider
architectural comparisons remain open.

```bash
/venv/main/bin/python -m pytest -q test_dit_gain.py test_dit_design.py test_dit_cifar10.py test_class_fidelity.py
/venv/main/bin/python launch_dit_design.py --gain-screen --launch
```

The [completed screen](README.md#completed-gain-and-functional-update-screen)
used seven calibrations and six confirmations. All five selected families met
the early ±25% tolerance. Their step-500 fixed-function update ratios span
0.852–1.178 relative to scaled basis rank 8, while live delta/base RMS is
0.322/0.445/0.572 for basis ranks 2/4/8 and 0.047/0.076 for bottleneck ranks 2/4.
Thus comparable function movement does not imply comparable branch magnitude.
The bottlenecks needed lower adapter rates (0.632/0.793), despite producing
smaller learned deltas. Neither raising LR simply to inflate their outputs nor
calling their small outputs under-optimization is justified by this screen.

Matching is approximate rather than constant over time: the rank-2 bottleneck's
step-250 update ratio was 1.38. Fixed rank scaling also leaves a substantial live
branch-magnitude difference across basis ranks. These are observations for one
seed and recipe, not evidence of universal scaling rules or quality superiority.
An initial faster-precision diagnostic run was superseded by a full-FP32 probe
rerun; keep that distinction when reading raw artifacts. The
[audited report](reports/dit-gain-engineering.md) includes the trajectories,
pre-activation magnitudes, precision scope, and artifact hashes. All 46 tests
passed; both managed workers released their GPU claims without saving weights.

## Fresh-seed calibrated-recipe quality workflow

The [quality workflow](dit_gain_quality.py) transfers the corrected audited
gain/rate choices into one full-data seed-47 screen, with nine configurations:
basis ranks 2/4/8, conditional bottleneck ranks 2/4, raw-gain rank-8 DPLR, and
AdaLN widths 192/216/240. Recipes are frozen before launch, and no rate is
selected from this screen's quality. Each uses 5,000 steps, batch 128, base LR
0.0004, warmup 50, cosine floor 1%, global clip 1, and in-memory EMA decay .999.
The original 45,000/5,000 train/held-out split is retained; the subset bank is
replaced by the full augmented training loader. This is a short quality screen,
not the prior 30k-step, three-seed confirmation protocol.

Final EMA sampling uses DDIM 50 steps, seed 2028, 5,000 real/fake images for
FID, and the original frozen auxiliary classifier on the same generated stream.
The AdaLN width controls are rerun at the same new budget/seed. Source/configs,
the corrected calibration report hash, evaluator hashes and storage/retention
are recorded before launch. At most three finite managed workers use the shared
lifetime GPU claim; the workflow retains no model weights/checkpoints.

`train_dit_cifar10.py` accepts opt-in `--diagnostic-steps` as comma-separated
integers. [Diagnostics](dit_quality_diagnostics.py) record live raw-model branch
magnitudes, parameter-group gradients, highest-FP32 function movement on fixed
inputs/conditions, batch/noise digests at observed steps, initial parameter
hashes and the final noise RNG state. Probe/logging time is excluded from train
throughput windows and included in total loop wall time. Diagnostic memory is
excluded from training peaks; fixed probe inputs remain resident and are named
in the allocation scope. EMA quality and raw optimization diagnostics are
distinct states. The early update match is measured on new data/seed rather
than assumed to transfer.

The [audit](audit_dit_gain_quality.py) verifies frozen recipe transfer, source,
equal splits/budgets/evaluator inputs, common width-192 base initialization,
paired raw/scaled rank-8 initialization, observed batch/noise digests, zero
initialization and adapter gradient wake-up, FID/agreement denominators and
successful exits. Diagnostic digests do not constitute a hash of every image
in the training stream. There are no official-test results, new evaluator
weights, or sequence-conditioned attention comparisons. One-seed endpoints
cannot establish multi-seed or long-budget superiority.

```bash
/venv/main/bin/python dit_gain_quality.py prepare --launch
/venv/main/bin/python audit_dit_gain_quality.py runs/dit-gain-quality-YYYYMMDDTHHMMSSZ
```

The [completed fresh-seed screen](README.md#fresh-seed-quality-screen-of-calibrated-recipes)
has short quality results for all nine configurations. Width-240 AdaLN has the
lowest observed FID (135.96); raw DPLR r8 has the highest requested-class
agreement (28.80%). Bottleneck r2 gives FID 138.95 with only 13,836 added
parameters and agreement 23.84%, close to width-192 AdaLN's 24.02%. Basis r2
gives FID 140.47 and agreement 25.38%, with 221,376 extra parameters. No recipe
dominates global FID and conditional agreement, and a one-seed 5k-step screen
does not establish long-run superiority.

The early update match does not fully transfer. Bottleneck r2/r4 update ratios
relative to scaled basis r8 are 0.749/0.641 on the new early window and
0.568/0.486 at step 5,000; basis r2/r4 ratios are 1.016/0.965 early and
0.965/0.768 at the endpoint. Full-data/seed calibration or explicit tracking of
this drift should accompany longer paired-seed comparisons of the small rank-2
variants. Simple rate corrections are recipe choices, not a proven architecture
scaling rule. See [quality and transfer figures](reports/dit-gain-quality.md).
All 53 tests and the independent audit passed; no new weights or checkpoints
were saved and managed workers released their claims.

## Evidence boundary and broader conversation

The [completed diffusion confirmations](README.md#diffusion-three-seed-confirmation-results)
test only class/time broadcast conditioning. DPLR improved requested-class
agreement under a fixed auxiliary classifier and beat the closest parameter
control on FID; the widest AdaLN control retained better FID. Concurrent DPLR
training throughput was about 20% lower. This does not establish an overall
quality, isolated speed, or wall-clock win. The self-conditioned ViT comparison
showed no clear advantage over widening. The new normalized-input gain/generator
variants now have the longer paired-seed confirmation below. Modulated/static-affine
inputs and other proposed transforms remain without quality evaluation.

Sequence conditioning remains a separate priority: compare pooled-context AdaLN,
pooled-context DPLR, cross-attention, and cross-attention plus DPLR; then consider
query-dependent retrieval feeding token-specific transforms. These comparisons
must distinguish context access from channel transformation. A broadcast vector
does not by itself reproduce position-dependent selection from a context sequence.

The supplied setup conversation is reconciled at the start of this document.
It establishes the motivation and original implementation strategy, not a new
quality result or a mandate to write a fused kernel. The present record still
lacks the original promising runs' configurations/metrics and a concrete
sequence-valued conditioning benchmark. Keep those as open context rather than
inferring them from the setup narrative.

## Optimization references

- [Tensor Programs V / muP](https://arxiv.org/abs/2203.03466): initialization and
  update scaling for hyperparameter transfer across widths; the
  [authors' implementation](https://github.com/microsoft/mup) distinguishes fixed
  and scaling dimensions. Our conditional factorization still needs validation.
- [Rank-stabilized LoRA](https://arxiv.org/abs/2312.03732): motivates testing
  `1/sqrt(r)` scaling in its LoRA setting, not a proven rule for this generator.
- [LoRA+](https://arxiv.org/abs/2402.12354): motivates different factor learning
  rates; a ratio for our bases and coefficient heads has not been established.

## Full-loader rank-2 calibration and longer paired confirmation

This comparison uses `dit_rank2_confirmation.py` and the frozen protocol in
`runs/dit-rank2-20261008T010252Z/protocol.json`. Calibration runs the actual
45,000-example training loader, horizontal flips, batch size 128, optimizer,
clipping, diffusion noise and model initialization on seeds 48 and 49. It runs
500 steps of the same 30,000-step LR schedule as confirmation, with warmup 50.
The optional `lr_schedule_steps` field makes that prefix explicit; leaving it
unset preserves the trainer's existing schedule. An integration test verifies
identical diagnostic states and validation outputs over a short/long prefix.

The selection metric is the arithmetic mean of independent fixed-input
transform-update RMS across all six blocks, steps 50/100/250/500, and both
calibration seeds. Probes use FP32 with highest FP32 matmul precision, without
autocast; training remains BF16. Scaled basis r8 at rate multiplier 1 is the
reference. Start each rank-2 family at multiplier 1; if its mean falls outside
25% of the reference, propose `clip(rate * target / observed, 1/16, 16)`, with
at most two corrections. Choose the candidate closest in absolute log ratio.
This rule never reads loss, validation or generated quality. One global rate
per family is frozen before independent confirmation seeds begin.

Both initial choices matched without correction: basis r2/reference 0.9637,
bottleneck r2/reference 1.0345. The chosen multiplier is 1 for both. The
bottleneck choice therefore differs from the earlier subset-calibrated 0.6316.
This is matching one aggregate over a longer window, not a guarantee of equal
updates at every step or seed. Individual block/seed ratios and later drift
remain diagnostics. No variance-preservation or muP scaling rule is established.

Confirmation is 30,000 steps on seeds 50/51/52 for basis r2, bottleneck r2,
scaled basis r8, raw-gain r8, AdaLN-192 and AdaLN-240. All use the same base LR
4e-4, actual full loader, held-out 5,000-example split, optimizer, EMA .999,
50-step DDIM sampler without CFG, and 5,000 real/generated-image FID with the
existing frozen requested-class classifier. Warmup 50 differs from the older
30k study's warmup 500; comparisons should use the newly run controls. The
official test set is untouched. A fixed sampling seed isolates training-seed
variation but does not measure sampling-seed uncertainty.

At most six private-supervisor workers acquire lifetime shared GPU claims.
Expected compute before launch is 6–9 aggregate GPU-hours. Declared compact
storage is at most 512 MiB; configs, sources, provenance, splits, metrics,
diagnostics, logs, reports and 100-image grids are retained until explicitly
removed. No diffusion checkpoints, final weights or new evaluator weights are
saved. This tests class/time broadcast conditioning; sequence-valued conditioning
and cross-attention remain separate work.

## Small conditional core (implemented, engineering screen)

Implemented on 2026-10-08 as `--variant conditional_core` in
[dit_cifar10.py](dit_cifar10.py). The diagonal-plus-low-rank version below is
`--variant conditional_dplr_core`, with inner `--core-rank` distinct from outer
`--rank`. Both use static learned U/V, broadcast class/timestep conditioning,
common non-affine normalization/AdaLN/gates and the parallel MLP-input delta.
Tests cover explicit dense-matrix forward/all-gradient oracles, BF16, zero
initialization/wake-up, trainer diagnostics and parameter budgets. Neither core
has a longer quality confirmation.

A useful intermediate between the diagonal conditional bottleneck and generated
factor bases is a Tucker-style conditional core. Keep both large bases static
and learn a small dense transform in their rank space:

```python
# h: [B, T, d], c: [B, q]
# U: [d, r], V: [r, o]; both initialized nonzero
G = core_head(c).view(B, r, r)        # head: Linear(q, r*r)
z = h @ U                           # [B, T, r]
z = torch.bmm(z, G)                 # [B, T, r]
delta = gain * (z @ V)              # [B, T, o]
pre_act = mlp_in(h_mod) + delta      # common GELU/output/gate follow
```

The existing bottleneck is the special case `G = diag(gains(c))`; it only
changes the strength of fixed rank directions. A dense core also mixes them,
while its input and output subspaces remain fixed for a given set of learned
bases. Current basis DPLR can change those subspaces with the condition, and its
two generated factors give quadratic condition interactions. A linear dense
core is a different function class, not an equivalent implementation of DPLR.
A nonlinear or factorized core head is another independent generator choice.

With bias in the core head and none in U/V, six width-192 blocks with output
width 768 add `6 * (192*r + r*768 + (192+1)*r*r)` parameters:

| Core rank | Added parameters | Broadcast generated values/sample/block | Token-conditioned generated values/token/block |
|---|---:|---:|---:|
| 4 | 41,568 | 16 | 16 |
| 8 | 120,192 | 64 | 64 |
| 16 | 388,608 | 256 | 256 |

At rank 8 and batch 128, the broadcast core itself is 8,192 values (16 KiB in
BF16) per block, plus the token bottleneck and the rest of the block's activations.
For token conditioning, use an einsum or equivalent batched contraction with
`G[B,T,r,r]`; core storage then scales with `B*T*r*r`. These are generated-core
counts, not total autograd or peak-memory estimates. The dense core adds
`2*r*r` multiply/add FLOPs per token beyond the two static projections, and
its head costs roughly `2*q*r*r` FLOPs per condition. Static GEMMs may execute
differently from the current generated-factor BMMs; throughput must be measured.

Zero the core head's weight and bias, keeping U/V nonzero, to start the parallel
delta at zero. The head receives gradients after the shared downstream zero
initializations open; U/V receive gradients once the core becomes nonzero.
Avoid zeroing all three components. Changing the bases' coordinates can be
compensated in G, so factor gauges and optimizer-dependent scaling remain.
Measure function updates, branch RMS/correlation, gradient wake-up and actual
peak/throughput rather than assuming the extra matrix multiplication preserves
variance or learns more easily. This implementation does not alter the frozen
rank-2 comparison and still does not provide sequence-context retrieval.

### Diagonal plus low-rank core (implemented)

Diagonal plus low-rank has a direct coordinate correspondence inside the
square rank-space core. Let `p` denote its inner factor rank, distinct from
outer transform rank `r` and generator width `s`:

```python
# g: [B,r], L: [B,r,p], R: [B,p,r], generated from c.
# Mathematical core: G = diag_embed(g) + L @ R.
# Apply its factors directly instead of allocating [B,r,r]:
z = h @ U                           # [B,T,r]
core_out = z * g[:, None, :] + torch.bmm(torch.bmm(z, L), R)
delta = gain * (core_out @ V)        # [B,T,o]
```

This generates `r+2*r*p` entries rather than `r*r`; it saves head parameters
and generated storage only when `r+2*r*p < r*r`. Including static U/V and a
single direct affine head, the parameter count is
`r*(d+o) + (q+1)*(r+2*r*p)`. For `q=d=192,o=768,r=8,p=2`, that is 15,400
parameters/block and 40 generated values/sample, versus 20,032 and 64 for the
dense core. Both keep the outer update rank at most `r` and fixed outer
subspaces. The diagonal can retain a full-rank core even when `p` is small;
it does not make the complete d-to-o update full rank.

The implemented zero-core initialization sets the diagonal head and R head to
zero, the L head nonzero, and both static U/V nonzero. The diagonal and R heads
can learn once downstream shared gates open; L and U/V receive gradients after
the core opens. Zeroing both L and R would stall the low-rank core component.
This introduces additional factor gauges, and actual speed depends on the
diagonal operation plus small contractions, not just fewer generated entries.
The code uses separate affine `gains`, `core_left` and `core_right` heads;
their combined parameter count equals a packed affine head. L uses the default
nonzero PyTorch linear initialization, while static U/V use fan-in scaling.
This is an explicit starting recipe, not a derived gain/LR scaling rule.

## Completed full-loader rank-2 confirmation (2026-10-08)

The six calibration and 18 longer confirmation runs finished successfully.
| Variant | Added adapter parameters | FID ↓ | Requested-class agreement ↑ |
|---|---:|---:|---:|
| AdaLN 192 | 0 | 68.94 ± 0.42 | 40.11 ± 0.73% |
| Bottleneck r2 | 13,836 | 69.55 ± 0.91 | 40.06 ± 0.53% |
| Basis r2 | 221,376 | 65.89 ± 0.73 | 44.67 ± 0.32% |
| Basis r8 / sqrt(8) | 774,336 | 61.99 ± 0.33 | 50.15 ± 0.72% |
| Raw DPLR r8 | 774,336 | 60.81 ± 0.91 | 51.17 ± 0.54% |
| AdaLN 240 | 0 | 61.59 ± 0.43 | 43.81 ± 0.36% |

Values are mean ± sample SD across training seeds 50/51/52, with 30,000 steps
per run; they are not confidence intervals. Basis r2 beats AdaLN-192 on FID and
requested-class agreement in all paired seeds. It remains behind rank 8.
Bottleneck r2 shows no clear base-quality improvement despite a tiny parameter
addition. Raw r8 beats AdaLN-240 on both metrics in all paired seeds with fewer
parameters and lower whole-training peak allocation, but has lower recorded
concurrent throughput. Equal steps are not an equal wall-clock comparison.

Early update/reference ratios average 0.935 (basis r2), 0.955 (bottleneck r2)
and 2.503 (raw r8); late ratios are 1.025, 0.689 and 2.295. Bottleneck seed 50
misses the early ±25% transfer tolerance at 0.690, although calibration matched
its two-seed aggregate. The rate remains frozen. Scalar update matching cannot
be treated as identical magnitude, seed/block trajectories or optimizer geometry.

The final live delta/base RMS averages 0.041/0.356/0.672/0.540 for bottleneck
r2/basis r2/scaled r8/raw r8. Combined pre-activation/base RMS averages
1.001/1.139/1.367/1.266, with centered base–delta correlations
0.004/0.100/0.179/0.173. These fixed eight-image raw-model diagnostics make
reinforcing contributions visible. They do not establish variance preservation
for all tokens/conditions or identify a universal gain/LR rule.

A useful exact parameter-budget control for basis r2 is width-192 AdaLN with
MLP hidden width 864 (`mlp_ratio=4.5`). Raising each MLP from 768 to 864 adds
`6 * (2*192 + 1) * 96 = 221,760` parameters, only 384 more than basis r2.
That control has not been run; current width-192/240 controls do not isolate
conditioning from added capacity. Test it before attributing the rank-2 gain
specifically to conditional transforms. Isolated eager/compiled benchmarks and
the conditional-core proposal above can then test whether the extra operation
pays for its cost. Sequence-context access remains a separate benchmark.

All 62 pre-launch tests and the independent audit passed. The audit checks
full splits/budgets/schedule prefixes, calibration and frozen recipes,
source/runtime/evaluators, paired base/adapter initialization, observed batch/
noise digests, final noise RNG states, zero initialization/gradient wake-up,
lifetime-claim GPU UUIDs, metric denominators, finite results and successful
exits. No new weights/checkpoints or official-test evaluations were retained/
performed. Around 40 MiB of compact raw evidence is retained; supervisors are
shut down and claims released. See [the complete report](reports/dit-rank2-confirmation.md)
and [README results](README.md#full-loader-calibrated-rank-2-confirmation).

## Generator/core engineering workflow (2026-10-08)

[benchmark_dit_generators.py](benchmark_dit_generators.py) declares 14 cases:
AdaLN-192; its FFN-expanded budget control (`mlp_ratio=4.5`); basis r2/s16;
scaled/raw basis r8/s16; basis r8 at s=48/12/6; basis r16/s6; conditional
bottleneck r8; dense cores r4/r8/r16; and diagonal-plus-low-rank core r8/p2.
Fractional widths use `max(1,q//divisor)` and the current q=d=192. All adapters
receive normalized input. Fixed gain is `1/sqrt(r)` except the raw basis
reference; adapter LR multiplier remains 1, without function-update calibration.

The bounded screen uses 500 steps/case, warmup 50 and the prefix of a 30k cosine
horizon, base LR 4e-4, AdamW (.9,.999), no weight decay, global clipping at 1,
batch 128/BF16 and seed 42. It samples a cached 4,096-image training-only bank
with replacement and shared flips/noise, with 256 separate held-out validation
images. This is a learning/engineering diagnostic, not full-loader quality
confirmation. The FFN total budget differs from basis r2 by only 384 parameters,
but changing MLP shapes also changes initialization; identical base hashes are
required only among equal-shape variants. Raw/scaled basis r8 share adapter
initialization. Do not pool these timings or losses with the completed 30k study.

After training each case, the first-block pre-activation projection is measured
in eager and compiled forward/backward for broadcast and synthetic per-token
conditions. Dense token cores retain `[B,T,r,r]`; factored cores apply their
diagonal/L/R directly. Basis token probes use activation mixing. Hooks record
branch magnitude/correlation, factor/head gradients, and fixed-input highest-FP32
function updates outside timed training. Noise-MSE validation is not FID or a
quality selection rule. There is no EMA, sampling, official test evaluation,
model-weight saving or periodic checkpointing.

Launch under one lifetime GPU claim and private supervisor:

```bash
/venv/main/bin/python launch_dit_design.py --generator-screen --steps 500 --launch
/venv/main/bin/python runs/WORKFLOW/source/audit_dit_generators.py runs/WORKFLOW
/venv/main/bin/python runs/WORKFLOW/source/summarize_dit_generators.py runs/WORKFLOW
```

The manifest records purpose, seeds, frozen source, initialization/scaling,
disk capacity and a 128 MiB upper bound for compact evidence before launch.
Compiler caches are outside that evidence bound. Retain source/configs/splits,
metrics/logs/reports/manifests until explicitly removed. The full 14-case CPU
smoke completed and was audited; all 74 targeted tests passed before the real
screen launched as `runs/dit-generators-20261008T031001Z`.

### Completed generator/core engineering screen

All 14 cases completed on one claimed GPU in about 5.4 minutes including
projection probes and compilation. The independent audit verifies frozen
source/specifications, parameter/FLOP budgets, shared batch/noise streams,
equal-shape base initialization, raw/scaled basis pairing, zero deltas,
eventual nonzero adapter gradients and compiled numerical parity. The maximum
compiled output/gradient relative-L2 error across the probes is below 0.5% in
BF16. This is one-seed engineering evidence, not quality confirmation.

| Case | Added parameters versus AdaLN | Whole-training peak MiB (no EMA) | Final mean block delta/base RMS |
| --- | ---: | ---: | ---: |
| AdaLN | 0 | 560.6 | 0 |
| FFN budget control | 221,760 | 591.1 | 0 (no adapter) |
| Basis r2/s16 | 221,376 | 587.4 | 0.275 |
| Basis r8/s16, scaled | 774,336 | 603.4 | 0.574 |
| Basis r8/s48 (q/4) | 2,323,008 | 628.1 | 0.839 |
| Basis r8/s12 (q/16) | 580,752 | 600.8 | 0.596 |
| Basis r8/s6 (q/32) | 290,376 | 596.8 | 0.417 |
| Basis r16/s6 | 566,856 | 609.6 | 0.585 |
| Dense core r8 | 120,192 | 584.8 | 0.282 |
| DPLR core r8/p2 | 92,400 | 585.0 | 0.134 |

Dense/DPLR r8 cores reduce adapter parameters by 84.5%/88.1% relative to
scaled basis r8, while whole-training peak allocation falls only about 3%.
Their first-block eager per-token saved storage is 35.05/34.70 MiB versus
38.76 MiB for basis r8; these include common projection-path saved storage,
not adapter-only tensors. Dense core r16 uses 38.39 MiB in that probe, so
increasing core rank still has a meaningful r*r token-storage cost.

Trading s16/r8 for s6/r16 saves 26.8% of generator parameters but increases
broadcast peak allocation to 609.6 MiB; token saved storage falls modestly to
37.63 MiB. Broadening s to q/4 triples the adapter parameters and raises token
saved storage to 48.25 MiB. The budget differences are measured; a quality
tradeoff has not been established.

Final mean highest-FP32 fixed-input adapter-delta update RMS is approximately
0.000346 for scaled basis r8, 0.000686 for dense core r8, and 0.000619 for DPLR
core r8/p2, despite the cores' smaller live deltas. Smaller output magnitude
does not imply smaller optimizer-induced function movement. Equal scalar
gain/LR is not a functional-update or signal-magnitude match; calibrate future
core recipes independently before attributing quality to their capacity.
Zeros for base/FFN adapter probes mean no adapter, not zero base learning.

Compilation improves some probes and hurts others, with material timing
variation across repeats/cases. The GPU is exclusively claimed, but CPU/host
resources are shared. No stable architectural speed winner is selected from
this single ordered run. Longer paired-seed FFN/core quality evaluation is
recorded in the subsequent section. None of
these results concern sequence-context retrieval or cross-attention.

Around 4.6 MiB of compact raw evidence is retained under the 128 MiB bound,
plus the exported compact report. No weights/checkpoints/images/test evaluation
were retained/performed. See [the report](reports/dit-generator-engineering.md),
[full data](reports/dit-generator-engineering.json) and
[audit](reports/dit-generator-engineering-audit.json).

## Full-loader core calibration and fresh-seed quality confirmation

[dit_core_confirmation.py](dit_core_confirmation.py) implements the seven-family
quality study: AdaLN-192, the FFN budget control (MLP
ratio 4.5), basis r2, scaled/raw basis r8, dense core r8 and DPLR core r8/p2.
Every family has three new confirmation seeds, 55/56/57, at 30,000 steps.
Calibration uses separate new seeds 53/54; none overlaps the prior studies.
The base/FFN/basis-r2/raw-reference LR multipliers remain fixed at 1. Core
gains remain `1/sqrt(8)`, independent of the calibrated LR multipliers.

The calibration runs the actual 45,000-image loader for the first 500 steps
of the 30k schedule, with warmup 50. Its metric is the arithmetic mean
highest-FP32 fixed-input adapter-delta update RMS over six blocks, steps
50/100/250/500, and both calibration seeds. At most two corrections use
`clip(previous_rate * target / observed, 1/16, 16)`, selecting the closest
log-distance to the scaled-r8 target and requiring aggregate ±25% tolerance.
No training loss, validation, FID or class agreement enters selection. Recipes
are frozen before confirmation; rates are never adjusted for confirmation
seed drift or quality results.

In `runs/dit-core-20261008T040258Z`, both cores matched after one correction:

| Family | Initial update / target | Frozen adapter LR multiplier | Selected update / target |
| --- | ---: | ---: | ---: |
| Scaled basis r8 | 1 | 1 | 1 |
| Dense core r8 | 2.412 | 0.414517 | 1.158 |
| DPLR core r8/p2 | 2.115 | 0.472845 | 1.064 |

All 81 targeted tests passed before launch, including exact short/long training
prefixes for both cores and the FFN control, plus diagnostic preservation of
training and EMA. The confirmation uses full training/held-out splits of
45k/5k, base LR 4e-4, AdamW (.9,.999), no weight decay, global clipping 1,
cosine floor .01, batch 128/BF16 and in-memory EMA .999. Evaluation uses
5,000 real and 5,000 generated images, 50-step DDIM without CFG, sampling
seed 2028 and the existing frozen FP32 classifier; no new evaluator weights.
All 21 confirmation runs completed successfully, and the independent audit
passed. Results and interpretation follow below.

Workers use private supervisors and lifetime GPU claims, at most six
concurrent, with an estimated 7–10 aggregate GPU-hours. The pre-launch manifest
declares a 512 MiB bound for compact source/configs/seeds/splits/metrics/logs,
reports and one 100-image grid per confirmation. No periodic checkpoints,
final model weights, generated-image datasets or official test evaluation.
Retention remains until explicitly removed. Changing FFN shapes changes its
initialization; compare the near-exact capacity budget without claiming an
identical base. Equal-shape base hashes and raw/scaled adapter hashes are
paired. Scalar calibration does not ensure identical branch magnitude,
optimizer geometry, every seed/block update, or variance preservation.

```bash
/venv/main/bin/python dit_core_confirmation.py prepare --launch
/venv/main/bin/python runs/WORKFLOW/runner/audit_dit_core.py runs/WORKFLOW
/venv/main/bin/python runs/WORKFLOW/runner/summarize_dit_core.py runs/WORKFLOW
```

Audit/export follow successful worker completion and closure of this
workflow's private controller supervisor. The audit independently reconstructs
calibration, configs/counts, initialization hashes, source versions, shared
batch/noise traces, evaluator identity/denominators and absence of checkpoints.
Concurrent timings and equal-step budgets cannot establish a stable speed or
equal-time quality winner; sequence-valued conditioning remains untested.

### Completed core/FFN comparison

The [audited report](reports/dit-core-confirmation.md) exports full aggregates,
per-seed paired differences, quality/update figures and artifact hashes.
Values below are mean ± sample SD over seeds 55/56/57, not confidence intervals.

| Family | Added parameters | FID ↓ | Class agreement ↑ | Whole-training peak MiB |
| --- | ---: | ---: | ---: | ---: |
| AdaLN-192 | 0 | 68.91 ± 0.73 | 40.21 ± 0.82% | 574.3 |
| FFN budget control | 221,760 | 67.06 ± 0.40 | 40.96 ± 0.97% | 605.7 |
| Basis r2 | 221,376 | 65.33 ± 1.43 | 45.35 ± 0.25% | 602.0 |
| Scaled basis r8 | 774,336 | 61.43 ± 0.65 | 50.38 ± 0.73% | 620.2 |
| Raw basis r8 | 774,336 | 62.23 ± 1.08 | 50.82 ± 0.26% | 620.2 |
| Dense core r8 | 120,192 | 68.21 ± 0.91 | 41.11 ± 0.83% | 599.0 |
| DPLR core r8/p2 | 92,400 | 68.34 ± 0.19 | 41.49 ± 0.62% | 599.2 |

Basis-r2 minus FFN FID differences are `[-2.13, +0.10, -3.17]` in seed order
55/56/57; agreement differences are `[+5.76, +3.66, +3.74]` percentage points.
Mean FID improves by 1.73 and agreement by 4.39 points, at almost the same
parameter budget. This supports useful conditional mixing beyond adding FFN
capacity for this recipe, but not an all-seed FID win. FFN's different base
initialization, finite 5k FID and three seeds limit causal/statistical claims.

Both cores have worse FID and agreement than basis r2 and both r8 references
in every paired seed. Mean FID improvements over AdaLN are only 0.70/0.57,
with one seed worse for each core; agreement improves by 0.91/1.28 points.
The clear measured benefit is 84.5%/88.1% adapter-parameter compression versus
basis r8. Whole-training peaks, including in-memory EMA, fall approximately
3.4%. These percentages are different denominators: adapter parameters are
only part of the model, and model/optimizer/activations all contribute to peaks.

| Adapter | Early update / paired scaled r8 | Final update / paired scaled r8 | Final live delta/base |
| --- | ---: | ---: | ---: |
| Basis r2 | 0.962 ± 0.065 | 0.886 ± 0.245 | 0.359 ± 0.023 |
| Scaled basis r8 | 1 | 1 | 0.611 ± 0.059 |
| Raw basis r8 | 2.498 ± 0.075 | 2.053 ± 0.169 | 0.574 ± 0.027 |
| Dense core r8 | 0.981 ± 0.136 | 0.432 ± 0.113 | 0.131 ± 0.019 |
| DPLR core r8/p2 | 0.927 ± 0.191 | 0.450 ± 0.125 | 0.080 ± 0.007 |

Early ratios use the same step/block metric as calibration but pair each new
seed to its own reference. DPLR-core seed 55 is outside the early tolerance
at 0.741; rates remain frozen. Every adapter parameter wakes by step 10 and
has nonzero endpoint gradients, so the audit finds no permanently dead
zero-initialized branch. Nevertheless core updates drift lower, and live
magnitudes shrink after the early phase. Final combined/base ratios are
1.015/1.005 for dense/DPLR cores versus 1.316 for scaled basis r8. These raw
model probes and live held-out branch statistics do not measure EMA quality
or the entire network update, and do not establish variance preservation.

Scaled-r8 FID is lower than raw r8 in all three new seeds by
`[0.27, 1.45, 0.67]`, reversing the prior confirmation cohort's FID ordering.
Raw r8 still has slightly higher mean agreement (0.44 points), with differences
`[+0.90, +0.48, -0.06]`. Keep both gain references; the evidence does not select
a universal raw/scaled rule or justify pooling historical cohorts.

Static outer `U0/V0` restrict the core's possible condition-dependent input and
output subspaces. This is a plausible capacity constraint, while later update
drift is a plausible optimization issue. The current study does not separate
those explanations, and scalar calibration alone does not resolve learned
basis gauges. The next useful quality checks are moving-factor generators
with `r=8,s=6/12` and `r=16,s=6` (generator widths near `q/32,q/16`), using fresh
seeds, declared budgets and the same optimization diagnostics. These remain
proposals; this confirmation does not measure their longer generative quality.
Isolated eager/compiled timing remains necessary for any stable speed claim.

Training/validation/diagnostics used 7.035 aggregate GPU-hours and final
generation/evaluation 0.198; calibration/startup are additional. All 21 workers
and the controller exited successfully, private supervisors were closed, and
all workflow GPU claims were released. Approximately 51.5 MiB of compact raw
evidence is retained under the 512 MiB bound, with no new weights/checkpoints,
generated image datasets or official-test evaluation. The audit reproduces
source/configs/counts, initialization hashes, paired traces, calibration
selection and evaluator identity/denominators. Three training seeds, one
sampling seed/held-out split and concurrent timings bound interpretation.

## Revised roadmap: parameter trends and token self-conditioning

User direction after the core confirmation: defer the dedicated isolated
speed/compile benchmarking track. Continue generator parameter search, richer
generators, sequence conditioning and broader operator choices, with token
self-conditioning as a first-class application. Normal training logs and
optimization diagnostics remain useful; no new speed study or training jobs
are launched by this roadmap update.

### Joint rank and generator-width search

The next proposed coarse search crosses transform rank `r=[2,4,8,16]` with
generator widths near `q/[32,16,8,4]`. At `q=192`, this gives `s=[6,12,24,48]`,
with the existing `s=16` rank-2/rank-8 recipes as anchors. This tests width
trends at fixed rank and rank trends at fixed width, rather than selecting a
few disconnected configurations. The exact grid and compute budget must be
frozen in its launch protocol.

Include approximate decoder-budget comparisons such as
`(r,s)=(2,48),(4,24),(8,12),(16,6)`: all have `r*s=96`, so the dominant decoder
term `s*r*(d+o)` is equal. Encoder costs `2*q*s` and biases differ; these are
not exactly equal-total-parameter configurations. Report actual parameter
counts and compare ordinary FFN capacity at relevant shortlisted budgets.

Use a common schedule, placement, initialization convention and declared gain
rule, with at least two seeds in a bounded coarse screen. Inspect learning
curves, rank/width saturation and quality versus parameter count. Retain
functional-update and branch-magnitude diagnostics to expose optimization
confounds. Short denoising MSE alone does not establish a generative-quality
trend. Confirm a small set of promising/trend-defining cases at longer budgets
on fresh seeds; targeted LR/gain sensitivity checks can test whether the
apparent capacity trend survives optimization changes.

### Token self-conditioning: ViT, then a small causal LM

The condition is the token's current contextual hidden state, normally after
pre-norm, rather than its discrete vocabulary ID or a pooled condition:

```python
# x: [B,T,d]; adapter maps d -> o at MLP input.
h = norm_mlp(x)
c = h                         # [B,T,d], one condition per token
delta = adapter(h, c)         # [B,T,o], added beside the static MLP projection
pre_act = mlp_in(h) + delta   # common scale/shift can be retained by the design
```

`vit.py` already passes the normalized token as the condition and exposes a
detached-condition ablation. The LM path in `transformer.py` still needs an
adapter integration. Start the renewed self-conditioning comparison with ViT
MLP-input placement, then adapt a small causal LM. In the LM, each hidden token
may include its allowed prefix through causal attention; conditioning must not
introduce access to future tokens. Retain ordinary self-attention so this
experiment tests token-dependent channel transforms, not attention removal.

Keep common norm/affine settings, placement and base initialization across
controls. The earlier ViT used static LayerNorm affine parameters, unlike the
diffusion setup; changing them is an explicit ablation. Compare ordinary
capacity controls, token-conditioned diagonal scale/shift, static low-rank
updates and dynamic factors. Differentiate through `c=h` by default; compare
`c=h.detach()` selectively to distinguish condition-path gradients while
preserving forward conditioning.

Per-token factors would have shapes `[B,T,d,r]` and `[B,T,r,o]`. Use the
existing basis activation-mixing path, which avoids storing these full factors
and uses basis intermediates shaped `[B,T,s,r]`. This remains a correctness and
feasibility requirement despite deferring dedicated speed benchmarks. Measure
ViT held-out accuracy/loss and LM held-out cross-entropy at declared budgets.
The LM corpus/tokenizer/model size remain to be selected in its protocol;
synthetic tokens can serve as a smoke test, not language-model quality evidence.

### Further generators and sequence-valued conditions

After the basic rank/width trend, introduce nonlinear bottleneck heads and
channel-modulated static factors as separate, affordable ablations. Channel
modulation can move effective outer subspaces through input/output gains, in
contrast to mixing only inside a static core. Further CP/Tucker or block
compression targets the dominant decoder parameters; a shared coefficient
head alone saves only the smaller encoder portion in the current design.
Diagonal-plus-low-rank generator maps need an explicit coordinate/skip design
for rectangular outputs. Preserve zero-delta initialization and check gradient
wake-up for each new parameterization.

Sequence-valued external conditioning remains a separate track: start a
controlled associative-retrieval pilot with distractors and compare pooled
AdaLN/DPLR, cross-attention, and cross-attention plus DPLR. Then study
query-dependent retrieval feeding token-specific transforms. Token
self-conditioning does not itself replace access to an external sequence.
Use controlled context-access comparisons before attributing retrieval quality
to the channel transform.

Kronecker, block-sparse, Hadamard/FFT-style and tensor-train produced operators
remain later alternatives. Begin with shape/correctness, initialization and
parameter/activation cost checks; prioritize a few feasible candidates for
quality runs. Dedicated isolated speed work stays deferred. Every actual
launch still records its finite compute/storage budget and compact artifact
policy; generic model checkpoints remain off by default.

## Launched rank/width screen and renewed token-conditioning pilot

`runs/rank-width-20261008T084000Z` implements the revised roadmap's first
bounded screen. Its frozen protocol contains 64 runs with paired seeds 58/59:
44 broadcast-conditioned diffusion runs and 20 token-conditioned ViT runs.
The diffusion grid crosses `r=[2,4,8,16]` with `s=[6,12,24,48]`, adds the
rank-2/rank-8 `s=16` anchors, raw-gain rank-8, AdaLN and two FFN capacity
controls. Gain is `1/sqrt(r)` for dynamic factors except the raw control;
generator LR multiplier stays at one. Width-dependent optimization remains
a possible explanation of quality trends, so fixed-input function updates
and live branch magnitudes are retained alongside quality metrics.

Diffusion uses the full 45k/5k CIFAR-10 split and 5,000 steps of the existing
30k schedule, with 5k-image EMA FID/class-agreement evaluation. ViT uses the
full split for ten epochs, comparing plain, diagonal, static rank-8, five
dynamic rank/width choices, a detached-condition control and an approximate
FFN parameter match. Its condition is `c=norm_mlp(x)` per token, with ordinary
self-attention retained; activation mixing avoids materialized per-token U/V.
Static LayerNorm affine parameters remain common across the ViT cases.

Before the corrected launch, CPU checks passed (61 passed, two GPU tests skipped), claimed
GPU BF16 checks passed (three passed), and all 32 family configurations were
serialized, reconstructed and checked for parameter counts and shared base
initialization. The two matched controls differ from their targets by 384
parameters in diffusion and 216 in ViT. Diagnostics were checked to preserve
the training trajectory and RNG, including early gradient wake-up.

The first attempt, `runs/rank-width-20261008T082627Z`, failed during final
evaluation after six workers had trained: the classifier's source guard
compared its old implementation with the updated active ViT trainer. Its
source/logs remain intact and both private supervisors are closed. The corrected
evaluator verifies and loads the classifier's original frozen source in isolated
modules, restoring the active trainer modules afterward. A CPU regression
checks identical logits and rejects changed evaluator source; a claimed GPU
smoke exercised the real 5k-image evaluator with 20 generated images. That smoke
is compatibility evidence only. The new workflow repeats the fixed screen and
records the failed attempt; its compute estimate includes this recovery.

The declared budget is 4–7 aggregate GPU-hours plus startup, six concurrent
GPU-claimed workers, and at most 512 MiB of compact evidence retained until
explicitly removed. No new weights, checkpoints, evaluator weights, image
grids or official-test evaluation are enabled. Dedicated speed benchmarks
and automatic longer confirmation remain deferred. All 64 runs completed on
2026-10-08 at 09:23:32 UTC; the independent audit passed, both private
supervisors are closed and this workflow's GPU claims are released.

### Completed screen: trends, magnitude and next comparisons

[The report](reports/rank-width-search.md) contains all 32 two-seed family
summaries, plots, whole-training memory, paired differences and function-update
diagnostics; [the evidence](reports/rank-width-search-evidence.json) retains
each run's full results/config. Means and sample SD describe two seeds; the
5k-step diffusion prefix is separate from the earlier 30k confirmations.

On the approximately equal decoder-budget contour, the outcomes are:

| Transform rank / generator width | Added parameters | FID | Requested-class agreement | Training peak MiB |
| --- | ---: | ---: | ---: | ---: |
| r2/s48 | 664,128 | 141.28 ± 1.90 | 27.05 ± 0.61% | 609.8 |
| r4/s24 | 608,544 | 132.10 ± 1.22 | 28.55 ± 0.58% | 611.3 |
| r8/s12 | 580,752 | 128.78 ± 3.00 | 27.76 ± 0.42% | 616.8 |
| r16/s6 | 566,856 | 127.27 ± 0.66 | 26.16 ± 0.48% | 625.5 |

Higher transform rank improves mean FID here while reducing the smaller
encoder/bias budget, but class agreement favors middle ranks and allocation
increases. This supports considering higher rank with a narrower generator;
it does not establish that one point preserves every metric. Rank-8/s16
remains the common anchor: 126.18 ± 0.55 FID and 28.41 ± 0.27% agreement.
At fixed rank 8, widths 6/12/16 give mean FID 133.82/128.78/126.18; widths
24/48 give 126.94/125.55. More width is not a steady improvement, especially
at smaller ranks.

Rank-16/s24 gives FID 120.19 ± 0.38 and agreement 32.03 ± 0.81%, adding
2,267,424 parameters with 657.6 MiB training allocation. Rank-16/s48 adds
4,534,848 parameters and uses 699.3 MiB, giving FID 118.28 ± 21.56 and
agreement 33.62 ± 0.28%. Its FIDs 103.03/133.52 show why a mean-only ranking
would be misleading. Longer fresh-seed confirmation is needed for a quality
frontier; the AdaLN baseline itself has large FID spread at this short budget.

ViT token conditioning gives plain accuracy 67.17 ± 0.21%, diagonal
69.31 ± 1.65%, rank-8/s12 69.66 ± 0.54%, rank-8/s16 69.75 ± 1.29%, and
rank-16/s6 69.51 ± 0.18%. Rank-8/s16 improves on diagonal by +0.18/+0.70
points in paired seeds, but costs another 774,336 parameters and 92.5 MiB
allocation (603.9 versus 511.4 MiB). Its small mean advantage does not yet
pay off decisively. The rank-8/s12 detached control gives 67.30 ± 0.31%,
versus 69.66 ± 0.54% with condition gradients: paired gains +1.76/+2.96
points. Keep the full-gradient default and retain detached/diagonal controls
when moving to other data or a causal LM. Self-attention remains present.

Optimization and magnitude are material confounds. For diffusion rank 8,
early function updates at s6/s48 are 0.423 ± 0.021 / 2.794 ± 0.007 times
the paired s16 reference despite common rank gain/LR. Final live delta/base
grows from 0.388 at r2/s6 to 1.056 at r16/s48. The transform is a zero delta
at initialization, but that and `gain=1/sqrt(r)` do not ensure later variance
preservation or comparable updates. These results warrant targeted gain/LR
sensitivity before attributing generator-width trends entirely to capacity.
DiT probes measure individual optimizer steps; ViT probes measure intervals
between diagnostic records. Their ratios are compared within each regime.

The FFN controls underperform their plain/AdaLN baselines under these short
fixed recipes. Their hidden shapes and base initialization differ. Their
relative weakness is evidence about this recipe, not a demonstration against
fully optimized ordinary capacity. No isolated speed conclusion is drawn.

Next comparisons should retain scaled r8/s16, contrast r8/s12 with r16/s6
at similar decoder budget, and include a larger r16/s24 point only when its
parameter budget serves the question. The renewed token-conditioning path
should keep diagonal and full/detached controls. Richer heads, causal LM
integration, sequence retrieval and structured operators remain subsequent
tracks; this screen launches none of them or any longer confirmation.

Successful loops used 3.451 aggregate GPU-hours and final diffusion evaluation
0.439. The failed first attempt used another 0.417 claimed GPU-hours, including
startup/failed evaluation; other startup and the small compatibility smoke
are additional. Raw evidence was 68.3 MiB at export, below the 512 MiB bound.
The full audit reconstructs all configs/counts/initialization, checks common
splits and paired traces, gradient wake-up, evaluator denominators, GPU claims,
successful exits and artifact policy. It also matches the six repeated
training configs and batch/noise traces to the preserved failed attempt.
Training source stays frozen; corrected analysis source is separately retained
and hashed by [the artifact manifest](reports/rank-width-search-manifest.json).

## Targeted rank/width gain and LR confirmation

Following authorization to proceed, `runs/dit-width-20261008T165955Z` tests
the narrower generators r8/s12 and r16/s6 against the scaled r8/s16 anchor.
The seven quality recipes are the anchor and three variants per narrow shape:
default `gain=1/sqrt(r), adapter_lr_multiplier=1`, that gain with an
update-matched LR, and `gain=(16/s)/sqrt(r)` with an update-matched LR.
The gain alternative is an empirical width-compensation heuristic. It is
neither a variance-preservation rule nor a μP prescription.

Calibration uses independent seeds 60/61 and the full 45k loader for 500 steps
of the same 30k schedule (base LR 4e-4, warmup 50). The metric is arithmetic
mean highest-FP32 fixed-input single-step low-rank function-update RMS across
six blocks, steps 50/100/250/500 and both seeds. It changes the global adapter
LR multiplier, bounded to `[1/16,16]`, by `rate *= target/update`. One rate
response is always verified for each non-anchor family, followed by at most
two further corrections if the most recent candidate misses ±15%. Selection
chooses the observed rate closest in log update ratio to the r8/s16 target;
loss, validation, FID and class agreement do not enter selection. Aggregate
matching does not guarantee matching in each seed/block, later updates or
live branch magnitudes. If the bounded selection misses tolerance, quality
confirmation is withheld.

Once selected, gains and rates freeze before 21 quality runs: seven recipes
times fresh seeds 62/63/64 at 30k steps. The common recipe retains width 192,
depth 6, patch 4, three heads, non-affine norms/AdaLN-Zero, normalized adapter
input, broadcast class/time, BF16, AdamW with zero weight decay, clipping 1,
cosine floor .01 and in-memory EMA .999. Final quality uses the existing
5k-image FID and frozen-classifier evaluators, DDIM50, no CFG, and the same
held-out 45k/5k CIFAR-10 split. Default-rate controls stay at multiplier one.
Gain/rate variants of a shape share adapter initialization; all seven share
base initialization within a seed. The comparison is targeted sensitivity,
not an exhaustive quality hyperparameter search.

Prelaunch evidence records 56 passing CPU tests (one GPU-only test skipped),
serialized/reconstructed configs and shared initialization for every recipe,
compatibility of the generalized audit helper with the previous completed
core study, and a claimed GPU smoke checking the larger-gain r16/s6 recipe,
gradient wake-up and real FID/classifier integration. The smoke uses only
12 steps with warmup 2 and 20 generated images; it is compatibility evidence,
not a quality experiment.

The declared budget is 7–10 aggregate GPU-hours plus startup, six concurrent
GPU-claimed workers and 512 MiB of compact evidence retained until explicitly
removed. There are at most 34 calibration runs and exactly 21 quality runs if
selection passes. Disk capacity was 321 GiB free before launch. No model or
evaluator weights, generic checkpoints, generated datasets or grids are saved.
Dedicated speed work stays deferred. ViT/LM, nonlinear heads, external-sequence
retrieval and structured operators remain later tracks. The completed results
are recorded below.

Calibration completed in 18 runs (ten at multiplier one, eight verified
corrections). Both stages passed independent source/config/initialization,
paired-trace, gradient and hardware checks. Frozen rates are:

| Recipe | Gain | Adapter LR multiplier | Calibration update / target |
| --- | ---: | ---: | ---: |
| r8/s16 anchor | 0.353553 | 1.000000 | 1.000 |
| r8/s12 matched LR | 0.353553 | 1.273942 | 0.942 |
| r8/s12 gain + matched LR | 0.471405 | 0.983142 | 1.013 |
| r16/s6 matched LR | 0.250000 | 2.513860 | 0.879 |
| r16/s6 gain + matched LR | 0.666667 | 1.130930 | 1.009 |

The default controls retain multiplier one and the default gains. Rates were
frozen before fresh-seed confirmation started; all selected aggregate ratios
pass ±15%, but are not exact matches. Quality and later update/magnitude drift
are now available in [the audited report](reports/dit-width-calibration.md).

### Completed gain/LR confirmation: quality and drift

All 21 quality runs completed at 30k steps, with the independent audit passing
for the 18 calibration runs and fresh paired seeds 62/63/64. Values below are
mean ± sample SD, not confidence intervals. These results stay separate from
the prior 5k rank/width screen and historical 30k cohorts.

| Recipe | Adapter parameters | Total parameters | FID ↓ | Class agreement ↑ |
| --- | ---: | ---: | ---: | ---: |
| r8/s16 anchor | 774,336 | 4,954,032 | 62.48 ± 0.55 | 50.43 ± 0.71% |
| r8/s12 default | 580,752 | 4,760,448 | 62.75 ± 1.65 | 49.05 ± 0.31% |
| r8/s12 matched LR | 580,752 | 4,760,448 | 61.68 ± 0.35 | 49.77 ± 0.06% |
| r8/s12 gain + matched LR | 580,752 | 4,760,448 | 62.01 ± 0.64 | 49.25 ± 0.66% |
| r16/s6 default | 566,856 | 4,746,552 | 62.05 ± 1.79 | 46.77 ± 0.66% |
| r16/s6 matched LR | 566,856 | 4,746,552 | 61.38 ± 2.38 | 47.52 ± 0.21% |
| r16/s6 gain + matched LR | 566,856 | 4,746,552 | 62.16 ± 0.61 | 47.19 ± 0.21% |

LR matching improves mean FID relative to the same-shape default by 1.07
for r8/s12 and 0.67 for r16/s6. Each improves FID in two of three seeds;
class agreement improves in every seed, by 0.73/0.75 points on average.
Against the anchor, r8/s12 matched LR improves mean FID by 0.80 but loses
0.65 agreement points. Its paired FID differences are -1.05/-1.43/+0.08,
and agreement differences +0.06/-0.60/-1.42. It saves 25% of adapter
parameters and 3.91% of total parameters, making it an affordable candidate
with a modest conditioning tradeoff under this recipe.

r16/s6 matched LR has the lowest observed mean FID, but its 2.38 SD is large
and agreement is 2.91 points below the anchor, lower in all three seeds.
The near-equal decoder-budget comparison therefore does not establish a
benefit from spending the generator budget on higher transform rank. Its
26.79% adapter reduction is only 4.19% of the whole model. Whole-training
peak allocation is 625.5 MiB versus 620.2 for the anchor and 616.8 for
r8/s12: fewer projection parameters do not guarantee lower trainer memory.
These peaks include EMA/optimizer/gradients/activations and fixed probe buffers;
they exclude diagnostic forwards and final FID. No isolated speed claim follows.

The alternate gain worsens mean FID by 0.33/0.78 and agreement by 0.52/0.33
points relative to LR matching alone for r8/s12 and r16/s6. FID differences
are mixed by seed; r16/s6 agreement declines in every seed. This tests one
gain heuristic, not all possible gains, and does not support adopting it for
quality by default.

The update diagnostics make the optimization distinction concrete:

| Recipe | Fresh early update / anchor | Final update / anchor | Final live delta/base |
| --- | ---: | ---: | ---: |
| r8/s16 anchor | 1.000 | 1.000 | 0.716 |
| r8/s12 default | 0.733 | 0.821 | 0.593 |
| r8/s12 matched LR | 0.885 | 1.298 | 0.592 |
| r8/s12 gain + matched LR | 0.933 | 1.083 | 0.617 |
| r16/s6 default | 0.377 | 0.406 | 0.813 |
| r16/s6 matched LR | 0.847 | 2.100 | 0.796 |
| r16/s6 gain + matched LR | 0.990 | 0.981 | 0.793 |

Calibration tolerance does not guarantee fresh-seed transfer: r16/s6 matched
LR's fresh aggregate is just outside ±15%, and individual r8/s12 seed 64
and r16/s6 seeds 62/64 also miss. Recipes were not retuned. Final updates
are single-step FP32 fixed-probe RMS relative to the same-seed anchor at
30k, not accumulated updates; live ratios use the raw model on fixed held-out
images, while quality uses EMA. The gain alternative keeps r16/s6 updates
closer to the anchor over training, without improving mean quality. LR-only
matching drifts to roughly twice the anchor update despite similar final
live magnitude to the gain variant. This demonstrates that update size and
branch magnitude are different controls. It does not identify a gauge as
the cause or prove variance preservation.

For subsequent diffusion comparisons, retain r8/s16 as the conditioning
reference and r8/s12 with its independently calibrated LR as an affordable
candidate. Keep r16/s6 as a rank/width tradeoff control, rather than selecting
it solely on mean FID. Continue the token-conditioning path with diagonal,
full-gradient and detached-condition controls; recalibrate in that regime
instead of transferring broadcast diffusion rates. Richer/shared/nonlinear
generators, separate factor LRs or gauge controls, external sequence retrieval
and structured operators remain open comparisons. This study launches none
of those follow-ups, and dedicated speed work remains deferred.

Calibration loops used 0.121 aggregate GPU-hours, confirmation loops 7.552,
and final evaluation 0.212, with startup and the compatibility smoke additional.
Raw evidence at audit was 58.1 MiB, below the 512 MiB bound. Frozen sources,
configs/counts, paired batch/noise traces, reproduced initialization, gradient
wake-up, evaluator provenance/denominators and hardware claims passed. All
private supervisors are closed and claims released. No weights/checkpoints,
generated datasets, grids or official-test results were saved.
The [report](reports/dit-width-calibration.md),
[structured results](reports/dit-width-calibration.json),
[full evidence](reports/dit-width-calibration-evidence.json),
[audit](reports/dit-width-calibration-audit.json) and
[manifest](reports/dit-width-calibration-manifest.json) retain the paired
differences, diagnostic trajectories and analysis/source hashes.

## Proposed wider, nonlinear and shared projectors (2026-10-08)

This section records the subsequent design discussion and updates the next
experimental priorities. These projector variants and normalization controls
were proposals when this discussion was recorded. Whole-projector broadcast
sharing and direct SiLU coefficient heads are now implemented and screened
in the subsequent workflow below; the other alternatives remain proposals.
The completed prior evidence remains the linked rank/width and gain/LR studies.

Keep three axes separate: transform rank `r`, generator width `s`, and the
number of independent projector groups `G`. Nonlinearity is a fourth axis.
For an affine generator, `s` bounds each flattened-factor projection's rank.
With a nonlinear encoder, it is a latent/basis width, not a global linear
projection rank; decoded factors still lie in their `s`-dimensional decoder
spans. In both cases `U @ V` has rank at most `r`.

### Wider and nonlinear coefficient heads

The earlier 5k screen gives a reason to revisit wider generators: at rank 8,
s48 has 32.22% mean requested-class agreement versus 28.41% at s16, but
FID changes only from 126.18 to 125.55, with larger seed spread. Those are
two-seed short results with unmatched update scaling, not longer-budget
evidence of a width benefit. The latest calibrated 30k study tests narrower
widths only. Keep current controls in any wider-generator comparison rather
than pooling these cohorts.

```python
# Broadcast condition c: [B,q]; normalized tokens norm_h: [B,T,d].
# coefficient_head = nn.Linear(q, 2*s)
coeff = coefficient_head(c)                  # [B,2*s]
a, b = coeff.chunk(2, dim=-1)                 # current affine coefficients

# Proposed same-parameter nonlinear alternative:
a, b = F.silu(coeff).chunk(2, dim=-1)

# Decoder weights D_U: [s,d*r], D_V: [s,r*o].
U = (a @ D_U).reshape(B, d, r)
V = (b @ D_V).reshape(B, r, o)
delta = gain * ((norm_h @ U) @ V)             # [B,T,o]
```

The two `a,b` assignments describe alternative recipes, not consecutive
operations in the current implementation. SiLU adds no parameters, but changes
coefficient statistics, gradients and condition dependence. The common
class/time encoder already contains a nonlinearity; this adds one to the
block's factor generator. It does not enlarge the decoder span or guarantee
better optimization. Signed coefficients remain allowed.

A richer alternative is `Linear(h,2*s)(SiLU(Linear(q,h)(c)))`. With biases,
its coefficient head has `q*h + h + 2*s*h + 2*s` parameters, compared with
`2*s*(q+1)` for the affine head. At q192/s16/h64 this is 14,432 versus 6,176
per projector: adding 49,536 parameters over six independent layers, for
823,872 adapter parameters total. A shared hidden trunk couples condition
features for the U/V heads; separate nonlinear heads remain another option.
Nonlinearity followed by a free affine coefficient output retains a generator
coordinate gauge through that output and decoder. Direct coefficient SiLU
restricts some affine reparameterizations, but neither design removes the
U/V rank-coordinate gauge.

### Sharing across layers

Share only the low-rank generator components under discussion. Keep the base
attention/MLP, common AdaLN heads, norms and residual gates layer-specific.
Four sharing arrangements have different restrictions:

| Shared components | Layer-specific components | Main consequence |
| --- | --- | --- |
| Nonlinear condition trunk | Coefficient heads and factor decoders | Larger shared feature extractor; distinct factor subspaces per layer |
| Coefficient head | Factor decoders | Same mixtures, different factor subspaces; small parameter saving |
| Factor decoders | Coefficient heads | Common factor subspaces, different condition-dependent mixtures |
| Entire factor generator within a group | Base block and AdaLN/gates | Identical conditional transform within the group for the same condition |

```python
# Shared nonlinear trunk, layer-specific heads/decoders:
z = shared_trunk(c)                          # [B,h], once per forward
for layer in layers:
    a, b = layer.coefficient_head(z).chunk(2, dim=-1)
    U = (a @ layer.D_U).reshape(B, d, r)
    V = (b @ layer.D_V).reshape(B, r, o)
    # Apply in that layer's existing parallel MLP-input delta branch.
```

```python
# Entire-projector sharing, two contiguous groups in the six-layer DiT:
group_factors = [projector(c) for projector in projectors]  # two (U,V) pairs
for layer_index, layer in enumerate(layers):
    U, V = group_factors[layer_index // 3]    # layers 0-2 / layers 3-5
    # delta = gain * ((that_layer_norm_h @ U) @ V)
```

These snippets illustrate placement and tensor reuse, not runnable replacement
trainers. Use one registered module per group; copying parameter values into
independent modules is not sharing. Reuse factors within the current forward
without detaching them, and recompute after optimizer updates. Per-layer gains
would make identical shared transforms differ in amplitude and constitute a
separate variant; the initial comparison uses a common fixed gain.

At q=d192/o768/r8, the affine projector costs
`s*r*(d+o) + 2*s*(q+1) = 8,066*s`, including coefficient biases. Decoders
contain 122,880 of the 129,056 parameters at s16. For the six-layer model:

| Projector groups G | Layers per group | s per projector | Adapter parameters | Total model parameters |
| ---: | ---: | ---: | ---: | ---: |
| 6 independent | 1 | 16 | 774,336 | 4,954,032 |
| 2 shared | 3 | 48 | 774,336 | 4,954,032 |
| 1 shared | 6 | 96 | 774,336 | 4,954,032 |
| 2 shared, fixed-width control | 3 | 16 | 258,112 | 4,437,808 |

The equal-budget rows exchange layer specialization for wider shared decoder
spans; they are exactly parameter matched for this affine design and these
common block dimensions. Sharing only the s16 coefficient head across six
layers saves 30,880 parameters, whereas sharing the s16 decoders with six
independent heads gives 159,936 adapter parameters. Two groups of shared s16
decoders with independent heads give 282,816. These partial-sharing variants
are distinct from the whole-projector rows and remain subsequent candidates.
A shared q192->h64 SiLU trunk plus six h64->2*s16 heads and independent
decoders costs 762,112 adapter parameters. Larger shared nonlinear trunks can
therefore be affordable, but decoder compression is the larger saving lever.

With a common broadcast condition, an entire shared projector returns the
same U/V and transform in each group. Hidden inputs and base weights differ,
but this is a real restriction on conditional layer specialization. A layer
embedding supplied to a shared nonlinear projector can produce different
factors per layer, at the cost of separate factor generation; it still shares
decoder spans. With a purely affine layer-conditioned factor generator,
layer offsets do not change the condition-to-factor slope. Nonlinear
layer-conditioning can change that dependence and is a separate comparison.

Broadcast class/time allows reuse of a group's generated `[B,d,r]` and
`[B,r,o]` factors. At fixed s, fewer groups reduce factor-generation work;
at equal affine budget, `G*s` is fixed, so leading total factor-generation
matmul FLOPs are approximately unchanged. There are fewer factor sets and
different launch/GEMM shapes, while application FLOPs across all six layers
remain the same at fixed r. Actual training peaks and speed require evidence.
No isolated speed work is added to the roadmap by this proposal.

For token self-conditioning, each layer's `c=norm_h` is different. Shared
weights save parameters, but the generator must run again per layer. Keep
activation mixing rather than materializing full per-token factors; sharing
does not automatically reduce the `[B,T,s,r]` intermediates. A wider s96
shared projector could increase those intermediates substantially even with
unchanged total parameter count. Reusing one earlier layer's condition or
factors would change the conditioning architecture and is not implied here.

Shared parameters receive contributions from every layer that uses them.
Those gradients can reinforce or conflict; a group-size LR divisor is not
automatically correct for Adam or clipped training. Recalibrate with fresh
independent seeds and examine per-layer function updates, factor/head norms,
gradient wake-up and aggregate drift. Sharing leaves factor gauges present.
Retain random/nonzero input decoders and coefficient heads, zero output
decoders, and a fixed nonzero branch gain. Zeroing both coefficient outputs
or adding another zero multiplier can block the intended gradient path.
Reproduce common base initialization before attaching differently sized
adapters; same-shape affine/SiLU comparisons can share initial head/decoder
weights, but different-width models cannot share full adapter initialization.

### Magnitude control and normalization remain separate axes

Coefficient RMS normalization operates on `[B,s]` and is relatively small
for broadcast conditioning. It removes coefficient amplitude information;
decoder scaling and U/V rank gauges remain. Normalizing factors separately
constrains their magnitudes but changes the function and optimizer geometry,
introducing scale-invariant directions in the raw parameters. Normalizing
the composed transform is invariant to reparameterizations that preserve
`U @ V`, so it does not fix their internal gauge. A learned outer gain can
trade scale with unnormalized factors; it adds another amplitude coordinate
rather than automatically solving this issue.

For broadcast factors, obtain the transform Frobenius norm without allocating
the dense `[B,d,o]` matrix:

```python
gram_u = U.transpose(-1, -2) @ U             # [B,r,r]
gram_v = V @ V.transpose(-1, -2)             # [B,r,r]
transform_frob_sq = (gram_u * gram_v).sum((-2, -1)).clamp_min(0)
```

These contractions add work proportional to `B*(d+o)*r*r`; the result is
the squared Frobenius norm of U@V, before branch gain. It does not bound
amplification tightly in every direction or establish variance preservation.
This broadcast-factor recipe does not justify materializing factors in the
per-token activation-mixing path.

Hard unit normalization near the zero output factor needs special care:
`sqrt(norm_sq + eps)` is finite but can amplify the first derivatives by
`1/sqrt(eps)`. A proposed smooth cap instead preserves the initial local
scale, using a fixed positive threshold in transform-Frobenius units:

```python
raw_frob_sq = gain**2 * transform_frob_sq
amplitude = torch.rsqrt(1 + raw_frob_sq / tau**2)   # [B], tau > 0 fixed
delta = gain * ((norm_h @ U) @ V)
delta = delta * amplitude[:, None, None]
```

At zero transform, amplitude is one; this caps magnitude without demanding
a nonzero initial transform or adding a learned gain. It is a proposal, not
a validated threshold or variance rule. Scaling depends only on the condition,
so the transform remains linear in norm_h for fixed c. Normalizing the actual
delta using its input-dependent RMS is another option, but generally changes
that linearity and adds reductions over token/output activations. For
self-conditioning, the overall map already depends nonlinearly on the token.

### Revised next comparisons and retained tracks

First compare four affine broadcast recipes at r8: independent s16,
two-group shared s48, fully shared s96, and two-group shared s16. The first
three match total parameters; the fourth measures sharing at fixed width.
Together they help distinguish tying from reallocating its savings to width,
although they do not constitute a full factorial study. Keep the affordable
independent r8/s12 candidate for later comparisons where that budget matters.

Next compare affine and direct-SiLU coefficient heads in the promising sharing
arrangement. The independent r8/s16/s24/s48 x affine/SiLU six-case grid remains
a useful alternative for isolating width/nonlinearity without sharing. Add a
small nonlinear MLP or shared trunk when its separate capacity question is
being tested, not every possible combination at once. Output normalization
or a smooth cap follows as a distinct magnitude-control ablation if drift or
amplification warrants it; it is not a default change.

Each quality comparison needs independently selected update calibration and
frozen recipes before fresh paired confirmation, with same-split/schedule
controls, common base initialization, per-layer and aggregate diagnostics,
FID and class agreement, and clearly scoped memory measurements. Early
scalar matching remains approximate and does not guarantee later matching.
This is an experiment design, not a new approved launch protocol: exact run
counts, fresh seeds, budgets and artifact purposes will be recorded before
launch. Keep checkpoints disabled unless a concrete retention need is stated.
Dedicated speed work stays deferred. ViT diagonal/full-gradient/detached
controls, subsequent causal LM work, external sequence conditioning and
structured produced operators remain on the broader roadmap.

## Shared and nonlinear projector screen: implementation and launch

Following authorization to proceed, `runs/dit-projectors-20261008T205312Z`
implements six broadcast DiT recipes at rank8 and fixed gain `1/sqrt(8)`:

| Recipe | Projector groups | s | Coefficient activation | Adapter parameters |
| --- | ---: | ---: | --- | ---: |
| Independent affine reference | 6 | 16 | linear | 774,336 |
| Independent nonlinear | 6 | 16 | SiLU | 774,336 |
| First/second-half shared affine | 2 | 48 | linear | 774,336 |
| First/second-half shared nonlinear | 2 | 48 | SiLU | 774,336 |
| Fully shared affine | 1 | 96 | linear | 774,336 |
| First/second-half fixed-width control | 2 | 16 | linear | 258,112 |

`train_dit_cifar10.py` exposes `projector_groups` (zero means the original
independent path; a positive group count must divide depth) and
`coefficient_activation=linear|silu`. Shared/nonlinear options currently require
the DPLR variant. The common base is initialized before adapters are attached.
Each group owns one unique adapter object referenced by its contiguous blocks.
Named parameters, optimizer groups, EMA updates and statistics count unique
tensors; generated factors are reused within a shared forward without detach.
The independent path retains its previous computation order.

Layer-local identity observation points prevent shared-module hooks from
overwriting another layer's branch diagnostic. Fixed function-update records
remain indexed by layer; shared layers repeat one function on the same probe.
Additional untimed diagnostics retain gradients per projector group and
input/output factor RMS plus transform Frobenius magnitude on independent
normalized conditions. They change no training normalization or output gain.
The FLOP ledger counts generation per group and application per layer.

Prelaunch checks passed 78 targeted CPU tests, covering factor-generation
reuse, shared gradients equal to summed untied-layer gradients, layer-local
branch observations, unique optimizer parameters, EMA aliasing/single updates,
state roundtrip, gradient wake-up, unchanged training with diagnostics and
exact short-prefix behavior. A separate comparison with the preceding frozen
independent implementation matches parameters, nonzero outputs and gradients
exactly. All six serialized quality configs validate and share base initialization;
affine/SiLU pairs share adapter initialization and parameter count. Two claimed
GPU smokes cover two-group SiLU and fully shared affine, real-data BF16 training,
gradient wake-up, EMA and the existing FID/classifier evaluators. Their 12 steps,
warmup2 and 20 generated images check compatibility only. They retain 1.35MiB
of compact evidence under a declared 30MiB bound, with no weights.

Calibration uses fresh seeds65/66 and the full45k loader for 500 steps of the
30k schedule (warmup50/baseLR4e-4), averaging highest-FP32 single-step function
updates over six layer probes, steps50/100/250/500 and both seeds. Sharing
weights groups by their layer counts. The independent affine reference stays
at LR multiplier1. Each of the other five families verifies one rate correction,
then at most two further corrections if needed; multipliers stay within
`[1/16,16]`. Closest-log-update selection excludes all quality metrics and
requires the selected aggregate within ±15%; otherwise quality is withheld.
There are 22–42 calibration runs, depending on responses.

Selected rates freeze before twelve quality runs: six recipes x fresh seeds67/68,
5k steps of the same 30k schedule. All use width192/depth6/heads3/patch4,
batch128, full45k/5k CIFAR-10 split123, normalized adapter input, non-affine
AdaLN-Zero, AdamW(.9,.999), weightdecay0, clip1, BF16 and in-memory EMA.999.
Final EMA evaluation uses the existing 5k-image FID and frozen-classifier
evaluators, DDIM50/noCFG and sampling seed2028. This is a two-seed screen;
no automatic longer confirmation follows.

The declared launch estimate is 1.5–3 aggregate GPU-hours including bounded
calibration, validation, diagnostics and final evaluation, with startup and
compatibility smokes additional. At most six GPU-claimed workers run concurrently.
Storage is bounded to 512MiB of compact sources/configs/seeds/metrics/logs/audits/
figures, retained until explicitly removed; disk had about319GiB free before
launch. No checkpoints, model/evaluator weights, grids, generated datasets or
official-test evaluation are enabled. Dedicated speed work stays deferred.
Calibration and the twelve quality runs are complete and audited. Larger MLP/shared trunks, partial
decoder sharing, layer embeddings, output normalization, ViT/LM/retrieval and
structured operators remain subsequent comparisons.

Calibration completed in 22 runs and both stages passed independent audits.
All five non-reference families selected their verified first correction:

| Recipe | Frozen adapter LR multiplier | Calibration update / reference |
| --- | ---: | ---: |
| Independent affine | 1.000000 | 1.000 |
| Independent SiLU | 3.038430 | 1.095 |
| Two-group s48 affine | 0.353904 | 1.114 |
| Two-group s48 SiLU | 0.953765 | 0.971 |
| Fully shared s96 affine | 0.214559 | 1.053 |
| Two-group s16 affine | 1.319520 | 0.957 |

At multiplier1, initial ratios were 0.329 for independent SiLU, 2.826 for
two-group s48 affine, 1.048 for two-group s48 SiLU, 4.661 for fully shared
s96 affine and 0.758 for two-group s16 affine. These are optimization
differences under this initialization, not quality or speed evidence. All
selected aggregates pass ±15%, but are not exact matches.

### Completed screen: quality, optimization and next comparisons

The full workflow passed its source/config/selection/initialization/diagnostic/
evaluation/artifact audit: 22 calibration runs and twelve fresh quality runs.
Values below are mean ± sample SD over seeds67/68, not confidence intervals.
This is a 5k-step screen of a 30k schedule; keep it separate from earlier 30k
cohorts and do not interpret it as statistical or longer-run confirmation.

| Recipe | FID (lower is better) | Class agreement | Training peak MiB |
| --- | ---: | ---: | ---: |
| Independent s16 affine | 124.12 ± 3.34 | 28.58 ± 0.14% | 620.2 |
| Independent s16 SiLU | 127.22 ± 0.19 | 28.62 ± 0.42% | 620.2 |
| Two-group s48 affine | 131.96 ± 6.33 | 27.65 ± 1.00% | 616.1 |
| Two-group s48 SiLU | 134.34 ± 14.56 | 28.71 ± 0.61% | 616.2 |
| Fully shared s96 affine | 136.49 ± 3.91 | 26.59 ± 0.18% | 615.3 |
| Two-group s16 affine | 132.89 ± 1.29 | 27.62 ± 0.23% | 606.1 |

Independent affine has the best observed mean FID in this screen. Both
equal-budget shared affine recipes worsen FID and agreement in each paired
seed. Wider coefficient generators did not compensate for tying the complete
conditional transform across layers. The fixed-width two-group s16 control
saves 66.67% of adapter parameters and 10.42% of total parameters, at a mean
FID cost of 8.77 and agreement cost of 0.96 percentage points. Its whole-training
peak falls only 2.27%; equal-budget sharing saves less than 1% of that peak.
These peaks include EMA, optimizer state, gradients, activations and fixed
probes; diagnostic forwards and final FID evaluation are excluded. Factor
reuse is implemented, but concurrent eager runs do not establish speed gains.

Independent SiLU worsens FID in both seeds (+0.87/+5.32), with mixed agreement
effects. Two-group SiLU improves agreement in both seeds (+1.34/+0.78 percentage
points), while FID changes +17.15/-12.40. Its large FID spread prevents a clear
joint-quality conclusion. Direct SiLU leaves the decoder span unchanged;
small nonlinear MLPs and shared trunks remain different, untested hypotheses.

Calibration passes do not guarantee matching on fresh seeds or later steps.
Fresh early mean update/reference is 0.962 for independent SiLU, but its final
ratio rises to 2.839 ± 1.148. Its final live delta/base ratio is smaller than
the affine reference (0.656 versus 0.712), while fixed-condition transform
Frobenius magnitude is larger (4.368 versus 2.443). Fully shared affine retains
a final update ratio near 0.990 yet loses quality. Two-group SiLU's fresh early
mean ratio is 0.768 despite passing calibration; recipes were not retuned.
Update matching, operator magnitude, live branch magnitude and generated quality
measure different properties. These observations neither identify a gauge
cause nor prove variance preservation. Future operator diagnostics should also
use actual class/time conditions alongside independent normalized probes.

Retain independent affine as the current reference. The next proposed designs
should preserve layer specialization: shared decoder bases with independent
coefficient heads, or a shared nonlinear trunk with independent decoders, plus
a small MLP comparison. These isolate which sharing is affordable without tying
the entire conditional transform. Output normalization or a smooth cap remains
a separate magnitude-control experiment. No additional jobs or automatic 30k
confirmation are launched from this screen; dedicated speed work remains deferred.
Token self-conditioning retains its diagonal/full-gradient/detached controls and
requires its own calibration because conditions vary across layers and tokens.

Measured aggregate GPU-hours are 0.138 calibration, 0.697 quality training and
0.114 final evaluation, with startup and compatibility smokes additional. Raw
evidence at audit occupies 50.7MiB under the declared 512MiB storage bound.
All private supervisors are closed and workflow GPU claims released; no weights,
checkpoints, grids or generated datasets were retained.

[Report and figures](reports/dit-projector-screen.md) ·
[Structured results](reports/dit-projector-screen.json) ·
[Full evidence](reports/dit-projector-screen-evidence.json) ·
[Audit](reports/dit-projector-screen-audit.json) ·
[Artifact hashes](reports/dit-projector-screen-manifest.json).

## Shared bases with layer heads and small MLPs: two-GPU screen

The user requested proceeding with at most two GPUs. This cap applies to
compatibility checks, calibration, training and final evaluation in the new
workflow `runs/dit-basis-heads-20261008T214617Z`; it also remains the working
preference for subsequent experiments unless changed explicitly.

Whole-projector tying lost quality in the previous short screen. This study
instead ties only decoder bases, keeping six independent coefficient heads.
For layer `ell`, with broadcast condition `c: [B,192]`, the affine design is:

```python
coeff = head[ell](c)                 # [B, 2*s]; independent per layer
alpha, beta = coeff.chunk(2, dim=-1) # each [B,s]
U = einsum('bs,dsr->bdr', alpha, U_basis[group(ell)]) # [B,192,8]
V = einsum('bs,sro->bro', beta, V_basis[group(ell)])  # [B,8,768]
delta = (norm_h @ U @ V) / sqrt(8)   # [B,T,768]
pre_act = mlp_in(scale_shift_h) + delta
```

`U_basis: [192,s,8]` and `V_basis: [s,8,768]` are shared within contiguous
layer groups. The heads and generated transforms remain layer-specific;
therefore factors are generated for each layer, without the complete-transform
reuse of the previous study. Base attention/MLP, AdaLN scale/shift and gates
remain independent. Non-affine LayerNorm and the normalized parallel adapter
input are unchanged. Zero `V_basis` with nonzero `U_basis` and heads retains
an exact initial zero delta; common AdaLN-Zero gates also start at zero.

The small-MLP pair keeps independent s16 bases and heads, with hidden width48:

```python
hidden = linear_in(c)               # [B,48]
hidden = hidden                     # linear control
# or: hidden = silu(hidden)         # nonlinear comparison
coeff = linear_out(hidden)          # [B,32]; no final SiLU
```

Both MLPs have exactly the same tensors, initialization and parameter counts.
The hidden-linear control is a factorized affine projection; hidden48 exceeds
the32 output coefficients, so it can represent the same affine coefficient
function class as the reference. Its optimization and factor gauges may differ
from a single affine head. Hidden width48, basis width s and produced transform
rank8 are distinct dimensions. This control isolates
hidden nonlinearity from the extra projection/factorization. It does not prove
that all possible nonlinear generators or normalization designs behave alike.

| Recipe | Decoder groups | s | Head | Adapter / total parameters |
| --- | ---: | ---: | --- | ---: |
| Independent reference | 6 | 16 | affine | 774,336 / 4,954,032 |
| Two shared bases, fixed width | 2 | 16 | independent affine | 282,816 / 4,462,512 |
| Two shared bases, wider | 2 | 44 | independent affine | 777,744 / 4,957,440 |
| Fully shared bases, wider | 1 | 77 | independent affine | 769,692 / 4,949,388 |
| Independent bases, linear MLP | 6 | 16 | hidden48 linear | 802,272 / 4,981,968 |
| Independent bases, SiLU MLP | 6 | 16 | hidden48 SiLU | 802,272 / 4,981,968 |

The wider shared-basis recipes match whole-model counts within0.1%, rather
than exactly: +3,408/-4,644 parameters versus the reference. The two-group s16
control saves491,520 parameters (9.92% of the reference whole model). The MLP
pair adds27,936 parameters (0.56%); comparisons with the reference retain this
small budget difference. These are not equal-FLOP or equal-time comparisons.

`decoder_groups` describes U/V tying with independent adapter modules;
`projector_groups` retains complete-module tying for the previous designs.
These tying modes cannot be combined here. `coefficient_hidden=0` keeps the
old head and activation behavior exactly; positive width applies the selected
activation between two affine projections. Optimizer, EMA and statistics
count unique parameter tensors. Per-adapter gradient records overlap shared
decoders; separate decoder/head records describe unique ownership. Fixed
function updates and branch observations remain layer-specific.

Prelaunch validation passed108 targeted CPU tests, including shared-decoder
versus untied summed gradients, distinct transforms from independent heads,
explicit MLP activation placement, unique optimizer budgets, EMA/state aliases,
invalid configurations and unchanged training with diagnostics. A separate
comparison against the previous frozen implementation matches default
parameter states, nonzero outputs and gradients exactly, including the older
whole-sharing/SiLU paths. All six serialized quality configs validate and
share base initialization. Two claimed GPU compatibility checks passed shared-decoder aliasing, gradient wake-up, hidden SiLU heads, BF16, EMA and existing evaluators. They retain1.78MiB of compact evidence with no weights.

Calibration uses fresh seeds70/71, the full45k loader and500 steps of the30k
schedule, with the unchanged highest-FP32 fixed-function update metric across
six layers, steps50/100/250/500 and both seeds. The affine reference keeps
LR multiplier1; each other family verifies a first correction and at most two
additional corrections within[1/16,16]. Selection uses closest log update to
the reference and excludes loss/validation/FID/agreement. Selected aggregates
must pass±15% before quality. Rates then freeze for fresh paired seeds72/73:
six recipes, each5k steps of the same30k schedule. There are22–42 calibration
runs and twelve quality runs; failed calibration withholds quality.

Full CIFAR-10 split123, width192/depth6/heads3/patch4, rank8, gain1/sqrt(8),
batch128, warmup50/baseLR4e-4, AdamW(.9,.999), weightdecay0, clip1, BF16 and
in-memory EMA.999 remain common. Final EMA FID/class agreement use5k balanced
real/generated images, DDIM50/noCFG, sampling seed2028 and existing frozen
evaluators. Diagnostics include function updates, live branch ratios, factor
magnitudes and gradient ownership. Scalar early matching can drift; raw-model
operator norms do not establish variance preservation or EMA quality.

Storage is bounded to512MiB of compact sources/configs/seeds/metrics/logs/audits/
plots, retained until explicitly removed. Compatibility checks reserve at most
30MiB within that bound. Disk had about319GiB free before preparation.
Estimated work is1.5–3 aggregate GPU-hours with startup and compatibility
checks additional, at most two claimed workers. No checkpoints, new model or
evaluator weights, grids, generated datasets or official-test evaluation are
saved. Dedicated speed work and automatic longer confirmation remain disabled.
Shared nonlinear trunks, output normalization and new token-conditioned runs
remain subsequent designs. The screen is complete; findings and audit links follow.

Calibration completed in26 runs; all three stages passed independent audits.
The first correction sufficed for fixed-width/two-group wider bases and the
linear MLP. Fully shared bases and SiLU MLP required a verified second
correction. All selected aggregates pass±15%; all twelve fresh quality runs
are now complete. Rates were frozen before quality and were not retuned.

| Recipe | Frozen adapter LR multiplier | Calibration update / reference |
| --- | ---: | ---: |
| Independent affine s16 | 1.000000 | 1.000 |
| Two shared bases s16 | 1.041632 | 0.981 |
| Two shared bases s44 | 0.417542 | 1.133 |
| Fully shared bases s77 | 0.225863 | 1.017 |
| Independent linear MLP48 | 1.731733 | 1.009 |
| Independent SiLU MLP48 | 3.504777 | 0.961 |

### Completed two-GPU screen: findings and next comparisons

All26 calibration and twelve quality runs completed successfully. The full
source/config/initialization/selection/trace/gradient/evaluator/concurrency/
artifact audit passed. Values below are mean ± sample SD over fresh seeds72/73,
not confidence intervals. These5k-step results are a separate cohort from the
previous screen; compare current paired controls rather than pooling old seeds.

| Recipe | FID (lower is better) | Class agreement | Training peak MiB |
| --- | ---: | ---: | ---: |
| Independent affine s16 | 133.67 ± 5.35 | 27.96 ± 0.96% | 620.2 |
| Two shared bases s16 / independent heads | 138.72 ± 7.61 | 25.39 ± 0.10% | 612.6 |
| Two shared bases s44 / independent heads | 140.12 ± 1.22 | 25.26 ± 0.20% | 624.4 |
| Fully shared bases s77 / independent heads | 144.25 ± 0.66 | 23.72 ± 0.25% | 628.1 |
| Independent s16 / linear MLP48 | 131.60 ± 4.26 | 28.72 ± 0.06% | 620.7 |
| Independent s16 / SiLU MLP48 | 128.93 ± 4.72 | 27.18 ± 1.05% | 620.8 |

Independent heads do not recover the quality loss from tying decoder bases in
these wider, nearly parameter-matched recipes. Two-group s44 worsens FID by
+1.80/+11.09 and agreement by-2.16/-3.24 percentage points versus the reference
in seeds72/73. Fully shared s77 worsens FID by+7.26/+13.90 and agreement by
-3.74/-4.74 points. The s16 fixed-width control has mixed FID (-4.11/+14.21)
and lower agreement in both seeds (-1.82/-3.32). Increasing its width to s44
has mixed FID effects (+5.91/-3.12), without a consistent agreement gain.
These results do not establish that all partial sharing is ineffective; a
shared nonlinear trunk with independent decoders remains untested.

The s16 shared-base control saves9.92% of total and63.48% of adapter parameters,
but only1.23% of whole-training peak memory. Nearly parameter-matched s44/s77
instead increase that peak by about0.67%/1.27%. Each layer still generates its
own factors; wider heads/factor-generation work and static decoder savings are
separate costs. The MLP pair adds0.56% of total parameters with about0.1% peak
memory overhead. Peaks include EMA, optimizer state, gradients, activations and
fixed probes; diagnostic forwards and final FID are excluded. Concurrent eager
telemetry does not establish a speed advantage.

The matched MLP comparison gives a clearer tradeoff: hidden SiLU lowers FID
versus hidden-linear by-3.00/-2.34 in both seeds, while agreement drops
-2.24/-0.84 points. Its mean FID is the best observed here, but it is not a
joint-quality winner. Versus the affine reference, SiLU's FID changes
-11.86/+2.38 and agreement falls-0.84/-0.72. Linear MLP versus the reference
has mixed FID (-8.86/+4.72) and higher agreement in both seeds (+1.40/+0.12).
The MLPs have exactly equal parameters and initialization, with independently
update-calibrated rates; they are not equal-LR comparisons. Their small extra
budget relative to the affine reference and imperfect update matching limit
claims about a pure expressivity benefit.

Both MLPs miss the fresh-seed early matching tolerance despite passing
calibration. Linear MLP averages0.805× the paired affine reference (per seed
0.847/0.763); SiLU averages0.632× (0.630/0.634). At step5k, linear is
1.106 ±0.043× and SiLU2.663 ±1.676×. SiLU's live delta/base RMS is0.665,
near the reference0.647, and its fixed-condition transform Frobenius magnitude
is2.015 versus2.326. Larger late function updates therefore do not establish
larger live branch magnitude or variance explosion. Shared-base final update
ratios0.922/1.029/0.861 are much closer to the reference, yet their quality is
worse. Scalar calibration, condition use and generated quality remain distinct;
no gauge cause or variance-preservation claim is established.

Keep independent affine as the reference and retain both MLP controls for the
next comparison. Before a longer quality confirmation, strengthen the update
transfer check across more calibration seeds and add factor/operator magnitude
probes on actual learned class/time conditions alongside independent normalized
conditions. That would clarify whether the MLP tradeoff persists when early
updates transfer more reliably. An output cap or normalization should remain a
separate ablation with unchanged zero initialization. The nonlinear MLP is a
candidate for better FID, with conditional accuracy as a constraint; the linear
MLP is a candidate for improved agreement. Shared decoder bases are a lower
priority under this recipe. No additional jobs or automatic30k confirmation
were launched; token self-conditioning retains its own calibration and controls.

Measured aggregate GPU-hours are0.173 calibration,0.734 quality training and
0.122 final evaluation, with startup and compatibility checks additional.
Raw evidence at audit occupies74.2MiB, within the declared512MiB bound.
All private supervisors are closed and claims released. The audit verifies
at most two concurrent GPU workers; no checkpoints, model/evaluator weights,
grids or generated datasets were retained. Keep the two-GPU cap for subsequent
work unless the user explicitly changes it.

[Report and figures](reports/dit-basis-head-screen.md) ·
[Structured results](reports/dit-basis-head-screen.json) ·
[Full evidence](reports/dit-basis-head-screen-evidence.json) ·
[Audit](reports/dit-basis-head-screen-audit.json) ·
[Artifact hashes](reports/dit-basis-head-screen-manifest.json).

## Convergence and training-setup caution (2026-10-09)

The user cautioned that FID around130 at5k steps may not reflect converged
quality. Interpret the projector screens as early-training comparisons;
their rank order and FID/conditional-agreement tradeoffs may change later.
Shared-base losses at5k alone do not establish lower asymptotic capacity.
Before further broad architecture search, establish a more informative
baseline learning curve and sampling control. Keep the two-GPU cap.

The latest screen is a5k prefix of a30k cosine schedule, not a complete5k
schedule. Base LR4e-4 remains3.73904e-4 at step5k (93.48% of peak); warmup is
50 steps and the eventual floor is4e-6 at30k. Both reference runs processed
639,216 images, about14.20 passes over the45k training split. The model is a
small pixel-space epsilon predictor: width192/depth6/heads3, patch4 (64 image
tokens), MLP ratio4 and about4.95M reference parameters. AdamW(.9,.999),
weightdecay0, global clipping1, BF16, horizontal flips only, RGB[-1,1], uniform
1000 timesteps and squared-cosine betas are common. No dropout or stochastic
depth is configured. Architecture and base training hyperparameters have not
been exhaustively optimized; update-selected adapter rates are not quality-
optimized rates. No logged reference/MLP gradient norms exceeded the clipping
threshold at the eleven training log points; this does not audit every step.

EMA decay.999 has a half-life of about693 steps. At2.5k the reference's mean
raw/EMA held-out epsilon MSE is0.07227/0.08328, so EMA clearly lags early.
At5k it is0.06807/0.06749; EMA lag alone is not an established explanation for
high final FID. The initial-state EMA weight at5k is only0.00672. Fixed-noise
validation averages uniform timesteps; reference EMA MSE at5k is about0.311
in the lowest-noise decile and0.0015 in the highest-noise decile. Retain the
per-timestep errors instead of equating aggregate epsilon MSE with sample
quality or interpreting low-noise loss as an absolute quality target.

An earlier, separate three-seed30k r8/s16 reference study reached FID
62.48±0.55 and class agreement50.43±0.71%. Its mean EMA epsilon MSE continued
from0.06763 at5k to0.06338 at10k,0.06115 at20k and0.06071 at30k. This supports
undertraining as a material limitation of the5k screen. It is not a measured
FID130-to62 trajectory of the same seeds: earlier runs logged final FID only.
Even30k/FID62 is not proof of saturation or a tuned diffusion baseline.

Current generation uses EMA, deterministic DDIM50 (`eta=0`), inherited
`leading` timestep spacing (980 down to0), predicted-clean-image clipping and
no classifier-free guidance. FID uses5k real/generated images, 2048-dimensional
Inception features and the held-out training split, with balanced class requests.
These choices affect interpretation; their separate effects on this model's
FID are unmeasured. Do not attribute FID130 primarily to estimator bias or the
sampler without controls, or compare it directly with published50k-image FID.

Proposed next priority: an independent affine reference and plain AdaLN-Zero
under the same sufficiently long schedule, measuring intermediate FID/class
agreement at selected states (e.g.5k/10k/20k/30k) plus raw/EMA validation and
per-timestep errors. Evaluate a small declared visual diagnostic grid and
DDIM50 versus a higher sampling budget on the same in-memory state; keep raw
versus EMA sampling as a separate control. A narrow baseline LR comparison
can follow if the curve or noise-band diagnostics warrant it. Once a useful
quality/convergence regime is established, compare the affine/linear-MLP/SiLU-
MLP candidates there instead of selecting them solely from early FID.
Intermediate evaluation can run in process and needs no saved weights. Any
future launch still requires a concrete recorded protocol, resource estimate
and artifact purpose/retention; no new jobs are launched by this note.

[Latest5k screen](reports/dit-basis-head-screen.md) ·
[Separate30k reference evidence](reports/dit-width-calibration.md) ·
[Earlier plain-AdaLN comparison](reports/dit-rank2-confirmation.md).

## Optimizer, compilation and CFG controls (2026-10-09)

The user proposed beta1=.95, fused Adam, model compilation and class guidance
at1.5/2.0, and asked whether the model is small. These are now opt-in trainer
settings. Historical frozen experiments retain their original settings and
sources. This implementation does not launch another quality sweep.

The width192/depth6/heads3 r8/s16 reference has4,954,032 parameters. Enabling
CFG training adds a192-element learned null-class embedding:4,954,224 total.
For the same patch4, MLP ratio4, independent r8/s16 adapters and CFG support:

| Width | Depth | Heads | Total parameters |
| --- | --- | --- | --- |
|192|6|3|4,954,224|
|256|8|4|11,135,280|
|384|8|6|23,928,112|
|384|12|6|35,600,304|

These are capacity options, not measured quality improvements. Measure model
size independently from optimizer and sampling changes. A larger plain
AdaLN-Zero control is necessary before attributing larger-model gains to DPLR.
The previous width240 plain model was6.50M and reached FID61.59 at30k, comparable
to the smaller r8/s16 DPLR cohort; see the earlier rank-two confirmation report.

`adam_beta1` and `adam_beta2` default to.9/.999. Setting beta1=.95 doubles the
rough first-moment averaging horizon from10 to20 steps. It may help or hurt;
retain a.9 control and log actual gradients and parameter updates. Existing
adapter-LR calibration under.9 does not automatically transfer to.95.
`fused_adamw` defaults totrue on CUDA and false on CPU. Fused AdamW was already
used by the historical CUDA trainer, so simply enabling it is not a new speed
improvement for those runs.

`compile_model=true` wraps training forward/backward with
`torch.compile(...,fullgraph=True,dynamic=True)`. `compile_mode` defaults to
`default`; `compile_dynamic` is configurable. Optimizer, EMA, named-parameter
statistics and diagnostics retain ownership of the original module. Validation
and sampling use the original eager module. Record cold compilation cost and
source/runtime provenance; compilation is not a convergence intervention and
its throughput benefit has not been benchmarked here. Floating-point fusion
can alter rounding, so CUDA numerical parity is a compatibility check, not a
promise of bitwise-identical training. Dynamic batch support avoids deliberate
padding/data changes; compiler guards may still specialize particular sizes.

The original model is already class-conditional. CFG additionally needs an
unconditional *class* path, while preserving timestep conditioning. With
`class_dropout_prob=.1`, replace10% of training labels by sentinel10 using a
separate seeded RNG (`seed+30000`). The model chooses a learned null vector
instead of the ten-way class embedding. The old class table and all existing
parameter initialization remain unchanged; the null vector uses a separate
local initialization generator and adds only `width` parameters. Dropout0
preserves the old model and avoids extra RNG draws. Each example retains its
own timestep. Dropping class information does not drop diffusion time.

Sampling uses:

```python
# x: [B, 3, 32, 32]; c: [B]; null labels are 10
conditional = model(x, t, c)
unconditional = model(x, t, torch.full_like(c, 10))
epsilon = unconditional + cfg_scale * (conditional - unconditional)
```

For scale1, execute only the conditional pass. Other scales evaluate the
conditional/unconditional pair in a combined2B batch and combine outputs in
FP32. All three predicted RGB epsilon channels are guided; there are no
learned-variance output channels. Guidance therefore increases sampling compute
and peak activation memory. Scale0 is the unconditional prediction. A model
trained without label dropout cannot supply a trained null-class prediction;
non-unit guidance is rejected for that configuration.

Set `guidance_scales="1,1.5,2"` to evaluate all scales on one final in-memory
EMA state with the same initial-noise seed, class requests, DDIM settings and
FID reference. `guidance_scale=1.0` remains the primary result; additional
results live in `guidance_evaluations`. Only the primary scale may save the
explicitly requested diagnostic grid. Guided FID and class agreement must be
reported together; stronger guidance is not a guaranteed FID improvement.
The historical checkpoints were not retained, and those models did not train
a null class, so this requires fresh training.

An example override for the next *controlled* cohort is:

```json
{
  "adam_beta1": 0.95,
  "adam_beta2": 0.999,
  "fused_adamw": true,
  "compile_model": true,
  "compile_mode": "default",
  "compile_dynamic": true,
  "class_dropout_prob": 0.1,
  "guidance_scale": 1.0,
  "guidance_scales": "1,1.5,2"
}
```

Proposed next quality comparison: current-size plain AdaLN-Zero and independent
r8/s16 affine DPLR, each with beta1=.9/.95, common10% label dropout and all three
sampling scales. Keep scheduler, training horizon, EMA, seeds and evaluator
fixed; inspect selected intermediate states in process without saving weights.
This compares betas within the new CFG-training regime. Include a dropout0
control if we need to isolate the training cost of CFG itself. Treat compiler
choice as a recorded common runtime setting after compatibility validation,
not another quality-search axis. Then compare width256/depth8 against current
size under the selected optimizer recipe. This protocol is proposed; only bounded compatibility checks have been
launched. The existing two-GPU cap applies.

Implementation: `dit_cifar10.py`, `train_dit_cifar10.py`.
Regression and numerical ownership checks: `test_dit_training_options.py`.

Validation completed in `runs/dit-training-options-20261009T221005Z`:
128 CPU tests passed. Actual CUDA Inductor fullgraph/dynamic FP32 outputs
matched eager within1.79e-7 absolute error at batches4/3; maximum gradient
absolute error was9.32e-10. The null-class embedding received nonzero gradients,
and fused AdamW/optimizer/EMA ownership checks passed. The full current-size
model completed12 BF16 compiled training steps with diagnostics at0/1/3/5/10/12,
10% label dropout (157 of1,536 labels dropped), beta1=.95, EMA and final
CFG1/1.5/2 FID integration checks. The20-image/two-DDIM-step evaluator is a
compatibility test only; its FID values are not quality evidence.

The first smoke config used a20-image holdout together with the existing
classifier, which correctly rejected nonmatching validation indices. Its logs
are retained. The retry omitted that unchanged classifier and completed the
CFG/FID path. All source hashes match the frozen runner. One GPU was claimed
at a time, all private workers are closed and claims released. About68MiB of
compact evidence and compiler cache were retained under the declared1GiB
bound. No model weights, checkpoints or image grids were saved.
See [compatibility audit](runs/dit-training-options-20261009T221005Z/audit.json).

## Paired beta1/CFG convergence study: launch (2026-10-09)

The user approved proceeding with the current-size optimizer/CFG comparison.
`dit_optimizer_cfg.py` freezes and manages eight30k-step runs: plain AdaLN-Zero
and independent affine r8/s16 DPLR, each with beta1=.9/.95, paired seeds86/87.
All use width192/depth6/heads3/patch4, MLP ratio4, LR4e-4, warmup50, a30k cosine
schedule, adapter LR multiplier1, gain1/sqrt(8), BF16, fused AdamW, compiled
training, EMA.999, batch128 and10% label dropout. Beta2=.999 and the other
optimizer settings are common. Adapter rates are deliberately fixed here;
changes in momentum and adapter update magnitude are part of the measured
contrast rather than calibrated away. Keep their diagnostics with quality.

FID and classifier agreement are measured in process at5k/10k/20k/30k.
Intermediate states evaluate CFG1; the final state evaluates CFG1/1.5/2 with
the same EMA weights, initial-noise seed, balanced class requests and DDIM50
settings. Each quality evaluation uses the full5,000-image real holdout and
5,000 generated images. Raw/EMA epsilon MSE and timestep-decile errors are
logged every5k. The auxiliary classifier retains its exact full validation
indices. The official test split is untouched.

Intermediate evaluation is opt-in via `generative_eval_steps`. Evaluator
construction runs inside an RNG-preserving context for Torch CPU/CUDA, Python
and NumPy. It cannot consume the training noise or label-dropout generators.
Unused evaluator allocator cache is released before training memory accounting
resumes. Generation time is excluded from training windows but included in
`loop_seconds` and recorded separately. Intermediate states save no grids or
weights; final states save one100-image CFG1 visual diagnostic each. We do not
claim an isolated compilation speed comparison from concurrent trainer timing.

The launched workflow is `runs/dit-optimizer-cfg-20261009T224730Z`. It first runs
a12-step CUDA preflight with the full45k/5k loader, the existing classifier,
intermediate generation and all three final scales. The tiny preflight's FID
is compatibility evidence only. The controller starts quality only after a
successful preflight and limits all claim-owning workers to two, including
sampling/evaluation. Beta pairs are submitted together with their order seeded
and frozen; the second seed reverses architecture order. Finite private
supervisors close worker groups after each stage; failures stop this workflow
and preserve its evidence. The controller automatically writes a comparison,
convergence/guidance figures and an audit after the eight runs complete.

Prelaunch validation passed138 CPU tests, including exact preservation of
training, EMA and RNG states with intermediate evaluation enabled and an
evaluator that deliberately consumes global random draws. Frozen sources,
launched configs, runtime versions, full split indices, initialization hashes,
observed batch/noise/effective-label digests, final noise/dropout RNG states,
worker exit records, hardware claims, metrics and evaluator hashes are retained.
The postflight audit reconstructs the fixed recipes, paired traces, denominators,
learning-rate schedules and concurrency before a comparison is exported.

The declared budget is3–5 aggregate GPU-hours, approximately1.5–2.5 hours on
two GPUs subject to availability, and2GiB of retained artifacts (including up
to1.5GiB of expected compiler cache). Prelaunch free space was about319GiB.
No recovery checkpoints or final model weights are enabled. The purposes and
retention of compiler caches and diagnostic grids are recorded before launch;
all declared artifacts remain until explicitly removed. Existing evaluator
weights are read without duplication or retention changes.

This study estimates beta effects within the CFG-training regime; without a
fresh dropout0 control it does not isolate the effect of label dropout itself.
Plain and DPLR have equal base width/depth, not equal total parameters. Two seeds
support paired trends, not significance claims, and30k steps do not establish
saturation. Larger model sizes and projector changes remain subsequent axes.

[Launch protocol](runs/dit-optimizer-cfg-20261009T224730Z/protocol.json) ·
[Planned configs](runs/dit-optimizer-cfg-20261009T224730Z/planned_configs) ·
[Workflow implementation](dit_optimizer_cfg.py).

## Beta1/CFG study: completed results and postflight recovery (2026-10-10)

All eight30k-step runs in `runs/dit-optimizer-cfg-20261009T224730Z` completed.
The two-GPU cap held, all private worker/controller supervisors are closed,
and no checkpoints or final weights were saved. Learning curves and final
same-state guidance comparisons are now retained in the exported report.

CFG2 is the best tested scale for every recipe in both paired seeds, improving
FID and auxiliary classifier agreement together. With beta1=.9, DPLR mean
FID changes62.11 →54.32 →49.50 at CFG1/1.5/2, with agreement48.67% →68.08%
→79.98%. Plain AdaLN-Zero changes68.63 →63.02 →59.05, with agreement39.84%
→55.82% →67.93%. DPLR's roughly18.5% greater parameter count remains a
capacity confound. The classifier's real holdout accuracy is80.72%; agreement
near80% does not establish human fidelity or erase classifier domain shift.

Beta1=.95 improves5k CFG1 FID in both paired seeds: DPLR means115.78 versus
131.34 at.9, and plain means125.89 versus140.01. This early lead largely
shrinks later. Final DPLR CFG2 FID is49.43 versus49.50, with paired changes
+0.04/-0.18 and agreement declines in both seeds. Plain final CFG2 FID is
59.30 versus59.05, with mixed seed effects. Keep.95 as an early-convergence
candidate rather than claiming a final optimizer win. The actual reference
DPLR CFG1 curve is131.34/81.86/64.99/62.11 at5k/10k/20k/30k; this is now a
measured same-cohort trajectory, strengthening the5k-undertraining concern.
The continuing20k-to30k improvement is not evidence of saturation.

The automatic postflight audit stopped on a bookkeeping bug: the generic
summary contains a live byte tally, which grew when its own summary file was
written. Replay equality now excludes only `audit.bytes_retained_under_screen`;
all configurations, metrics and source/evidence hashes remain compared. Actual
storage is independently recomputed and checked against the original bound.
Four focused tests passed, including rejection of changed metric/hash evidence,
and the full completed-run audit passed. The original failed protocol and
controller log, unchanged frozen training runner, unchanged metrics/report,
separate corrected analysis source and explicit recovery record are retained.
The workflow's status was updated to complete with recovery provenance;
its original controller exited1, and that fact is not rewritten as success.
No training was rerun. Approximately437MiB were retained at the initial audit;
quality loops plus all generation used2.506 aggregate GPU-hours, with preflight
and startup additional.

Proposed next control: retain beta1=.9 and compare CFG1/2 while testing
width256/depth8 with plain and independent affine DPLR. Preserve current-size
controls for capacity attribution. No larger-model or additional-guidance
training is launched by this result update.

[Report and figures](reports/dit-optimizer-cfg.md) ·
[Structured comparison](reports/dit-optimizer-cfg.json) ·
[Full evidence](reports/dit-optimizer-cfg-evidence.json) ·
[Audit](reports/dit-optimizer-cfg-audit.json) ·
[Postflight recovery](reports/dit-optimizer-cfg-postflight.json) ·
[Artifact manifest](reports/dit-optimizer-cfg-manifest.json).

## Larger-model capacity controls: launch (2026-10-10)

The user approved the larger-model comparison. `dit_capacity.py` launches six
30k-step runs: three width256/depth8/heads4 models, paired seeds86/87. Reusing
these seeds allows comparison with the completed width192/depth6 beta1=.9
anchors; the old runs, metrics, report and evaluator weights remain unchanged.
The trainer/model/diagnostic source hashes must match the previous study before
launch, and the postflight audit checks implementation/runtime compatibility
and observed data/noise/effective-label traces against those anchors.

| Model | Static MLP hidden width | Parameters |
|---|---:|---:|
| Plain AdaLN-Zero |1024|9,758,768|
| Plain AdaLN-Zero, wider MLP |1360|11,137,712|
| Independent affine DPLR r8/s16 |1024|11,135,280|

The wider static MLP matches DPLR within2,432 parameters (about0.02184%). It
controls for additional static capacity while keeping the base width/depth,
condition embedding and conventional scale/shift/gates. Parameter matching does
not match FLOPs, activation shapes or function class. The ratio4 plain model
and DPLR have identical base initialization for each seed. The wider-MLP model
has different-shaped weights and its own reproducible initialization; all
three retain paired data, noise and label-dropout streams. Training peak
allocated memory and matrix-FLOP estimates are retained alongside quality;
concurrent compiled throughput is not an isolated speed benchmark.

Keep beta1=.9/beta2=.999, LR4e-4, warmup50,30k cosine horizon, batch128,
BF16 compiled/fused training, EMA.999,10% class dropout and the same45k/5k
split. DPLR retains rank8/basis count16, gain1/sqrt(8), normalized input,
independent affine projectors and adapter LR multiplier1. These rates/gains
are held fixed, not retuned or update-calibrated for the larger models.
Raw/EMA validation and timestep-decile errors are recorded every5k. FID and
class agreement use5k real/generated images, fixed classifier and DDIM50.
Intermediate states5k/10k/20k evaluate CFG1; the final30k state evaluates
CFG1/2 on the same in-memory EMA weights and initial-noise seed. Final CFG1
alone saves a100-image visual diagnostic grid.

The launched workflow is `runs/dit-capacity-20261010T014610Z`. Before quality,
it checks DPLR and the wider static MLP concurrently on CUDA:12 compiled BF16
steps over328 training examples yield repeated128/128/72 batches, exercising
the full loader's ragged batch shape. The holdout remains the full5k for the
existing classifier. Intermediate and final guidance evaluation paths are
included; the preflight's20-image/two-DDIM-step FID is compatibility evidence
only. The same finite-worker controller enforces a two-GPU cap across preflight,
training and evaluation and stops this workflow on failure. Completed quality
runs receive a reconstructed-evidence audit and automatic comparison/figures.
The corrected disk-tally comparison from the prior postflight is reused.

Prelaunch validation passed142 CPU tests, including exact parameter counts,
shared plain/DPLR base initialization, the fixed six-case recipe matrix,
paired contrast direction and the existing trainer/regression checks. The
resource estimate is4–6 aggregate GPU-hours, approximately2–3 hours at two
GPUs subject to availability. The declared retained-artifact bound is2GiB,
including up to1.5GiB of expected compiler cache; prelaunch free space was
about318GiB. Frozen sources/configs, split indices, runtime/evaluator hashes,
metrics, diagnostics, observed batch/noise/label digests, final RNG states,
worker logs/claims, reports and specified final grids remain until explicitly
removed. No periodic checkpoints, recovery checkpoints or final model weights
are enabled, and evaluator caches are not copied or deleted.

A two-seed30k comparison can establish paired trends, not significance or
saturation. The historical small-model anchors are separately executed cohorts.
Projector changes, additional CFG scales and longer schedules remain later
experiments rather than simultaneous axes in this capacity comparison.

[Launch protocol](runs/dit-capacity-20261010T014610Z/protocol.json) ·
[Planned configs](runs/dit-capacity-20261010T014610Z/planned_configs) ·
[Workflow implementation](dit_capacity.py).

## Larger-model capacity controls: completed (2026-10-10)

All six 30k-step runs completed and passed the automatic evidence audit.
At CFG 2, plain AdaLN-Zero reaches FID 45.59 ± 0.20; the wider static MLP
reaches 42.90 ± 0.69; independent affine DPLR r8/s16 reaches 39.68 ± 0.50.
These are means and sample SDs over paired seeds 86/87. The wider MLP and
DPLR have 11,137,712 and 11,135,280 parameters respectively, within 0.022%.
DPLR improves both FID and classifier agreement in both seeds against this
matched control: mean FID improvement 3.22, agreement improvement 6.61 pp
(83.62% versus 77.01%). At CFG 1 the FID advantage is smaller, 0.88.

The earlier smaller DPLR anchor reached CFG 2 FID 49.50; increasing capacity
improves it to 39.68. Plain models also improve with capacity. Final EMA noise
MSE is almost equal between larger DPLR and the matched MLP, emphasizing the
value of generation metrics. DPLR's training peak allocation is 28.8% lower
than the wider MLP; estimated forward matmul FLOPs are 15.9% lower. Neither
measurement establishes isolated runtime performance. FID uses 5k held-out
real/generated images and agreement is an auxiliary classifier proxy; two
seeds do not establish significance or saturation.

The private controller and workers are closed and all workflow claims released.
No checkpoints or final model weights were saved. Frozen training evidence,
configs, logs, diagnostics, final grids and declared caches remain retained.
The next proposed axis is a focused higher-s/nonlinear-projector comparison
with fixed transform rank and model size, explicit budgets and update-transfer
diagnostics. Longer schedules remain a separate convergence question. No
additional training was launched during postflight.

[Report and figures](reports/dit-capacity.md) ·
[Structured comparison](reports/dit-capacity.json) ·
[Full evidence](reports/dit-capacity-evidence.json) ·
[Audit](reports/dit-capacity-audit.json) ·
[Artifact manifest](reports/dit-capacity-manifest.json).

## Higher-s and nonlinear-projector confirmation: launch (2026-10-10)

The user approved the focused higher-s and matched linear/SiLU comparison.
The workflow is `runs/dit-projector-confirmation-20261010T043230Z`. It runs
seven new recipes over the same paired seeds 86/87 (fourteen 30k-step runs),
and reuses the six audited larger-capacity runs as frozen anchors. Reuse avoids
retraining identical recipes; comparisons retain their separately executed
cohort provenance. No longer schedule, shared projector, token-conditioning,
transform-normalization or structured-factorization axis is added here.

| DPLR recipe | Basis count s | Coefficient head | Total params | Coefficient-head params | Matched static MLP width / params |
|---|---:|---|---:|---:|---|
| Affine anchor |16|Linear(256,32)|11,135,280|65,792|1360 / 11,137,712 (reused)|
| Higher s |32|Linear(256,64)|12,511,792|131,584|1696 / 12,516,656|
| Higher s |64|Linear(256,128)|15,264,816|263,168|2368 / 15,274,544|
| Factorized linear |32|Linear(256,64), Identity, Linear(64,64)|12,545,072|164,864|1704 / 12,549,488|
| SiLU MLP |32|Linear(256,64), SiLU, Linear(64,64)|12,545,072|164,864|same static control as linear|

Plain AdaLN-Zero at 9,758,768 parameters is also retained as a historical
anchor. New static controls are three separate recipes. Each static budget
matches its DPLR candidate within 0.065%, with hidden widths rounded to multiples
of eight. Parameters are matched; FLOPs and activation shapes are not. Most of
the growth with s comes from learned factor bases, rather than the small
coefficient head. Head parameter totals include eight independent layers.

The broadcast tensor path remains:

```python
# c: [B,256]; h_norm: [B,64,256]; r=8; s in {16,32,64}
a, b = coefficient_head(c).chunk(2, dim=-1)  # each [B,s]
U_basis = U.view(256, s, 8)
V_basis = V.view(s, 8, 1024)
u = einsum('bs,dsr->bdr', a, U_basis)  # [B,256,8]
v = einsum('bs,sro->bro', b, V_basis)  # [B,8,1024]
delta = bmm(bmm(h_norm, u), v) * (1 / sqrt(8))
pre_act = mlp_in(adanorm(h_norm, c)) + delta
# Existing nonlinearity, MLP output projection and AdaLN residual gate follow.
```

Transform rank r is fixed. Increasing s enriches factor generation while the
shapes of the generated u/v factors stay fixed. The hidden-linear and SiLU pair
has exactly the same initial parameter tensors and base initialization; the
activation is the only config difference. Hidden width64 equals coefficient
output width64, so the linear two-layer head can represent any affine256-to64
map; it tests parameterization/optimization rather than a stricter rank bound.
SiLU changes activation magnitude as well as expressivity.

All DPLR recipes keep independent layer heads/bases, normalized adapter input,
nonzero U, zero V, gain1/sqrt(8), and adapter LR multiplier1. Zero V makes the
initial delta zero; conventional AdaLN heads and gates also start at zero.
Rates and gains are fixed, not calibrated in this study. Increasing s and
changing activation can alter the effective update scale, so magnitude and
update diagnostics are essential context for quality differences. Additional
normalization or gain compensation remains a separate later ablation.

Keep width256/depth8/heads4/patch4, static DPLR MLP hidden1024, AdamW betas
(.9,.999), LR4e-4, warmup50,30k cosine horizon, batch128, BF16 compiled/fused
training, EMA.999 and 10% class dropout. Use the same45k/5k split and observed
paired data/noise/effective-label streams. Intermediate5k/10k/20k evaluations
use CFG1; the final30k EMA state uses CFG1/2 with the same initial-noise seed.
FID uses5k real/generated images and DDIM50, alongside the existing frozen
classifier. The official test remains untouched.

Diagnostics retain initial/per-step coefficient RMS, basis parameter RMS,
generated-factor RMS, transform Frobenius norm, per-block delta/base and gated
residual ratios, clipped gradients, and actual single-step function updates
on fixed normalized tokens/conditions. The comparison records paired update
ratios against the s16 anchor. These are FP32 raw-model probes, not EMA probes
or update calibration. Historical anchors lack the new coefficient telemetry;
missing historical values remain absent. Factor magnitudes depend on the
learned basis gauge, while transform/function probes describe its combined
effect. No basis normalization is introduced.

Only `dit_quality_diagnostics.py` changes among training/evaluation sources:
extra no-grad magnitude fields, including step zero. CPU parity checks confirm
that diagnostics preserve raw/EMA training outcomes for linear/SiLU heads.
Model, trainer, data pipeline and evaluator source hashes match the anchors.
The new protocol explicitly records the diagnostic-only exception; prior
sources, results, audits and artifacts remain unchanged. Postflight checks
frozen sources/evaluators, source/runtime compatibility, paired observed
training streams, initialization/gradient activation, full evaluation budgets,
claimed hardware, GPU cap, storage and absence of weights/checkpoints.

Prelaunch validation passed208 CPU tests. Four CUDA checks cover affine s64,
linear/SiLU s32 h64, and the largest static MLP. Each uses12 compiled BF16 steps
with328 train examples (128/128/72 ragged batches), full5k holdout, and tiny
20-image/two-DDIM-step generation for execution compatibility only. A private
supervisor and gpu-claim wrappers cap this workflow at two workers throughout
preflight, training and evaluation. Other projects' claims are protected.
Quality starts automatically only if all preflights pass; a failure stops this
workflow and preserves evidence.

Before launch, free shared disk is about318GiB. Declare6–9 aggregate GPU-hours,
approximately3–4.5 wall hours at two GPUs, dependent on availability. Retained
artifacts are bounded at4GiB, including up to3GiB of expected compiler cache.
Retain frozen sources/configs, split indices, runtime/evaluator provenance,
metrics, diagnostics, RNG/input stream hashes, logs/claims, summaries/audits,
compiler caches and final diagnostic grids until explicitly removed. Save one
100-image final CFG1 grid per quality run and one10-image grid per preflight.
Existing shared evaluators are reused without copying. No periodic/recovery
checkpoints or final model weights are enabled.

The primary comparisons are s32/s64 versus s16, s64 versus s32, linear hidden
head versus affine s32, SiLU versus matched linear, and each DPLR recipe versus
its parameter-matched static control. Retain both CFG scales, paired seed
results and magnitude/update context; do not select winners from5k FID alone.
Two seeds and finite5k-image FID establish trends, not significance or saturation.

[Protocol](runs/dit-projector-confirmation-20261010T043230Z/protocol.json) ·
[Planned configs](runs/dit-projector-confirmation-20261010T043230Z/planned_configs) ·
[Workflow](dit_projector_confirmation.py).

Launch verification: all four CUDA preflights completed successfully; adapter
gradients activated by step5. The preflight supervisor is closed, and quality
training has started under the same two-worker cap. Compact verification is
recorded in `runs/dit-projector-confirmation-20261010T043230Z/launch_verification.json`.

## Projector confirmation: interim review (2026-10-10)

As of 2026-10-10T06:28:43.566446+00:00, 9 of fourteen new runs have completed;
the full paired comparisons and automatic audit remain pending. Both s64 seeds
are complete (mean CFG2 FID37.61 versus39.68 for the reused s16 anchor).
First-seed SiLU is worse than its matched linear control. Fixed-input raw-model
function updates expose a scale difference: at step500, s64 updates are
3.03–4.14 times s16, and SiLU is0.39 times its matched linear head in seed86.
Keep these optimization differences alongside the quality comparison; they
are descriptive associations, not causal attribution. Finish the declared pairs
before selecting a design. A targeted gain/LR control is a possible next study;
current gains, rates, retention and worker cap remain unchanged.

[Interim evidence](runs/dit-projector-confirmation-20261010T043230Z/interim/20261010T062843Z.md).

## Adapter-LR controls: implementation ready, awaiting paired study (2026-10-10)

The current higher-s/linear/SiLU comparison is still running. The next control
is implemented in `dit_projector_lr_controls.py`; preparation/launch rejects
an unfinished or unaudited reference, active reference worker supervisor,
changed training sources, or insufficient shared disk space. Existing runs,
settings and retention remain unchanged. This implementation has not launched
additional GPU jobs. Fifteen focused CPU checks passed, including the parent
projector regression checks and recipe isolation, rate-only quality changes,
paired arithmetic, exact pilot grid, and withheld quality on a missed match.

Only the adapter LR multiplier varies; base LR4e-4, rank8, model width256/depth8,
transform gain1/sqrt(8), initialization, data/conditioning and the30k training
schedule remain fixed. The multiplier applies to both learned bases and the
coefficient head. It is a scalar functional-update control, not a separate
head-only rate, gauge fix, transform normalization or muP prescription.

Use fresh pilot seeds90/91, two fixed target recipes, and three fixed candidate
rates per family. All pilots train500 steps on the full45k/5k split, using the
same prefix of the30k LR schedule and compiled BF16/fused AdamW. Pilot FID,
generated datasets, grids and model checkpoints are disabled.

| Candidate | Target at multiplier1 | Candidate adapter LR multipliers |
|---|---|---|
| Affine s32 | Affine s16 |0.25,0.5,1.0|
| Affine s64 | Affine s16 |0.125,0.25,0.5|
| SiLU s32 h64 | Linear s32 h64 |1.0,2.0,4.0|

The targets account for four pilot runs; nine candidate recipes across two
seeds account for eighteen. The total is strictly bounded at22 pilots, with
no extra correction rounds. Compute mean single-step adapter-function update
RMS over the eight blocks on fixed normalized tokens/conditions, at steps
50/100/250/500. Pair each seed and observation with its target before forming
the ratio. Select the closest absolute log geometric-mean ratio among candidates
with aggregate ratio within30% of one and every seed/observation ratio within
0.5–2.0. If none meet both bounds, retain the closest miss for diagnosis and
withhold all quality runs. Selection never reads loss, FID or agreement.

On a successful pilot match, freeze the selected multipliers and their
calibration evidence hash before six30k quality runs: three recipes × paired
seeds86/87. Reusing quality seeds supports rate-only comparisons with the
completed reference, including identical base/adapter initialization and
observed data/noise/label streams. Pilot seeds are separate, but these quality
seeds and the evaluation split have already been observed; this is an
exploratory follow-up rather than an independent generalization estimate.

Final CFG1/2 use the same EMA state and initial-noise seed, DDIM50,5k real/fake
images and the fixed classifier. Preserve intermediate learning curves and
measure quality-seed early function-update ratios against the reused targets.
A pilot match does not imply matching transfers to other seeds or remains
matched throughout training. Report the measured transfer alongside quality;
do not interpret a missed transfer as a clean architecture-only comparison.

The initial candidate grid is motivated by the observed first study's update
scales, not selected from generated-image quality. In the currently completed
observations, affine s64's early geometric-mean update ratio is about4.04×
the s16 anchor; the first-seed SiLU/linear ratio is about0.43×. These inverse
rate hints are approximations: Adam, clipping and diverging trajectories can
change the response, hence the separate measured pilot bracket.

Declare3–5 aggregate GPU-hours (roughly1.5–2.5 hours at two GPUs) for the bounded
pilots and successful quality phase. Before launch, require8GiB shared free
space and record live capacity. Retain at most4GiB, including up to3GiB of
expected compiler cache, frozen sources/configs/splits/runtime/evaluator
provenance, diagnostics, recipe selection, metrics, logs/claims, reports and
six specified100-image final CFG1 grids. Retain these until explicitly removed;
reference artifacts and shared evaluator caches stay in place. Save no periodic,
recovery or final model weights. Private supervisors and gpu-claim wrappers
cap pilot, training and evaluation at two workers throughout.

[Concrete plan](reports/dit-projector-lr-plan.json) ·
[Implementation](dit_projector_lr_controls.py) ·
[Tests](test_dit_projector_lr_controls.py).

Adapter-LR queue update: the follow-up is now queued in `runs/dit-projector-lr-20261010T064856Z`.
Seventeen focused CPU checks passed, including a successful prerequisite
release and an abort on prerequisite failure. The managed waiter uses no GPU
workers. After the parent comparison completes, passes its audit and closes its
worker supervisor, it rechecks frozen training sources, runtime/evaluator
compatibility and live disk capacity before starting pilots. The wait is bounded
at eight hours; a failure stops the queued workflow and preserves evidence.
Quality still requires the declared calibration guards. The study has a4GiB
retained-artifact bound; the enclosing queue declares5GiB total including frozen
queue sources and logs. No model weights or checkpoints are enabled. Every
compute stage retains the two-GPU cap.

[Queue protocol](runs/dit-projector-lr-20261010T064856Z/queue_protocol.json) ·
[Updated plan](reports/dit-projector-lr-plan.json).

## Higher-s and projector confirmation: completed (2026-10-10)

All fourteen new30k runs completed; the six reused capacity anchors and full
reconstructed-evidence audit passed. No failure/recovery was needed. At CFG2,
affine s16/s32/s64 reaches mean FID39.68/38.22/37.61, with classifier agreement
83.62/86.13/87.40%. Increasing basis count improves both metrics in both seeds,
with diminishing FID returns. The s32-to-s64 gain is0.62 FID for2.75M additional
parameters. Affine s32 is a useful working parameter tradeoff pending rate controls.

Against matched static MLPs, s32 improves CFG2 FID by3.62 and agreement by9.41pp;
s64 improves them by1.40 and8.00pp. Both seeds favor these DPLR contrasts. At
CFG1 the s64 static MLP has better FID (50.77 versus51.44), while DPLR retains
higher agreement. Keep both guidance scales and metrics when interpreting capacity.

The hidden-linear s32/h64 head reaches CFG2 FID38.59 and85.71% agreement;
its matched SiLU head reaches40.50/82.36%. SiLU loses both metrics in both
seeds at both guidance scales. This is a fixed-recipe result, not a rejection
of nonlinear projectors generally. At step500, SiLU function updates are only
0.34–0.39 times its linear control; s32 and s64 updates are1.75–2.08 and
3.03–4.14 times the s16 anchor. The fixed gain/rate comparisons mix capacity
and effective optimization scale; these associations are not causal attribution.

Transform rank8 keeps generated factor shapes fixed as s increases. The
additional cost mainly resides in learned bases; the s32 coefficient heads
are131,584 parameters total (about1.05% of the model). Forward-matmul estimates
increase modestly,862.1M→864.8M→870.3M FLOPs/image across s16/32/64. Measured
training peak allocation is1,195MiB at s32 versus1,748MiB for its static control,
and1,247MiB at s64 versus1,982MiB. Whole-training allocation is not isolated
factor memory; concurrent compiled throughput is not an isolated speed benchmark.

The completed parent controller and workers are closed. The LR queue released
automatically after the successful audit, with two pilot workers active. It
retains the fixed22-pilot limit, separate calibration seeds, update-only
selection, conditional six30k quality runs, fixed transform gain/base LR, and
two-GPU cap. No checkpoints or final model weights are saved. Two paired seeds
and finite5k-image held-out FID establish trends, not significance or saturation.

[Report and figures](reports/dit-projector-confirmation.md) ·
[Structured comparison](reports/dit-projector-confirmation.json) ·
[Evidence](reports/dit-projector-confirmation-evidence.json) ·
[Audit](reports/dit-projector-confirmation-audit.json) ·
[Artifact manifest](reports/dit-projector-confirmation-manifest.json).


## Adapter LR controls: completed (2026-10-10)

The bounded study in `runs/dit-projector-lr-20261010T064856Z/study`
completed at09:04 UTC. Its automatic audit passed all checks and the queue
controller exited successfully. Both stage supervisors and the queue
supervisor are closed. Compact retained artifacts were1,501,791,521 bytes,
within the4GiB study bound; no model weights were saved.

[Report and plots](reports/dit-projector-lr.md),
[comparison](reports/dit-projector-lr.json),
[exact training evidence](reports/dit-projector-lr-evidence.json),
[calibration](reports/dit-projector-lr-calibration.json),
[frozen recipes](reports/dit-projector-lr-recipes.json),
[audit](reports/dit-projector-lr-audit.json), and
[artifact manifest](reports/dit-projector-lr-manifest.json) preserve the
selection and quality evidence without modifying either reference cohort.

The exact22-run grid passed calibration on separate seeds90/91. Affine s32
selected adapter LR multiplier0.5 with geometric mean function-update ratio
1.146 to s16; affine s64 selected0.25 with ratio1.126; SiLU selected2.0 with
ratio0.756 to hidden linear. Every selected seed/observation ratio satisfied
the0.5–2 bound and each aggregate was within30% of1. Rates were frozen before
quality and selected without reading FID, agreement or loss. SiLU is near
the lower aggregate tolerance; calling these exactly equal updates would
be incorrect.

On quality seeds86/87, early transfer ratios ranged0.912–1.453 for affine
s32,0.892–1.363 for s64, and0.690–0.925 for SiLU. These are FP32 fixed-input
raw-model probes over steps50/100/250/500, not EMA or whole-model updates.
They measure one early window rather than the full optimization trajectory.

| Family | Selected adapter LR multiplier | CFG1 FID | CFG2 FID | Paired mean CFG2 change from rate1 | CFG2 classifier agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| Affine s32 |0.5|52.54 ±0.35|38.88 ±0.18|+0.66|84.76%|
| Affine s64 |0.25|52.94 ±0.00|40.01 ±0.33|+2.40|83.79%|
| SiLU s32/h64 |2.0|52.39 ±0.21|39.17 ±0.51|−1.33|83.91%|

Each rate-effect FID direction holds in both seeds at both evaluated guidance
scales. Lower rates reduce the affine models' quality under this recipe;
higher rate improves SiLU substantially. SiLU at rate2 still loses to hidden
linear at rate1: paired CFG2 FID differences +0.471/+0.689 and mean39.17
versus38.59. The experiment demonstrates optimization sensitivity. It does
not prove that update matching is optimal, completely equalize dynamics,
establish variance preservation, or derive a muP scaling rule. Gauge-dependent
factor magnitudes remain descriptive; combined adapter-function probes are
the matching signal. Initial matching cannot remove all capacity/optimizer
confounds.

Unit-rate affine s32 remains the selected parameter tradeoff:12.512M
parameters and CFG2 FID38.22 versus12.517M/41.84 for matched static capacity.
Unit-rate s64 remains the best30k guided mean at15.265M/37.61. Selection uses
previously observed seeds/split, so the next study uses fresh initialization
seeds. The held-out split and evaluation noise remain reused research proxies.

## Fresh-seed longer confirmation: launch (2026-10-10)

The authorized follow-up is launched in
`runs/dit-longer-confirmation-20261010T154131Z`, implemented by
[dit_longer_confirmation.py](dit_longer_confirmation.py). Training/model and
diagnostic sources match the audited LR study exactly; only orchestration,
schedule budget and seeds change. Five focused CPU tests verify exact recipe
changes, real parameter counts/base initialization, paired contrasts, the
incomplete-reference launch gate and failure of a preflight withholding long
training. Frozen source/test hashes and CPU validation are in the run root.

| Family | Parameters | MLP hidden width | Transform rank / basis width | Seeds | Steps |
| --- | ---: | ---: | --- | --- | ---: |
| Plain AdaLN-Zero |9,758,768|1024|none|92/93|60,000|
| Static capacity |12,516,656|1696|none|92/93|60,000|
| Affine DPLR |12,511,792|1024|r8/s32|92/93|60,000|

The static control matches adapter-model parameters within0.039%, not FLOPs,
activation shape or base initialization. Plain and DPLR share base
initialization; observed training data/noise/class-dropout traces are paired
within each seed. Fresh seeds92/93 do not independently validate the reused
held-out split.

All use width256/depth8/heads4, patch4, the45k/5k training split, batch128,
AdamW(.9,.999), LR4e-4, no weight decay, clipping1,50-step warmup and a fresh
60k cosine horizon with0.01 floor. DPLR uses normalized branch input,
independent affine heads and bases, adapter gain1/sqrt(8) and adapter LR
multiplier1. Compiled BF16/fused training, class dropout0.1 and in-memory
EMA0.999 remain fixed. This is a fresh training schedule, not a resume of the
30k schedule or a schedule-matched estimate of extra training alone.

Two12-step compiled CUDA preflights cover DPLR and the larger static MLP,
including ragged128/128/72 batches and the full5k holdout. Failure withholds
all six long runs. Noise validation occurs every5k steps; EMA CFG1 generation
at10k/20k/30k/45k/60k uses DDIM50 and5k real/generated images. Final CFG1/2
share one in-memory EMA state and initial noise seed. The audit checks frozen
sources/runtime/evaluators, paired streams, initialization and gradient
activation, generation denominators, actual hardware claims, the two-worker
cap and absence of saved model weights.

Declared resources: at most two GPU workers across preflight, training and
evaluation; estimated5–8 aggregate GPU-hours, approximately2.5–4 wall hours
with two available GPUs. Shared disk free at launch was about313GiB; preparation
requires8GiB headroom. Retained artifacts are bounded at4GiB: compact source,
config, seed/split/runtime evidence, metrics, diagnostics, logs, reports,
compiler caches expected up to3GiB, six final100-image CFG1 grids and two
preflight10-image grids. Retain these until explicitly removed. No periodic,
recovery or final weights are enabled; existing evaluator artifacts and other
projects remain unchanged. Sixty thousand steps still do not prove saturation.

### Next token self-conditioning comparison: preparation only

After the longer comparison, renew ViT/CIFAR-10 with its own optimization
calibration and parameter-matched controls. Earlier50-epoch ViT results did
not establish a DPLR quality/compute advantage over ordinary width; the recent
10-epoch screen favored full condition gradients over detaching them. Neither
evidence licenses copying diffusion's projector width or adapter rate blindly.

Keep MLP-input placement, ordinary self-attention and identical LayerNorm
settings across controls. Use `h = norm_mlp(x)` and `c = h` by default; the
detached ablation uses `c = h.detach()` with the same forward computation.
Retain plain, dynamic diagonal scale/shift, static low-rank plus diagonal,
parameter-matched static capacity and dynamic full/detached controls. A short
basis-width trend can use s12/s24/s48 at model width192 (d/16,d/8,d/4), with
transform rank fixed at8 and separate pilot seeds. This is a proposed family
set, not a frozen selection or launched/queued study.

The broadcast diffusion path can produce factors once per image. ViT must
avoid `[B,T,d,r]` and `[B,T,r,o]` generated tensors. Existing activation mixing
uses intermediates `[B,T,s,r]`; peak storage, saved activations and whole-model
allocation must accompany accuracy/parameter evidence as s grows. Its interval
update probes differ from DiT's single-step probes, so calibrate within the
ViT regime. Dedicated speed benchmarking remains deferred; concurrent training
telemetry alone will not support speed claims. No ViT GPU work overlaps this
longer confirmation.


Both longer-comparison CUDA preflights passed at15:43 UTC: correct parameter
counts, ragged-batch budget and final CFG1/2 evaluations. DPLR's V gradients
activated by step3 and U/coefficient gradients by step5. The six60k-step quality
runs are now underway with two claimed workers; this is launch validation,
not a longer-run quality result.


## Longer confirmation: completed (2026-10-10)

The fresh-seed60k comparison completed at17:40 UTC. Every automatic audit
check passed, including parameter budgets, paired streams, initial zero
branches, gradient activation, evaluator denominators, frozen provenance,
two-GPU claims and absence of saved weights. Retained artifacts were402,334,912
bytes, within the4GiB bound; the private controller and both worker supervisors
are now closed. Original report SHA256 is
`d37d7d8dbc4d428578abb7b8aef6ef0c740d79b6a2fe4622ef09f53420a84609`.

[Report and figures](reports/dit-longer-confirmation.md),
[exact evidence](reports/dit-longer-confirmation-evidence.json),
[audit](reports/dit-longer-confirmation-audit.json), and
[export manifest](reports/dit-longer-confirmation-manifest.json).

| Family | CFG1 FID | CFG2 FID | CFG2 classifier agreement |
| --- | ---: | ---: | ---: |
| Plain AdaLN |47.11 ±0.46|35.07 ±0.39|82.68 ±0.31%|
| Matched static MLP |44.37 ±0.66|32.33 ±0.90|84.77 ±0.44%|
| Affine DPLR s32 |43.66 ±1.32|30.08 ±1.25|90.40 ±0.23%|

DPLR's guided paired FID differences to matched static capacity are−2.49/−2.00
for seeds92/93, with agreement differences+5.16/+6.10 percentage points. Both
FID and agreement favor DPLR in each seed at both guidance scales. This
supports persistence of the earlier capacity-matched advantage under a longer
schedule. Fresh seeds/schedules and the reused split prevent interpreting the
old30k-to-new60k endpoint change as the isolated effect of extra training or
independent generalization. Agreement is the fixed auxiliary classifier proxy,
not human-judged fidelity. Saturation remains unproven.

## Projector terminology clarified

The factorized affine generator can be described without extra machinery:

```python
# For one input factor, generator bottleneck p and transform rank r:
mid_u = Linear(cond_dim, p)(c)
u_flat = Linear(p, d * r, bias=False)(mid_u)
u = u_flat.reshape(B, d, r)
```

Our term "basis mixing" describes this second linear map: each learned row of
its transposed weight is one flattened factor. In current terminology p=s.
The condition supplies p signed coefficients and combines those learned
vectors. Output-factor generation has its own mid_v and decoder. Joint
coefficient heads produce separate mid_u/mid_v; using one shared mid for both
would be a further architectural choice.

AdaLN's `SiLU(c) -> Linear(cond_dim, 6*d)` is a separate scale/shift/gate branch.
Our affine adapter head takes c directly; the tested nonlinear head is
`Linear(cond_dim,h) -> SiLU -> Linear(h,2*s)`, followed by linear factor
decoding. No nonlinearity occurs between applying the generated U and V.
Transform rank r, generator bottleneck s, and optional hidden width h are
independent dimensions.

Generator factorization reduces parameters but does not change the shapes of
materialized factors. Broadcast factors `[B,d,r]`/`[B,r,o]` can be reused
across tokens. Per-token factors would be `[B,T,d,r]`/`[B,T,r,o]`; the existing
ViT activation-mixing route rearranges contractions into `[B,T,s,r]`
intermediates without materializing every token's factors. Parameterization
and execution strategy are separate decisions.

## Factor-offset ablation: launch (2026-10-10)

User authorized testing zero-initialized decoder biases after the completed
longer comparison. The implementation adds opt-in `factor_offsets` to the
broadcast DiT model and trainer; default false preserves the historical
model. The current per-token ViT path is unchanged; bringing offsets to it is
a subsequent integration after this focused broadcast comparison.

```python
# Static trainable offsets shared across all samples/tokens of this adapter:
U_offset: [d, r]       # initialized to zero
V_offset: [r, o]       # initialized to zero
u = decode_u(mid_u) + U_offset
v = decode_v(mid_v) + V_offset
delta = gain * ((norm_h @ u) @ v)
```

The offsets are factor biases, not an output shift. Coefficient-head biases
remain present; they already give static offsets constrained to the decoder
span. Explicit offsets remove that span constraint. Keeping both introduces
redundancy: a coefficient-bias change can be compensated by a decoder-offset
change. There are still factor scaling/rotation gauges. This comparison does
not claim to remove them, normalize the transform or preserve variance.

Both offsets start zero without consuming RNG. Existing U decoder and
coefficient-head initialization remains nonzero; V decoder remains zero, so
the complete delta starts zero and can learn. Offsets are cast to the
computed factor dtype before addition, preserving BF16 factor storage under
autocast. They share the existing adapter optimizer group/rate. Decoder sharing,
when explicitly configured, shares the offsets with U/V and counts them once.
The present experiment uses independent adapters/decoders in every layer.
Diagnostics include offset magnitudes and decoder/input/output gradient groups.

[Implementation](dit_cifar10.py), [feature checks](test_dit_factor_offsets.py),
[orchestrator](dit_factor_offset_study.py),
[protocol checks](test_dit_factor_offset_study.py).

The source change is confined to the optional feature/config plumbing in
`dit_cifar10.py`/`train_dit_cifar10.py` and its diagnostics in
`dit_quality_diagnostics.py`/`benchmark_dit_design.py`. The new disabled model
matches the frozen60k-study model's state, post-initialization RNG, nonzero
forward outputs, gradients and three Adam updates exactly on CPU. Tests also
check dense-operator output/gradient equivalence, zero start, real parameter
budgets, BF16 factor dtype, shared-offset optimizer deduplication, invalid
settings, gradient activation and diagnostics preserving raw/EMA training.
The final focused suite passed15 checks; an earlier broader regression suite
passed61 before the final BF16 dtype guard. Both logs are retained. The frozen
runner includes the old baseline model for replaying the parity check.

The workflow is `runs/dit-factor-offset-20261010T193609Z`:

| Generator width | Offsets | Parameters | Seeds | Steps |
| --- | --- | ---: | --- | ---: |
| s8 |off|10,447,024|94/95|30,000|
| s8 |on|10,528,944|94/95|30,000|
| s32 |off|12,511,792|94/95|30,000|
| s32 |on|12,593,712|94/95|30,000|

Each enabled model adds81,920 parameters:10,240 per block, approximately0.78%
of the s8 model or0.65% of s32. This is a modest extra-capacity ablation, not
an exact parameter match. Both offsets change together, so a result cannot
identify which one matters. Within a width, every existing parameter tensor,
initial function and RNG state is preserved; all four families share base
initialization and observed input/noise/class-dropout streams within each seed.

All other settings stay fixed: width256/depth8/heads4/patch4, transform rank8,
MLP hidden1024, normalized adapter input, gain1/sqrt(8), affine independent
coefficient heads, adapter LR multiplier1, AdamW(.9,.999), LR4e-4, weight
decay0, clip1, batch128,50-step warmup and fresh30k cosine schedule with0.01
floor. Training is compiled BF16/fused, with class dropout0.1 and in-memory
EMA0.999, on the same45k/5k CIFAR-10 split. Generation uses DDIM50 and5k
real/fake images, CFG1 at5k/10k/20k/30k and final CFG2 on the same EMA state
and initial noise. Two12-step offset-enabled GPU preflights at s8/s32 check
ragged batches, real evaluation paths and activation of all offsets by step10;
failure withholds all eight quality runs.

Resource declaration: at most two GPU workers across every stage; estimated
4–6 aggregate GPU-hours, approximately2–3 wall hours with two available GPUs.
Preparation requires8GiB shared free space; about314GiB was available. Retained
artifacts are bounded at4GiB: compact frozen source/config/seed/split/runtime
records, diagnostics/metrics/logs/audits/reports, compiler caches expected up
to3GiB, eight final100-image CFG1 grids and two preflight10-image grids.
Retain these until explicitly removed. No periodic/recovery/final weights,
new evaluator weights or official-test evaluation are enabled. The completed
longer-study evidence and evaluators are preserved.

The current comparison does not search LR or use quality metrics to select
rates, and it cannot disentangle new capacity from new update directions.
These are two fresh training seeds on a reused research holdout. Prior60k
endpoints are context rather than paired bias controls. ViT token
self-conditioning remains next; this user-requested small offset ablation
precedes it and does not launch any additional projector/structured-operator
search or dedicated speed benchmark.


Both factor-offset compiled GPU preflights passed at19:38 UTC. All V offsets
activated by step3 and U offsets by step5, with zero initial gates/deltas and
correct parameter/evaluation budgets. The eight paired30k quality runs are
underway with two workers. This verifies launch behavior; offset quality
results are pending.

## Factor-offset results and reporting recovery (2026-10-11)

The eight fresh 30k runs completed successfully on October 10. Means ± sample
SD over paired seeds 94/95:

| Generator width | Factor offsets | Parameters | CFG1 FID | CFG2 FID | CFG2 classifier agreement |
| --- | --- | ---: | ---: | ---: | ---: |
| s8 | off | 10,447,024 | 54.17 ± 0.36 | 41.55 ± 0.48 | 80.16 ± 0.42% |
| s8 | on | 10,528,944 | 53.21 ± 0.48 | 40.88 ± 1.03 | 80.65 ± 0.47% |
| s32 | off | 12,511,792 | 52.19 ± 0.26 | 38.57 ± 0.20 | 85.90 ± 0.88% |
| s32 | on | 12,593,712 | 51.29 ± 0.29 | 37.85 ± 0.72 | 86.79 ± 0.13% |

Paired offset-minus-no-offset CFG2 FID differences are −0.273/−1.061 at s8
and −0.357/−1.090 at s32. Agreement differences are +0.46/+0.52 percentage
points at s8 and +1.42/+0.36 at s32. FID and agreement also improve in both
seeds at CFG1. The mean CFG1 convergence curve favors offsets at all four
recorded states. This supports keeping offsets available as a small optional
capacity addition. It does not isolate U versus V offsets, eliminate factor
gauges, establish variance preservation or determine their optimal rate.
Both offsets remain disabled by default. Higher generator width s32 still
outperforms s8 with either setting; transform rank remains 8.

The original controller exited with status 1 after all workers completed:
`summarize_dit_cifar10_screen.py` omitted `factor_offsets` from architectural
differences and raised `Training or evaluation settings differ across variants`.
The fix permits this architecture flag while preserving the comparison of
other training/evaluation settings. Twenty-two focused checks passed, including
six rejection cases for unplanned LR, beta2, step budget, split, FID count or
sampler changes. The previous feature/preflight validation did not exercise
the final mixed off/on reporting comparison; this regression covers that gap.

Recovery used `recover_dit_factor_offsets.py` and a separate frozen reporting
runner under `runs/dit-factor-offset-20261010T193609Z/reporting-recovery`.
The original auditor/model dependencies are unchanged; only the summarizer
differs. Original failed protocol, controller logs and training source hashes
are preserved alongside a recovery manifest. No training or GPU evaluation
was repeated, and no training/evaluation outputs were edited. All 13 full audit
checks passed, including exact configs, paired initialization/data/noise/label
streams, zero start and gradient activation, source/runtime/evaluator hashes,
successful worker exits, claimed GPU identities, two-worker cap, evaluation
denominators, no weights/test evaluation and the storage bound.

The recovered evidence SHA256 is
`ed0866026309d79b47b0321c232e4839a79473c3cb6c1387dcc4e82a05bad0ac`.
Ten compact exports were hash-verified; retained raw artifacts are about
516 MiB, within the declared 4 GiB bound. All private supervisors are closed
and our GPU claims are released. The next planned track remains ViT token
self-conditioning; this status/recovery work launched no new experiments.

[Report and exact evidence](reports/dit-factor-offset.md) ·
[Recovery manifest](reports/dit-factor-offset-recovery.json) ·
[Export manifest](reports/dit-factor-offset-manifest.json).

## Forward offset default and renewed ViT study (2026-10-11)

The user accepts the observed offset gain as potentially noise or directionally
positive and elects to enable it as the working default for subsequent method
extensions. This supersedes the previous recommendation to keep it optional
in new recipes. New condition-generated U/V recipes explicitly enable both
zero-initialized static offsets; off controls remain available. Existing frozen
runners and legacy defaults are preserved for replay. Structured methods that
do not parameterize U/V factors need their own definition rather than blindly
adding redundant biases.

ViT now supports these offsets through the same exact activation-mixing
contraction, for both full-gradient and detached token conditions:

```python
h = block.norm2(tokens)                # [B,T,192], same static affine norm in all controls
c = h                                # or h.detach() for the detached control
a, b = to_coef(c).chunk(2, -1)         # each [B,T,s]; affine head
u_activations = (h @ U).view(B,T,s,r) # U: [192,s*r]
z = (u_activations * a[...,None]).sum(-2) + h @ U_offset
g = (z[...,None,:] * b[...,None]).flatten(-2)
delta = gain * (g @ V + z @ V_offset) # V: [s*r,768], V_offset: [r,768]
pre_act = base_fc1(h * (1 + scale(c)) + shift(c)) + delta
tokens = tokens + fc2(gelu(pre_act))  # original residual MLP; attention retained
```

`U_offset: [192,8]` and `V_offset: [8,768]` initialize at zero without
consuming RNG. V remains zero; U/coefficients remain nonzero. Offsets add46,080
parameters across six blocks. The factorization is still rank≤8 for each token;
the code includes all cross terms without generating complete token factors.
The largest mixing intermediate is `[B,T,s,8]`. Offsets use adapter LR and
zero weight decay as biases; the learned basis matrices retain0.05 decay.
This does not remove affine/factor gauges or guarantee variance preservation.

The new study uses28 early calibration pilots on seeds96/97, then eight
families × two fresh quality seeds98/99 for ten epochs (3520steps). Width192,
depth6, heads3, patch4, MLP hidden768, batch128, LR5e-4, AdamW(.9,.999),
fused compiled BF16, label smoothing0.1, clip1 and the same45k/5k CIFAR split
remain common. The warmup is5% of the3520-step horizon, including in500-step
pilots. The trainer's new optional horizon preserves old behavior when unset;
an exact-prefix test checks matching losses, gradients and diagnostics.

Full-gradient s24 at unit adapter LR is the update reference. Widths12/48,
detached s24 and static rank8+diagonal each try multipliers0.25/1/4; the static
control also tries16. Select the closest geometric update magnitude only among
rates whose paired per-seed ratios at100/250/500 stay in[0.5,2]. Accuracy never
selects rates. Any failed preflight/audit/calibration bound withholds quality.
The matched static MLP has hidden1483:4,341,772 parameters versus4,342,378
for s24 (606 fewer). s12/s48 have3,761,626/5,503,882 parameters. These widths
have different capacity/compute; matching only s24 is deliberate.

Read-only FP32 diagnostics run eagerly on the underlying model outside compiled
training. A separate CPU saved-storage probe complements CUDA training peak
allocation; it is a one-layer eager FP32 measurement, not compiled/GPU memory.
Quality runs preserve complete validation curves, fixed-token update intervals,
gradient activation and paired observed augmentation streams. This short trend
study cannot establish asymptotic quality or replace the later longer confirmation.
Dedicated speed benchmarks remain deferred.

Resource declaration before launch: at most two GPUs for preflight/calibration/
quality,3–6 estimated aggregate GPU-hours (about1.5–3 wall hours when available),
and a4GiB retained-artifact bound. Compact evidence and compiler caches (expected
at most3GiB) are retained until explicitly removed. Preparation requires8GiB
free; approximately312GiB is currently free on shared storage. No periodic,
recovery or final weights, official-test evaluation or new evaluator artifacts.

[Detailed plan](reports/vit-token-study-plan.md) · [Implementation](vit_token_study.py) ·
[Feature tests](test_vit_factor_offsets.py) · [Protocol tests](test_vit_token_study.py).

### Launch and preflight status

Workflow: `runs/vit-token-20261011T070503Z`. Forty-one CPU checks passed before
launch. Both GPU preflights finished12 steps/1312 images with the declared
parameter budgets and complete5k validation denominators; every DPLR adapter
parameter received gradients by step3. Their audits passed. Calibration is
underway on two claimed workers; quality remains gated on the28-pilot audit
and update bounds. Adapter-rate multipliers apply to the existing trainer's
entire conditional parameter group, including scale/shift, rather than only
the low-rank branch. Thus matched low-rank probe updates do not imply matched
diagonal optimization. The larger static MLP has a different base shape and
initialization; exact base-tensor pairing applies to controls with hidden768,
while observed augmentation streams are paired across all families.

Two launch issues are preserved openly. First, CPU saved-storage preparation
used `fork_rng()` without restricting CUDA devices and briefly initialized
contexts on all eight devices, though no model computation ran there. The
preparation process exited. The workspace helper now uses `devices=[]`, with
a regression test; active study workers remain capped at two. The running
frozen controller never calls this preparation-only probe.

Second, successful preflight workers were followed by a row-assembly TypeError
because `dict(**job, config=cfg)` supplied `config` twice. The original
controller exited1 before any calibration launch. Its failed protocol/logs
and original source snapshot remain preserved. A new separately frozen
orchestrator retains the launch config path and config object in distinct
fields, re-audited the completed preflights and continued the same bounded
plan. Seven protocol regression checks passed, including both launch fixes.
No preflight was repeated, and training sources/retention/budgets are unchanged.

The one-layer CPU saved-storage probe reports815,360/927,680/1,152,320 bytes
at s12/s24/s48 (FP32, two images×65tokens, excluding parameter storage).
These are eager saved-storage counts including the input/loss, not CUDA or
compiled peaks; the quality results will report whole-model training allocation.

[Launch validation](runs/vit-token-20261011T070503Z/launch_validation.json) ·
[Preflight audit](runs/vit-token-20261011T070503Z/preflight_validation.json) ·
[Recovery provenance](runs/vit-token-20261011T070503Z/preflight-reporting-recovery/manifest.json) ·
[Live plan](reports/vit-token-study-plan.json).

### Coarse calibration completed; bounded intermediate-rate refinement

All28 pilots and both preflights passed their individual training/evidence
audits. The calibration gate failed before any ten-epoch quality run. The fixed
ratio guard is0.5–2 at100/250/500 in each of seeds96/97. s12 at unit rate
is borderline low (minimum0.471); at4× it overshoots (maximum2.504).
The static control at4× remains low (0.230–0.340), while16× crosses the upper
bound (maximum2.241). s48×1 and detached s24×1 meet every bound. This is a
coarse-grid optimization issue; the bounds are retained rather than loosened.

Unit-rate early mean accuracies after500 steps are37.46%/37.30%/38.85% for
s12/s24/s48 and35.63% for detached s24. These are short pilots with different
update scales, not matched-update quality comparisons. They do not select
rates. In particular, wider s has larger unit-rate updates: s48 ratios are
about1.67–1.85, while s12 is about0.47–0.63. Static factors need substantially
larger adapter rates to approach the dynamic reference's function changes.

The separate refinement freezes exactly eight new cases before launch:
s12×2 and s48×0.5, plus static×8/×12, each on the same paired pilot seeds.
The intermediate rates come from update bracketing. Reuse the original28
pilots only after full audit and reference-hash checks, and validate unchanged
trainer/model/diagnostic sources, runtime, split, initial tensors and observed
augmentation streams across the new and original cohorts. No preflight is
repeated. Select from the exact36-case combined grid using the same ratio
bounds and update-only criterion; preserve all candidate diagnostics on failure.
If it passes, the original16 fresh-seed quality runs proceed with frozen rates.
No further search is automatically added. The original failed protocol and
controllers are retained, rather than relabeled complete.

The new artifact resource envelope is declared separately: two GPU workers,
2–4 estimated aggregate GPU-hours,4GiB retained compact evidence/compiler
caches, and8GiB minimum free-space headroom (about312GiB free now). No saved
weights, checkpoints, test evaluation, grids or evaluator artifacts; old
retention is unchanged. Nineteen focused CPU checks passed for refinement,
original protocol and factor-offset functionality.

[Original gate report](reports/vit-token-coarse-calibration.md) ·
[Refinement plan](reports/vit-token-refinement-plan.md) ·
[Refinement implementation](vit_token_refinement.py).

Refinement launch: `runs/vit-token-refinement-20261011T185940Z`. Preparation
re-audited both old preflights and all28 pilots, pinned the failed protocol and
pilot-summary hashes, and verified unchanged trainer/model/diagnostic source
hashes. The new source/config snapshot and19-pass CPU test log are retained in
the new workflow. Eight pilots are running with two claimed workers; the
original16 quality runs remain gated. Existing evidence/retention are unchanged.
[Live refinement plan](reports/vit-token-refinement-plan.json).

Calibration passed on 2026-10-11 at 19:04 UTC. All eight additional pilots
passed the evidence audit, and the combined 36-case grid passed the unchanged
selection gate. Frozen adapter LR multipliers are s12×2, s24×1, s48×0.5,
detached s24×1 and static rank8+diagonal×12. Selected early low-rank update
ratios span 0.778–1.397 across both seeds and all three probes. Rates were
selected using updates; validation accuracy played no role.

The 16 fresh-seed, ten-epoch quality runs have started with at most two claimed
GPU workers. Factor offsets remain enabled in dynamic methods. No weights or
checkpoints are saved. The quality comparison and final audit are pending; the
original coarse-grid failure remains preserved.

[Audited calibration selection](reports/vit-token-refinement-calibration.json).

ViT quality completed on 2026-10-11 at 19:26 UTC: all16 runs and the frozen
workflow audit passed. Mean validation accuracy is65.96% plain,69.04% diagonal,
66.64% static rank8+diagonal,63.16% matched static capacity,68.40% DPLR s12,
69.82% s24,70.05% s48 and68.72% detached s24. s24 beats diagonal in both seeds
by0.48/1.08 points; s48 versus s24 is mixed (−0.56/+1.02). Fresh-seed early
update bounds transfer, but the static control's12× adapter LR also scales
its affine head, and endpoint combined/base RMS is11.69–19.68. Separate
affine/low-rank rates before stronger claims. s24 remains a provisional
practical choice; these two-seed ten-epoch results do not settle long-run
quality. All private controllers are closed and our claims released; no
weights/checkpoints/test evaluation or additional GPU runs.

[Completed comparison and limitations](reports/vit-token-refinement.md).

The next ViT study is launched in `runs/vit-split-lr-20261011T214507Z`.
`train_cifar10.Config` now supports optional `affine_lr_multiplier` and
`lowrank_lr_multiplier`; each falls back to the historical shared multiplier
when unset. Affine means the condition-generated scale/shift heads. Low rank
includes the coefficient projector, learned U/V bases and factor offsets, or
static U/V. Original LayerNorm affine parameters retain the base rate. No
architecture, parameter-count or transform-memory change is introduced.
Explicit unit rates are recorded in the new forward ViT templates, while
legacy unspecified recipes retain the same optimizer topology and updates.

The frozen plan runs two compiled GPU preflights, 20 fresh calibration pilots
on seeds 100/101, then 16 fresh 30-epoch quality runs on seeds 102/103 if all
selected low-rank ratios stay within 0.5–2. Affine LR is fixed at 1 except
for the declared static joint12 diagnostic. Plain, diagonal, matched static
capacity, DPLR s24/s48, static unit/split/joint12 controls are included.
Pilots use 1,200 steps of the full 10,560-step schedule; old pilots and weights
are not reused. The detached-condition diagnostic is not repeated here.

The 58-pass CPU test log includes legacy trajectory equivalence, split role
coverage/decay, group schedule prefixes and quality failure guards; one
CUDA-specific exact-layer check is skipped. At most two of our GPUs are used,
with an estimated 3–5 aggregate GPU-hours and a 4 GiB artifact bound. No
weights, checkpoints or official-test evaluation; prior evidence is preserved.

[Independent-rate protocol](reports/vit-split-lr-plan.md) · [Live state](reports/vit-split-lr-plan.json).

Both compiled GPU preflights finished and passed the frozen evidence audit.
All applicable adapter parameters received nonzero gradients by step3. The
20 fresh calibration pilots are now underway; the 16 longer quality runs
remain gated on the declared update bounds. No sources or recipes changed
after launch.

[Audited preflight evidence](reports/vit-split-lr-preflight.json).

The original 30-epoch study was interrupted after four completed quality runs.
At the October 12 check, the container had started at 01:21:47 UTC, every old
worker was dead, and its last recorded progress was October 11 at 22:12:57 UTC.
The exact interruption time/cause is unknown. Calibration passed: s48 low-rank
LR 0.5 and static split LR 16, with affine LR 1. The original stale `quality`
protocol and all completed/partial outputs remain unchanged.

Recovery runs in `runs/vit-split-lr-recovery-20261012T013129Z`. Both original
preflights, all 20 pilots and the four completed workers passed a fresh audit.
The remaining 12 cases restart from original seeds/recipes without weights,
using a byte-identical original training snapshot and at most two GPU workers.
Thirty-two recovery/split-rate CPU checks passed. The wider static control's
completed accuracies are 77.40%/78.14% (mean 77.77%); static split seed 102 is 78.72%
and joint12 seed 103 is 78.68%. DPLR endpoints are still pending, so there is no
complete method ranking. Original/recovery cohorts are checked for matching
sources/runtime/split/initialization and observed streams at final completion.
No model weights, checkpoints, retuned rates or additional quality cases.

[Recovery plan and preserved evidence](reports/vit-split-lr-recovery-plan.md).

The recovered 30-epoch study completed on October 12 at 02:13 UTC. All 16 cases
and all seven final recovery audit checks passed. Mean validation accuracy:
plain 78.45%, diagonal 78.56%, matched static capacity 77.77%, DPLR s24 79.94%,
s48 79.19%, static unit 79.17%, static split 78.55% and static joint12 78.80%.
s24 minus diagonal is +1.54/+1.22 points across the two seeds; versus unit
static it is +1.52/+0.02. s48 adds parameters without improving the mean.
Selected early update bounds transfer to the fresh seeds, while keeping the
affine LR at 1 keeps static split's endpoint combined/base RMS 2.21–2.63 versus
13.80–21.02 for joint12. The low-rank rates differ too, so this is not an
affine-only ablation. Unit static is the stronger static quality control,
despite much smaller early low-rank updates. Confirm against that control
and diagonal with more fresh seeds before larger projector searches.
Recovery supervisors are closed, our GPU workers are inactive, and no weights/
checkpoints/test evaluation or new GPU jobs were produced by this status check.
Original interruption evidence and partial/completed worker files are unchanged.

[Completed comparison and diagnostics](reports/vit-split-lr.md).

The wider static MLP's below-plain result warrants an optimization control
before stronger capacity claims. Last-epoch augmented training accuracy is
85.43% plain versus 83.78% wider, with validation 78.45% versus 77.77%. Fixed
Linear initialization std 0.02 leaves wider fc2 fan-in 1483 versus 768; an
untrained CPU probe measured 1.379× greater initial MLP output RMS. A temporary
reference-gain correction gives 0.993×, but no training used it and it proves
no quality improvement. The revised next priority is a bounded LR/initial-gain
check for plain and wider controls, then fresh-seed confirmation with comparable
tuning budgets. Preserve the original recipes and historical evidence.

[Wider-control diagnostic and revised priority](reports/vit-capacity-diagnostic.md).

The wider-control plan now includes Xavier, explicit fan-in/Kaiming scaling
and named layer-specific mixes, with LR retained as an independent axis.
Fan-in scaling keeps fc1 scale fixed as hidden width changes and compensates
fc2 fan-in; Xavier also changes fc1 scale through fan-out. Neither choice
establishes width-invariant training or selects the GELU/residual gain. Probe
absolute branch RMS as well as the wide/narrow ratio, then freeze a bounded
initializer-by-LR comparison with comparable tuning budgets. Preserve the
conditional zero-init rules. No initializer default or training source changed
and no new GPU jobs were launched.

[Initializer and LR candidate plan](reports/vit-capacity-initializer-plan.md).

The initializer/LR control study completed in
`runs/vit-initializers-20261012T070919Z`. Plain, wider plain and DPLR s24
each receive four distinct initializer functions × three global LRs × two
tuning seeds, all trained for 30 epochs. Both corrected compiled GPU
preflights passed their audits; 72 CPU checks passed with one CUDA-specific
exact-layer check skipped. The mixed recipe uses an explicit sqrt(2) input
gain before GELU plus a Xavier output std and Gaussian-GELU output-gain
anchoring; the first gain 1 mix was caught as a duplicate of fan-in at narrow
width before tuning. Its preliminary preflights/source/logs remain preserved,
with zero tuning runs launched there. The corrected screen has 72 eligible
plus two diagnostic reference-gain runs. Four fresh confirmation seeds and
legacy anchors follow frozen selection, with no repeated identical cases.

Initialization only rescales base MLP weights, preserving other base values,
RNG draws and zero conditional/classifier initialization. It adds no parameters
or runtime operators. Fixed-token complete MLP function updates now accompany
the existing low-rank probes for all models. The CPU initial RMS probe records
plain legacy ≈0.0815, wider legacy ≈0.1134, wider reference/fan-in ≈0.0816/0.0818,
wider Xavier ≈0.0618 and wider mixed ≈0.0861. Gaussian anchoring is a model of
initial scale, not exact learned-distribution or update invariance.

The frozen budget is 10–16 aggregate GPU-hours with at most two of our GPUs
and a 4 GiB artifact bound for the corrected workflow; no weights/checkpoints,
official-test evaluation or evaluator downloads. Historical runs and the
legacy initializer default remain unchanged.

[Initializer/LR protocol](reports/vit-initializer-study-plan.md) · [Live plan](reports/vit-initializer-study-plan.json).

The initializer study completed October 12 at 12:15 UTC: 74 full-horizon
tuning/diagnostic cases and 24 fresh confirmation cases, all seven audits passed.
Four-seed selected validation means ± sample SD are plain 79.44 ± 0.34%,
wider 78.64 ± 0.54%, DPLR 79.35 ± 0.27%. Xavier improves plain on every
confirmation seed and improves wider on three of four, but wider still trails
plain on all four. Legacy DPLR averages 79.445%, essentially equal to selected
plain. There is no demonstrated DPLR advantage against equally tuned plain
here despite about 61% more parameters; previous cohorts remain historical
evidence, not confirmation against this stronger control. Selection remains
frozen and defaults unchanged. Same-holdout tuning and four seeds limit the
claim. Review optimization/generalization and matched-recipe controls before
expanding projector capacity. Our private supervisors are closed and GPU
workers inactive; no new jobs, weights, checkpoints or official-test evaluation.

[Completed initializer/LR comparison](reports/vit-initializers.md).

The fixed diagnostic completed the Xavier plain/DPLR × global-LR matrix
on seeds 106–109. Eight audited matching cells are reused; only eight missing
30-epoch cells are added: plain at 2.5e-4 and DPLR at 5e-4. Both paired method
gaps and their LR interaction are reported, without a new recipe selection.
These seeds and the holdout were already inspected, so this is an exploratory
follow-up. Training sources/runtime and all other settings remain identical.
The workflow `runs/vit-matched-lr-20261012T164445Z` is limited to two of our
GPUs, 1–2 estimated aggregate GPU-hours, a 4 GiB artifact bound and no weights,
checkpoints or official-test evaluation. Fifty-eight CPU checks and a read-only
reference quality audit passed before launch.

[Matched Xavier/LR protocol](reports/vit-matched-lr-plan.md) ·
[Live plan](reports/vit-matched-lr-plan.json).

The matched-Xavier/LR follow-up completed October 12 at 17:13 UTC, with
eight new runs and eight audited reused cells. All seven final audits passed.
Plain/DPLR four-seed means are 79.465%/79.350% at 2.5e-4 and
79.440%/79.525% at 5e-4. Paired method gaps −0.115/+0.085 points are small;
the large two-tuning-seed LR flip did not reproduce. No new recipe winner
is selected. DPLR fits augmented training better at both LRs without clear
validation improvement, keeping generalization/regularization and stronger
diagonal/static controls ahead of projector expansion. These seeds and the
holdout were already inspected, so this is exploratory. Private supervisors
are closed; our GPU workers are inactive, with no weights or test evaluation.

[Completed matched-LR comparison](reports/vit-matched-lr.md).

The completed bounded check measures condition magnitude and separate value,
affine-condition and low-rank-condition task-loss gradients. Current DPLR,
condition-only RMS normalization and condition-gradient scale 0.5 each get
two new seeds and 1,200 steps of the existing 10,560-step schedule. Two compiled
GPU preflights gate all six diagnostic prefixes; no early accuracy selection
or automatic longer training. The raw-model probes use four fixed validation
images in FP32 eval-mode, with forward/gradient reconstruction and explicit
undefined alignment for zero routes. Ninety-seven CPU checks passed, including
unchanged training trajectories with diagnostics on/off. Defaults preserve
the prior function and RNG; historical audits use their frozen runners.

Workflow `runs/vit-condition-routes-20261013T010835Z` retains the two-GPU cap,
a 4 GiB artifact bound and no weights, checkpoints or official-test evaluation.
These interventions have no established benefit; this tests the mechanism.
[Condition-route protocol](reports/vit-condition-route-plan.md) ·
[Live plan](reports/vit-condition-route-plan.json).

The condition-route check completed October 13 at 01:17 UTC: two compiled
preflights, six 1,200-step prefixes and all seven final audits passed. Reference
condition RMS at step 1,200 averages 0.976, condition/value gradient norm ratio
0.614, and cosine +0.297. All 36 reference block-level global cosines at steps
100/500/1200 are positive; some token-level opposition exists. Extra condition
RMS normalization slows early learning in both seeds (mean −4.96 points);
half-gradient scaling is mixed (−1.16/+0.82). These are local FP32 fixed-image
early probes, not final-quality selection or proof of absent gradient conflict.
Keep defaults unchanged and defer longer intervention training; external
conditioning remains a separate next question. Private supervisors are closed
and our GPU workers inactive, with no weights, checkpoints or test evaluation.

[Completed condition-route measurements](reports/vit-condition-routes.md).

The external-conditioning track now checks global-LR fairness before a broader
architecture claim. Plain AdaLN-Zero, wider AdaLN and offset-enabled DPLR r8/s32
each receive three LRs (2e-4/4e-4/8e-4) × two complete 30k tuning seeds. Frozen
final CFG2 FID selection gates twelve 30k confirmation runs on four fresh seeds.
Both CFG1/2 and paired differences are retained. Wider hidden width 1715 matches
DPLR within 920 parameters (0.0074%). The original DiT training/data/diagnostic
snapshot is reused byte-for-byte; recent ViT changes are excluded. The condition
remains normalized class-plus-timestep `[B,256]`, broadcast over image tokens.

Workflow `runs/dit-external-lr-20261013T070213Z`: two compiled preflights gate
18 tuning +12 confirmation runs, at most two of our GPUs, estimated 10–15
aggregate GPU-hours and a 4 GiB artifact bound. Fifty-four CPU checks and eight
checks inside the frozen snapshot passed; no training weights, checkpoints,
official-test evaluation or evaluator downloads. This is a bounded LR search
on the reused research holdout, rather than sequence-conditioned attention.
[External-conditioning LR protocol](reports/dit-external-lr-plan.md) ·
[Live plan](reports/dit-external-lr-plan.json).


The external-conditioning LR study completed October 13 at 12:59 UTC: two
compiled preflights, 18 tuning runs and 12 fresh-seed confirmation runs.
All nine final audits passed, including replay from the frozen runner. All
methods selected 8e-4, the highest tested LR; this grid does not establish
LR optima. Final four-seed CFG2 FID is 38.38 ± 0.62 for plain AdaLN,
37.20 ± 1.20 for parameter-matched wider AdaLN, and 33.89 ± 0.67 for DPLR.
DPLR beats both guided controls on all four seeds; class agreement is
79.76%/80.58%/89.01%. Unguided DPLR beats wider AdaLN on three of four seeds.
Measured concurrent training throughput is about 23% lower than wider AdaLN,
so the quality gain has an operational cost. These are 5k-image research
holdout metrics for one broadcast class/timestep-conditioned DiT, with no
sequence-conditioning or cross-attention conclusion. Private supervisors
are closed and our GPU claims released; no weights or checkpoints were saved.
No further GPU runs were launched.

[Completed external-conditioning LR comparison](reports/dit-external-lr.md).

## Roadmap steering: causal context and generator compression (2026-10-13)

Raise economical factor generators ahead of the sequence-conditioning pilot.
The current DiT adapter has 131,584 coefficient-head parameters versus
2,703,360 decoder/offset parameters; compression should target the latter,
while retaining a dense reference and measuring actual parameter, activation
and training-time costs. CP/Tucker or block compression of the factor decoders
and channel-modulated static factors are candidates. Diagonal-plus-low-rank
generator maps require an explicit coordinate/skip definition for rectangular
outputs. These choices change the generator function class; they are not
automatically equivalent replacements. Wider s can spend any demonstrated
savings. Retain zero-delta initialization and verify all trainable routes wake
up. No new implementation or GPU study is launched by this update.

The sequence-conditioning proposal is a controlled test of context access and
channel transformation. For example, a causal prefix can contain `(A,7)`,
`(B,3)` and `(C,9)`, followed by a query for B; the next-token target is 3.
The other key/value entries are distractors. Vary their number, distance and
key similarity under a frozen task-generation protocol, with held-out random
assignments. A transform fed only the current contextual token is a different
information-access design from a module that explicitly reads the prefix.
Do not attribute this access difference to transform expressivity.

Prefer a causal, unpooled initial pilot. Compare ordinary causal attention,
query-dependent masked retrieval feeding a diagonal conditioner or DPLR, and
retrieval without the additional conditional transform. Where cross-attention
reads a separate context sequence, every context state must have an explicit
availability boundary for the target position. Future tokens, future answer
states and bidirectionally encoded same-stream context must remain inaccessible.
Matched retrieval pathways isolate the diagonal-versus-low-rank question;
separately varying the access path addresses retrieval itself. The old
pooled-context comparison is no longer the preferred initial design.

An additional self-conditioning hypothesis is to reuse a causal early-layer
hidden sequence as the condition at later layers:

```python
c_anchor = anchor_norm(h_anchor)       # [B,T,q], causal contextual states
for layer in later_layers:
    h = layer.norm(x)                  # [B,T,d], current value states
    c = layer.condition_head(c_anchor) # aligned positions; no time mixing
    x = layer.forward_with_condition(x, h, c)
```

Each token receives its own aligned anchor state, rather than a last-token
vector broadcast to earlier positions. Anchor states must precede their
consumers and be computed with causal attention. Reuse separates the current
value representation from the condition representation at downstream layers,
but does not remove competing optimization demands: an attached anchor still
receives downstream value-path gradients and summed condition-path gradients.
A detached-anchor control removes the latter while retaining forward
conditioning. Earlier detached ViT results do not establish how this different
design would behave. Reuse can save repeated condition-trunk work if that trunk
is also shared, but does not by itself remove layer-specific decoders, factor
application costs or retained anchor activations. A separate condition stream
is another, more expensive hypothesis. No efficacy claim is made for these
unimplemented variants.

Keep external LR-boundary/control follow-ups as bounded studies. Generator
compression now takes higher priority; causal retrieval and the anchored
self-conditioning variant remain subsequent options. Dedicated isolated speed
benchmarking stays deferred, the two-GPU cap remains, and generic checkpoints
and final weights remain disabled.

## CP decoder compression: implementation and first screen (2026-10-13)

The first decoder-compression implementation keeps the existing independent
affine coefficient heads and sample-wise generated factors. Each dense
`[s,channel_dim,r]` basis is replaced by CP factors over its coefficient,
channel and transform-rank axes. The new `decoder_cp_width=p` is distinct from
coefficient width s and produced-transform rank r. Default p=0 preserves the
dense model function, state layout, RNG and tested training trajectory.

For U, learned arrays have shapes `[d_in,p]`, `[s,p]`, `[p,r]`; for V,
`[s,p]`, `[p,r]`, `[p,d_out]`. Weighted rank components have shape `[B,p,r]`,
and the produced U/V retain `[B,d_in,r]` / `[B,r,d_out]`. Training never
reconstructs the dense decoder basis tensors. Static U/V factor offsets remain
enabled, and generated transforms retain rank at most r=8. The CP channel
subspaces constrain expressivity; increasing s does not remove that constraint.
Three multiplicative factors also introduce additional scale gauges.

Input factor initialization matches dense marginal basis-entry variance in
expectation, with a zero output channel factor and nonzero companion factors.
All adapter routes wake in the CPU trainer checks. This is not a trained
variance-preservation or matched-functional-update claim. CP factor draws use
an isolated generator; identical coefficient heads and global RNG trajectories
are preserved within each s by discarding temporary dense initialization
tensors on CPU before moving the model to CUDA.

Workflow `runs/dit-decoder-compression-20261013T181112Z` launches five compiled
preflights, then ten complete 30k runs on two new seeds (120/121), gated by the
preflight evidence audit. Designs are dense s32, dense s64, CP s32/p64,
CP s32/p128 and CP s64/p128. Adapter counts are 2,834,944 / 5,587,968 /
909,824 / 1,606,144 / 1,803,264 respectively. The last spends some savings on
wider coefficients while retaining fewer parameters than the dense s32
reference. Both dense controls are freshly trained.

All designs use the same existing broadcast-conditioned DiT, offsets, rank8
gain, normalized adapter value input, global LR8e-4, relative adapter LR1,
BF16/compiled/fused training, 30k schedule and fixed CFG1/2 evaluation budgets.
No per-design optimizer calibration, selection, automatic expansion or longer
confirmation is launched. CP can increase decoder FLOPs despite parameter
savings; ordinary training time/memory and magnitude/update diagnostics remain
important. Dedicated speed benchmarking remains deferred.

Sixty-nine workspace CPU checks and twenty checks inside the frozen runner
passed, including explicit expanded-CP output/every-gradient equivalence,
gradcheck, default-dense trajectory parity, BF16 dtype, gradient wake-up,
diagnostic trajectory invariance, exact grids/configs and failure gating.
Original DiT training helpers are copied from the completed external-LR
snapshot; declared model/trainer/diagnostic/reporting integration changes are
pinned separately. Fifteen configs and all ten quality-model initializations
were verified on CPU without creating a CUDA context before launch.

At most two claimed workers including queued preflights; estimated 4–7
aggregate GPU-hours and a 4 GiB artifact bound. No weights/checkpoints,
official test evaluation, new evaluator downloads or causal retrieval/LM work.
[Frozen design and artifact protocol](reports/dit-decoder-compression-plan.md) ·
[Live plan](reports/dit-decoder-compression-plan.json).


All five compiled preflights completed and passed the frozen evidence audit,
including replay. Every adapter parameter tensor received a nonzero gradient
by step5 in every design. The ten full30k quality runs are now queued/running
under the two-worker cap. No quality conclusion is available from preflights.


The CP compression screen completed October13 at20:41 UTC; all eight audits
passed on frozen-runner replay. Guided FID is33.56±0.07 /32.81±0.29 for dense
s32/s64 and34.95±0.80 /34.34±0.50 /34.33±0.40 for CP s32/p64,
s32/p128 ands64/p128. CP saves36–68% of adapter parameters versus dense s32,
with worse guided FID in both paired seeds, about8–15% lower observed training
throughput and only1–3% lower whole-training peak allocation. Wider CP
coefficients have mixed FID effects. Fixed-probe function updates differ by
several-fold early, so optimization remains a confound. Keep dense as the
reference; test a freer channel-factorized decoder with bounded adapter-LR
controls next. Supervisors closed; no weights/checkpoints or official test.
[Completed CP compression results](reports/dit-decoder-compression.md).

## Channel-mode decoder core and adapter-LR screen (2026-10-14)

The next compression study releases CP's coefficient/rank core restriction
while keeping the same channel-basis width p128 and transform rank r8.
`decoder_tucker_width` selects a channel-mode Tucker-style factorization with
independent U/V cores `[s,p,r]`. They start as the paired CP basis/rank
products and then learn freely. This is distinct from the earlier conditional
r×r-core model with static outer factors. No original dense decoder basis is
reconstructed during training; generated factors remain sample-wise and
broadcast. Static offsets and zero V_channel initialization remain.

Initial CP/core channel matrices, coefficient heads and FP32 factor functions
are paired within arithmetic tolerance, with unchanged base/null weights and
RNG trajectories. BF16 reassociation and changed optimizer geometry remain
confounds. Channel/core maps retain a general invertible channel-basis gauge;
fewer multiplicative stages do not imply fewer gauges or easier optimization.
No new output normalization, gain or muP rule is introduced.

Workflow `runs/dit-decoder-core-20261014T071618Z` contains five compiled
preflights and twenty full30k runs: dense s32, CP s32/s64 at p128, and
channel-core s32/s64 at p128, each at adapter LR0.25/1 on fresh seeds122/123.
The quarter-rate point tests sensitivity suggested by CP's several-fold
larger early fixed-probe updates; it is not a calibrated update match. All
designs receive equal rate budgets, including dense. Actual optimizer group
ownership, counts, initial rates and logged schedule rates are audited.
All adapter parameters, including coefficient heads and offsets, share the
multiplier; common AdaLN/base parameters remain at global LR8e-4.

Adapter parameters are2.83M dense,1.61M/1.80M CP s32/s64, and2.05M/2.70M
channel-core s32/s64. The wider core total12,463,152 is1.04% below dense
s32's12,593,712; this is approximate rather than exact matching. p64 is
not expanded in this first core study. Keep the common30k schedule,
compiled/fused/BF16 training and final same-state CFG1/2 evaluation budgets.
All twenty cells train anew; no historical quality outcomes reused, no
automatic selection, longer confirmation or grid expansion.

Ninety-three workspace CPU checks and twenty-four frozen-runner checks passed,
including explicit expanded-decoder every-gradient equivalence, gradcheck,
dense/CP training compatibility, core pairing, BF16 dtype, quarter-rate
gradient wake-up/routing, diagnostic trajectory invariance and failure gates.
All twenty-five configs and twenty quality-model initializations/optimizer
routes were verified before launch without a CUDA context. The frozen source
snapshot preserves original data/training helpers with declared integration
and compact LR-logging overlays.

At most two of our GPU workers; estimated8–12 aggregate GPU-hours, about4–6
wall hours with two available GPUs, and a4GiB artifact bound. No weights,
checkpoints, official test, new evaluator downloads or causal retrieval/LM
launch. Dedicated speed benchmarking remains deferred.
[Channel-core/LR protocol](reports/dit-decoder-core-plan.md) ·
[Live plan](reports/dit-decoder-core-plan.json).


All five channel-core/LR compiled preflights passed the independent evidence
audit and frozen-runner replay. Every adapter tensor received gradients by
step5, including all64 tensors in each new channel-core model. Actual
optimizer group ownership/rates, initial core pairing and observed streams
were verified. The twenty full30k workers are now queued/running under the
two-worker cap. Preflights supply compatibility evidence, not quality results.


The channel-core/LR study completed October14 at12:13 UTC; all nine audits
passed on replay. Unit-rate CFG2 FID is33.34 dense s32,33.75 CP s32,33.98
CP s64,33.84 channel-core s32 and37.55±4.63 channel-core s64. Freer cores
show no consistent guided-FID improvement over CP. Quarter-rate effects are
mixed for wider designs and worse in both seeds for dense/CP/core s32.
The wide-core spread is a poor-quality outcome with finite completed evidence,
not a numerical failure. Keep dense as the reference and move to cheaper
channel-modulated static factors with static/dense/CP controls. No new default
selected. Supervisors closed; no weights/checkpoints or official test.
[Completed channel-core/LR results](reports/dit-decoder-core.md).


## Channel-modulated static factors (2026-10-14)

The completed core/LR screen motivates a cheaper restriction of the factor
generator, with fresh controls rather than a new default. Separate channel
scales multiply static input/output factors; they act inside the parallel
adapter delta and do not consume the base AdaLN scale/shift output. See the
[full tensor design and initialization](reports/dit-channel-factor-plan.md).

```python
# c [B,256], h [B,T,256]; r=8; output width1024
a, b = coefficients(c).chunk(2, dim=-1)  # each [B,s]
gu = 1 + a @ U_scale_basis               # [B,256]
gv = 1 + b @ V_scale_basis               # [B,1024]
U = gu[..., None] * U_static + U_offset  # [B,256,8]
V = gv[:, None, :] * V_static + V_offset # [B,8,1024]
pre_act = mlp_in(scale_shift(h,c)) + (h @ U @ V) / sqrt(8)
```

The scale bases start zero, coefficient heads nonzero, static U has variance
1/256, and static V plus both offsets start zero. Identity scales and a zero
delta preserve the starting function; trained variance preservation is not
established. Multiplicative and offset gauges remain. The static control
removes condition heads/scale bases while retaining paired static factors and
offset slots. It is condition independent only in this transform; common
AdaLN and upstream states remain class/time conditioned. Initial factors match
between static and both modulated widths in FP32/BF16; dense/CP Jacobians are
not matched to the channel designs. Rank remains at most8, including offsets.

Adapter counts: static163,840; channel s32/s64 623,104/1,082,368; CP s32/p128
1,606,144; dense s32 2,834,944. Two adapter LR points1/4 apply to every adapter
parameter, leaving common base/AdaLN at8e-4. All designs receive both points
and fresh seeds124/125. This tests a higher-rate direction after quarter-rate
mostly hurt earlier s32 recipes; it does not calibrate operator updates.
Five compiled higher-rate preflights gate twenty full30k runs. Report final
paired CFG1/2 quality, modulation-minus-static/dense/CP and fourfold-minus-unit
results, retaining both seeds. No automatic selection or grid expansion.

One hundred eighteen workspace and twenty-five frozen CPU checks passed,
including expanded outputs/every-gradient equivalence, gradcheck, identity
scales, all-route wake-up, BF16, old model/Adam compatibility, initialization
and RNG pairing, actual optimizer rates and failure gates. GPU preflights and
their independent audit are pending. Frozen sources and all25 configs are
pinned. Workflow `runs/dit-channel-factors-20261014T175748Z` uses at most two
GPU workers or claim waiters, no weights/checkpoints or official-test data,
a4GiB artifact bound and estimated4–6 wall hours. Per-token implicit execution,
causal retrieval/LM and dedicated speed benchmarking remain future work.


All five channel-factor compiled preflights passed the independent evidence
audit and frozen-runner replay. Every adapter parameter tensor received a
nonzero gradient by step5, including all64 tensors in each modulated model
and32 in the static control. Actual optimizer ownership/rates, paired initial
factors/base/head weights, data/noise/dropout streams and evaluator budgets
were verified. The twenty full30k quality runs have opened under the two-worker
cap. Preflight success is compatibility evidence, not a quality conclusion.
[Preflight replay](reports/dit-channel-factor-preflight.json).


The channel-factor study completed October14 at22:40 UTC; all nine audits
passed in independent frozen-runner replay. Dense s32/CP s32 at adapter LR1
reach33.59/34.25 CFG2 FID. Static/channel s32/channel s64 at adapter LR4
reach37.04/35.61/34.64. Higher LR improves every static/modulated seed,
and modulation beats static at4 in both seeds. Channel s64 uses61.8% fewer
adapter parameters than dense, but remains1.05 mean FID worse at these
different operating rates. LR4 severely degrades one dense and one CP seed;
retain both poor finite endpoints. No new default or automatic follow-up.
All20 full30k runs completed; supervisors closed, adalora claims released,
no weights/checkpoints or official test.
[Completed channel-factor results](reports/dit-channel-factor.md).


A focused adapter-LR follow-up launched October15 in
`runs/dit-channel-lr-20261015T061259Z`: dense s32, CP s32/p128 and channel s64
with equal multipliers1/2/4 on fresh seeds126/127, eighteen full30k runs.
The intermediate rate brackets the previous endpoints; repeating1/4 checks
seed sensitivity and whether severe high-rate degradation recurs. Three
compiled higher-rate preflights gate quality training; their audit is pending.
All36 reference files are preserved exactly; only controller/tests are added.
Thirty-six workspace and36 frozen CPU checks passed, all21 configs and18
model initializations/actual optimizer groups verified. At most two GPUs,
estimated4–5 wall hours,4GiB artifact bound, no weights/checkpoints or test.
The channel width was selected using the prior screen; this is fresh-seed
replication with a reused holdout, not independent architecture selection.
[LR follow-up protocol](reports/dit-channel-lr-plan.md) ·
[Live plan](reports/dit-channel-lr-plan.json).


All three adapter-LR follow-up compiled preflights passed the controller
evidence audit and independent frozen-runner replay. Every adapter tensor
received gradients by step5:48 dense,80 CP and64 channel s64. Actual optimizer
ownership/rates, paired initialization, observed streams, source and evaluator
budgets were verified. Eighteen full30k quality runs are queued, with two
active under the cap. Quality and final audits remain pending; early gradient
wake-up does not establish long-run stability.
[LR preflight evidence](reports/dit-channel-lr-preflight.json).


The fresh-seed adapter-LR follow-up completed October15 at10:28 UTC; all nine
audits passed on independent replay. Dense at1 reaches33.65 mean CFG2 FID;
CP at1/2 reaches34.18/33.90. Channel s64 at1/2/4 reaches36.82/35.78/35.58.
CP benefits modestly from2 in both seeds; channel2-to4 gains only0.20 mean.
Channel4 remains1.92 FID worse than dense1 with61.8% fewer adapter parameters.
One dense and one CP seed again severely degrade at4; all finite endpoints
are retained. No new default, automatic grid expansion or further runs.
All18 full30k workers finished, supervisors closed, adalora claims released,
no weights/checkpoints or official test.
[Completed fresh-seed LR follow-up](reports/dit-channel-lr.md).


## Structured generated operators: implementation and first screen (2026-10-16)

The research objective remains useful conditional expressivity for generator
parameters, activation memory and execution cost. Broader generated-operator
architectures are now prioritized after the compression/LR studies. Synthetic
retrieval and a new LM benchmark are deferred; keep the established diffusion
setup while changing the architecture.

The new square block and conditional Hadamard operators share a parallel
feature delta before the existing MLP input projection, with square diagonal
and low-rank controls. Sum before one shared projection GEMM; common AdaLN,
non-affine norms and gates remain. The old rectangular DPLR is a separate
reference, with different placement/output-subspace freedom. Block16 uses
1,147,136 adapter parameters versus1,212,928 for square rank8/s32 low-rank.
Hadamard uses400,128 versus396,032 for the wider square-diagonal s96 control;
its96-output coefficient heads start identically. Hadamard is a learned
conditional two-transform/three-diagonal/permutation chain inspired by
Fastfood, not the original random-feature kernel construction. All branches
start at zero; no trained variance/gain guarantee or speed benefit is assumed.

Workflow `runs/dit-structured-20261016T203614Z` has six compiled preflights
and22 fresh full30k runs on seeds128/129. Every adapter gets multipliers1/2;
plain AdaLN runs once per seed. Ninety workspace and90 frozen CPU checks
passed, including independent dense operators/every-gradient equivalence,
Hadamard norm checks, zero-delta/base pairing, one-projection integration,
BF16 diagnostic trajectory invariance, gradient wake-up, legacy Adam/RNG
compatibility, actual groups, finite grid and failure gates. Dynamic CPU
tracing of a nonzero Hadamard model passed. All28 configs/22 initial models,
43 frozen files,33 unchanged reference sources and package versions verified.
GPU preflight audit is pending.

At most two GPUs, estimated5–7 wall hours,4GiB artifact bound, compact evidence
and declared diagnostic grids only. No checkpoints/weights, official test or
evaluator downloads; isolated speed work, TT/FFT, shared nonlinear trunks
and causal retrieval/LM launches stay outside this bounded study.
[Structured operator protocol](reports/dit-structured-plan.md) ·
[Live plan](reports/dit-structured-plan.json).


All six structured-operator compiled preflights passed the controller audit
and independent frozen-runner replay. Every adapter tensor received a nonzero
gradient by step5, including all64 Hadamard tensors and32 block tensors.
Source/config/initialization/buffer pairing, actual optimizer groups/rates,
observed streams and evaluator budgets were verified. The22 full30k quality
runs have opened under the two-worker cap. Final quality/audit remain pending;
preflight success establishes compatibility, not a quality improvement.
[Structured preflight evidence](reports/dit-structured-preflight.json).


The structured-operator screen completed October17 at01:38 UTC; all nine
audits passed on independent replay. Plain AdaLN reaches39.43 CFG2 FID;
rectangular DPLR1/2 reaches33.93/33.59. Square low-rank1 reaches35.20; block2
35.63 and Hadamard2 36.62. At LR2, Hadamard beats its nearly budget-matched
diagonal38.38 in both seeds (−0.97/−2.56 FID), with22.1% lower observed
training throughput and lower peak allocation. At LR1 its FID gaps are mixed.
Block2 versus square-low-rank1 gives mixed paired quality,8.8% higher observed
throughput and19.0% higher allocation. Square low-rank2 includes one poor
finite endpoint51.15 FID. Retain all rates/seeds; no new default or follow-up.
All22 full30k runs finished; supervisors closed, adalora claims released,
no weights/checkpoints or official test.
[Completed structured-operator results](reports/dit-structured.md).


Conditional tensor-train operators are now implemented for the next bounded
DiT study. Four paired channel modes `[4,4,4,4]`, TT bonds6/11 and one shared
affine latent32 give420,608/1,180,928 adapter parameters. TT bond rank is not
ordinary matrix rank. Training applies mode contractions without a dense
operator; token-axis contraction intermediates still require measurement.
A zero final core gives an exact zero delta, with nonzero earlier static cores
and audited gradient wake-up. The24-run plan uses diagonal, square low-rank
and block controls plus the contextual rectangular reference, two fresh seeds
and the existing two adapter rates. Two-GPU cap, no weights/checkpoints.
Validation and compiled GPU preflight must pass before quality begins.
[Conditional TT protocol](reports/dit-tt-plan.md).


TT workflow `dit-tt-20261017T045056Z` has launched its six compiled CUDA
preflights; the24 full30k runs are gated on their audit. All109 workspace
and109 frozen CPU checks pass. Thirty configs,24 initial models/optimizer
groups,47 frozen source files and39 unchanged reference files verified.
Nonzero TT11 dynamic fullgraph CPU tracing/backward passed in BF16 at batch2/3;
CUDA compatibility and quality remain pending. Package versions match the
completed structured screen. Retention remains4GiB, two-GPU cap, no weights.
[TT CPU validation](reports/dit-tt-cpu-validation.json) ·
[TT live plan](reports/dit-tt-plan.json).


All six TT-study CUDA preflights passed both the controller audit and an
independent frozen-runner replay. Every adapter tensor received a nonzero
gradient by step5, including all80 tensors in each TT variant. Source/config,
paired streams/initializations, actual optimizer groups/rates, claim identity
and evaluator budgets verified. Short-preflight whole-run peak allocations
were1.91GiB (TT6) and2.37GiB (TT11); these are not sustained quality-training
resource results. The24 full30k quality runs are now open under the two-worker
cap. Final quality and audit remain pending; no model weights/checkpoints.
[TT preflight evidence](reports/dit-tt-preflight.json) ·
[TT startup status](reports/dit-tt-startup.json).
