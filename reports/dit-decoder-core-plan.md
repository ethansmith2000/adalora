# Channel-factorized decoder cores and adapter-LR sensitivity

The completed CP screen reduced adapter parameters by 36–68% relative to
dense s32, but guided FID was worse in both seeds, training throughput was
8–15% lower and whole-training peak allocation only 1–3% lower. Early
fixed-input function updates were several times larger than the dense
reference. This motivates a bounded test of decoder expressivity together
with adapter-LR sensitivity; the observed gap is not solely attributed to
representation capacity.

## Produced factors and decoder parameterization

All models retain the same independent affine coefficient heads, static factor
offsets, parallel MLP-input delta and produced-transform rank r=8:

```python
a, b = Linear(256, 2*s)(c).chunk(2, dim=-1)   # each [B,s]
```

CP constrains the coefficient/rank part of each decoder to a product:

```python
# U_basis [s,p], U_rank [p,r], U_channel [d_in,p]
u_weights = (a @ U_basis)[..., None] * U_rank  # [B,p,r]
U = U_channel @ u_weights                     # [B,d_in,r]
```

The new decoder compresses only the channel mode, with an unrestricted
coefficient/channel/rank core:

```python
# U_core [s,p,r], U_channel [d_in,p]
# V_core [s,p,r], V_channel [p,d_out]
u_weights = (a @ U_core.flatten(1)).view(B,p,r)
v_weights = (b @ V_core.flatten(1)).view(B,p,r)
U = U_channel @ u_weights                     # [B,d_in,r]
V = v_weights.transpose(-1,-2) @ V_channel    # [B,r,d_out]
U = U + U_offset                             # static [d_in,r]
V = V + V_offset                             # static [r,d_out]
delta = (h @ U @ V) / sqrt(r)                 # h [B,T,d_in]
```

This is a Tucker-style channel-mode factorization. It is distinct from the
existing conditional r×r-core model with static outer U/V factors. The
condition still determines both generated outer factors; no nonlinearity is
inserted between applying U and V.

The channel-mode basis width p=128 remains separate from coefficient width
s=32/64 and transform rank r=8. Both CP and the new decoder retain fixed
learned channel subspaces, with additional static factor offsets. Training
does not reconstruct the original dense `[d_in,s,r]` / `[s,r,d_out]` decoder
bases. Sample-wise factors retain the same broadcast memory shapes.

Per-layer decoder parameter counts, before offsets, are:

```python
dense = s*r*(d_in+d_out)
cp = p*(d_in+d_out+2*s+2*r)
channel_core = p*(d_in+d_out+2*s*r)
```

## Fixed grid and initialization

| Design | s | p | Total parameters | Adapter parameters |
|---|---:|---:|---:|---:|
| Dense reference |32|—|12,593,712|2,834,944|
| CP |32|128|11,364,912|1,606,144|
| CP |64|128|11,562,032|1,803,264|
| Channel core |32|128|11,807,280|2,048,512|
| Channel core |64|128|12,463,152|2,704,384|

The wider channel-core design has 1.04% fewer total parameters than dense s32;
this is approximate budget comparison, not exact matching. Paired CP controls
at each s separate the decoder-core change from widening coefficient heads.
The smaller p64 CP tradeoff remains documented; this first core study fixes
p128 rather than expanding every axis at once.

Channel-core initialization folds the paired CP factors:

```python
U_core = Parameter(U_basis[..., None] * U_rank)
V_core = Parameter(V_basis[..., None] * V_rank)
# Discard U_basis/U_rank/V_basis/V_rank; cores now learn freely.
```

CP and channel-core models share the same initial channel matrices, heads,
static offsets, base/null weights and global RNG trajectory at each s/seed.
Their initial input factor functions match in FP32 within a declared arithmetic
tolerance; BF16 reassociation is not claimed identical. V_channel starts zero,
with nonzero companion factors/cores. Delta and AdaLN gates remain zero at
start, and every trainable route must wake in early audits. Core initial
entries are correlated CP products, not independent Gaussian draws.

A learned channel basis and unrestricted core retain a general invertible
basis gauge: changes in the channel basis can be offset in the core. Fewer
multiplicative stages do not imply fewer gauges or easier optimization. We
apply no new output normalization, orthogonalization, gain or muP scaling rule.
No trained adapter is decomposed, loaded or resumed. Temporary dense/CP
initialization tensors exist only on CPU and are discarded before CUDA transfer.

## Equal adapter-LR budgets and evaluation

Each of five designs receives adapter LR multipliers 0.25 and 1, each on
two fresh seeds122/123: twenty complete 30k runs. The multiplier applies to
all adapter parameters: coefficient heads, decoders/cores and static offsets.
Base weights and common AdaLN heads remain at global LR8e-4. Initial optimizer
group ownership/counts/rates and actual logged group LRs are audited against
the schedule; lower LR is not inferred from a config label alone.

The quarter-rate point tests sensitivity suggested by several-fold larger CP
fixed-probe updates. It is not a calibrated function-update match. All designs
get the same two-point rate budget, including dense. Report both rates and
paired per-seed sensitivity/structural differences. No automatic winner
selection, confirmation, grid expansion or implicit resume. A best observed
two-point recipe is exploratory with two seeds, not an independently confirmed
optimum. The global-LR boundary from the earlier study remains unresolved.

Keep width256, depth8, heads4, patch4, MLP hidden1024, batch128, gain1/sqrt(8),
normalized value input, class/timestep `[B,256]` broadcast over64 image tokens,
fused AdamW `(0.9,0.999)`, zero weight decay, clip norm1, BF16, dynamic compile,
EMA0.999, class dropout0.1, 50-step warmup and a fresh30k cosine schedule
with minimum LR ratio0.01. Reuse the original frozen DiT data/training helpers
with only declared model/trainer/diagnostic/reporting integration overlays.
Historical quality outcomes are context only; all twenty cells train anew.

Five compiled12-step preflights cover every architecture, 128/128/72 ragged
batches, gradient wake-up and the full evaluator/diagnostic path before full
runs. These use unit adapter LR because model compilation/shape is identical
across rates. Quarter-rate routing/wake-up is tested in the CPU trainer;
actual group LRs and early wake-up are also audited for every full worker.
Preflight failure preserves evidence and withholds all full runs.

Raw/EMA validation every5k; CFG1 generation at10k/20k; final CFG1/2 use the
same EMA state and sampling noise. DDIM50, 5k real/5k generated images,
research45k/5k split seed123, validation noise seed2027, sampling seed2028,
and the same hash-pinned Inception/auxiliary classifier. Primary comparisons
use final30k CFG2 FID; CFG1 and class agreement are also retained. No
best-intermediate selection or official-test evaluation. Fixed independent
condition magnitudes, function updates, live branch magnitudes and gradients
remain diagnostics, not quality surrogates or proofs of variance preservation.

## Resources, evidence and limits

At most two of our GPU workers or claim waiters. Estimated8–12 aggregate
GPU-hours, about4–6 wall hours with two available GPUs. Require8GiB free
before launch for the4GiB artifact bound and shared headroom. Retain compact
configs, sources/runtime/seeds/splits, metrics/logs/diagnostics/audits/reports,
compiler caches expected below3GiB, twenty final100-image CFG1 grids and five
preflight10-image grids until explicitly removed. No periodic/recovery/final
weights, generated datasets, new evaluator downloads or retention changes to
prior workflows. Checkpoint policy is unchanged.

CPU checks cover expanded-decoder outputs and every gradient, gradcheck,
absence of saved dense decoder bases, existing dense/CP function/gradient/Adam
trajectory parity, paired core functions and RNG, BF16 dtype, all-route
wake-up, diagnostic trajectory invariance, actual optimizer groups/rates,
exact finite grid, config roundtrips and failure gates. Independent GPU
audits additionally pin source/runtime/evaluators, observed data/noise/dropout
streams, same-state guidance budgets, successful exits, claims and storage.

The reused research holdout/evaluation noise, two seeds and restricted LR
grid limit conclusions. This study changes decoder expressivity and optimizer
geometry together; matching initial functions does not disentangle them fully.
Concurrent throughput and whole-training allocation are descriptive. Dedicated
speed benchmarking remains deferred. Causal retrieval, anchored token
self-conditioning and causal-LM integration are subsequent tracks.
