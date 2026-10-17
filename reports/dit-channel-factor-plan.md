# Channel-modulated static factors: bounded quality and LR screen

The completed core/LR study did not establish a consistent quality benefit
from freeing CP cores. Dense s32 at unit adapter LR reached33.34 mean guided
FID; CP s32 and channel-core s32 reached33.75/33.84, with mixed paired gaps.
Wider cores were variable. Quarter adapter LR worsened most s32 recipes.
This follow-up tests a cheaper conditioning restriction already on the
roadmap, rather than selecting a new default from those two-seed results.

## Tensor design and placement

All models retain the common AdaLN-Zero heads/gates, non-affine pre-norm,
normalized value input and parallel MLP-input delta. For the new design:

```python
# c [B,256]; h [B,T,256]; output hidden width1024; r=8
a, b = Linear(256, 2*s)(c).chunk(2, dim=-1) # [B,s] each
gu = 1 + a @ U_scale_basis                # U_scale_basis [s,256]
gv = 1 + b @ V_scale_basis                # V_scale_basis [s,1024]
U = gu[..., None] * U_static + U_offset   # [B,256,r]
V = gv[:, None, :] * V_static + V_offset  # [B,r,1024]
delta = (h @ U @ V) / sqrt(r)             # [B,T,1024]
pre_act = mlp_in(scale_shift(h,c)) + delta
```

The delta does not consume the base scale/shift branch's output. Input/output
channel gains act only within the low-rank branch. Row/column scaling can
move the factor subspaces with condition, unlike gains applied only to rank
coordinates between fixed factors. It remains a restricted factor generator,
not an arbitrary dense conditional matrix. Produced rank stays at most r=8,
including the factor offsets. Scalars gu/gv can change sign during learning;
no positivity constraint, cap, normalization or orthogonalization is added.

This replaces the per-layer decoder term `s*r*(d_in+d_out)` with
`s*(d_in+d_out)` plus static factors and offsets. It does not compress the
whole model by that ratio. Generated U/V remain sample-wise and reused over
64 image tokens. Per-token implicit application is a later implementation
option; this study does not claim its memory or speed benefits.

## Initialization and static control

U_static has independent normal entries with variance1/d_in; V_static and
both static factor offsets start zero. The coefficient heads are nonzero,
while U_scale_basis and V_scale_basis start zero. Thus gu/gv start exactly1,
delta starts zero, and the nonzero head/static input factor allows the scale
decoders and then coefficient heads to learn once V opens. Zeroing both
coefficient heads and scale decoders would create dead projection routes and
is explicitly avoided. All trainable parameters must wake in early audits.

The static control keeps the same U_static/V_static/U_offset/V_offset slots
and removes every condition-head/scale-decoder parameter. Its transform is
condition independent, while common AdaLN and upstream hidden states remain
class/time conditioned. Redundant static factor/offset slots are retained to
match the shared parameter routes and starting factors, rather than silently
changing the control's optimization. They do not double the rank bound.

Static and both modulated widths have identical initial factors in eager FP32
and BF16 CPU checks. Base/null weights match across all designs; same-s
non-static coefficient heads and RNG trajectories are paired. The generator
uses an isolated seed `training_seed +100000+adapter_index`. Temporary dense
initialization tensors are discarded on CPU before CUDA transfer. Dense/CP
initial factor functions and Jacobians are not matched to the channel models:
their random condition-dependent input factors differ. All share zero delta.

Identity scales and zero delta do not establish trained variance preservation
or a muP rule. Products and independent offsets retain parameter gauges. No
proof that those gauges explain previous results is assumed.

## Fixed grid and parameter budgets

| Design | s | Adapter parameters | Total parameters |
|---|---:|---:|---:|
| Dense DPLR reference |32|2,834,944|12,593,712|
| CP reference, p128 |32|1,606,144|11,364,912|
| Static factor/offset control |—|163,840|9,922,608|
| Channel-modulated |32|623,104|10,381,872|
| Channel-modulated |64|1,082,368|10,841,136|

Every design receives adapter LR multipliers1 and4, each on fresh seeds124/125:
twenty full30k runs. The previous quarter-rate point did not help most s32
recipes. This screen tests the higher direction for the new, variance1/d
static input factors, with equal rate budgets for dense/CP/static controls.
Fourfold LR is exploratory, not calibrated function-update matching.
Global base/AdaLN LR stays8e-4; the adapter group therefore starts at8e-4 or
3.2e-3 before warmup/schedule scaling. The multiplier covers every adapter
parameter, including heads, scale decoders and offsets. Actual parameter-group
ownership/counts/initial rates and logged rates are audited.

At unit and higher rates report paired modulation-minus-static,
modulation-minus-dense/CP, and fourfold-minus-unit results, for both seeds
and CFG scales. Parameter/FLOP budgets are not matched; this is a quality/cost
comparison and a static-transform control. No automatic recipe selection,
grid expansion, implicit restart or longer confirmation. Two seeds and two
rates do not establish optima or independent confirmation.

## Training and evaluation

Keep the existing width256/depth8/heads4/patch4/MLP1024 DiT, rank8,
gain1/sqrt(8), normalized class-plus-timestep `[B,256]` condition broadcast
over image tokens, batch128, fused AdamW `(0.9,0.999)`, weight decay0,
clip norm1, BF16, dynamic compilation, EMA0.999, class dropout0.1,
fresh30k cosine schedule, 50-step warmup and minimum LR ratio0.01.
Data/training helper sources are copied from the completed frozen core/LR
runner; declared model/trainer/diagnostic/reporting changes are overlaid.
All twenty cells train anew; historical quality outcomes are context only.

Five compiled12-step preflights cover every architecture at the higher
adapter multiplier4, including128/128/72 ragged batches, all-route wake-up,
actual optimizer rates, raw/EMA validation, diagnostics and CFG/FID/classifier
evaluation. All must complete and pass independent source/config/initialization/
stream/evaluator audits before any full run. Failure preserves evidence and
withholds later work. Unit and higher-rate optimizer ownership and no-op/gradient
behavior are also verified on CPU. There is no recovery checkpoint policy.

Use the same45k/5k CIFAR-10 research split seed123, validation noise seed2027,
sampling seed2028 and cached hash-pinned Inception/auxiliary class classifier.
Validate raw/EMA every5k; sample CFG1 at10k/20k. Final CFG1/2 share one EMA
state and initial noise, with DDIM50 and5k real/5k generated images. Primary
comparisons use final30k CFG2 FID; retain CFG1, auxiliary class agreement,
all per-seed gaps and convergence curves. No best-intermediate selection,
official-test evaluation or evaluator downloads. Fixed independent-condition
factor/scale magnitudes, live delta/base ratios, function updates and gradient
groups are diagnostics rather than quality proxies.

## Resources and evidence

At most two of our workers or GPU claim waiters. Estimated8–12 aggregate
GPU-hours and4–6 wall hours with two available GPUs. Require8GiB free for
the4GiB artifact bound and shared headroom. Retain compact source/configs,
runtime/seeds/splits, metrics/logs/diagnostics/reports/audits, compiler caches
expected below3GiB, twenty final100-image CFG1 grids and five preflight10-image
grids until explicitly removed. No model weights, periodic/final/recovery
checkpoints, generated datasets or changes to historical artifact retention.

CPU checks cover explicit row/column-operator outputs and every gradient,
gradcheck, identity scales, zero delta, paired static factors/RNG, removal of
unused control heads, condition-independent control transforms, BF16 dtype,
all-route wake-up, diagnostic trajectory invariance, previous dense/CP/core
compatibility, exact finite grid, actual optimizer groups and failure gates.
GPU audits pin frozen sources, observed streams, claims, exits, rates and
evaluation budgets. Concurrent throughput and whole-training allocation are
descriptive; dedicated speed benchmarking remains deferred. This is broadcast
conditioning, not causal retrieval, anchored token conditioning or LM training.


## Launch verification

Launched October14 at18:05 UTC. All five compiled preflights completed, passed
the controller audit and an independent frozen-runner replay by18:10 UTC.
Every adapter tensor received gradients by step5. All36 frozen source files,
25 configs,20 initial models and actual optimizer routes were verified;
118 workspace and25 frozen CPU checks passed. The twenty full30k quality
runs are queued with two active workers. Final quality/audit remain pending.
[Preflight evidence](dit-channel-factor-preflight.json) ·
[CPU validation](dit-channel-factor-cpu-validation.json) ·
[Launch manifest](dit-channel-factor-launch-manifest.json).
