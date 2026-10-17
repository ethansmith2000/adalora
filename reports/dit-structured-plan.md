# Conditional structured operators: first square-placement screen

The objective is useful conditional expressivity for the parameters, memory
and computation spent. This study changes the generated operator after the
earlier generator-compression studies. It keeps the broadcast CIFAR-10 DiT
task to measure the architecture change in an established training setup.
Synthetic retrieval and a task switch are deferred. No method is presumed
to improve quality, preserve trained variance, or run faster.

## Common placement and controls

```python
# h [B,T,256], c [B,256], projected [B,T,1024]
h = norm_mlp(tokens)                       # non-affine LayerNorm
h_mod = h * (1 + scale(c)) + shift(c)     # existing AdaLN
delta = operator(h, c)                    # [B,T,256]
pre_act = mlp_in(h_mod + delta)           # ONE shared projection GEMM
out = mlp_out(gelu(pre_act))
tokens = tokens + gate(c) * out
```

The operator consumes normalized h, separately from the AdaLN scale/shift
branch. Summing before the projection is equivalent in exact arithmetic to
adding a bias-free projection of delta, with the base bias counted once.
BF16 rounding can differ; diagnostic hooks log the projected branch and
rounding residual separately. Fixed FP32 function probes use projected MLP
preactivation units and include shared projection-weight motion.

All four square operators use this placement. The established rectangular
DPLR reference retains `mlp_in(h_mod) + rectangular_delta(h,c)` and its free
output factors. Its comparison with square operators also changes the
output subspace constraint and placement; it cannot isolate operator structure.

## Operators

Square low-rank uses the existing factorized affine U/V generator and static
factor offsets, now generating `[B,256,8]` and `[B,8,256]`. Gain is1/sqrt(8).
Only the update rank is bounded by8, not the full base-plus-update map.

The block operator generates16 independent contiguous16×16 matrices:

```python
a = coefficient_head(c)                       # [B,32]
blocks = (a @ decoder + offset).view(B,16,16,16)
delta = apply_blocks(h, blocks) / sqrt(16)     # [B,T,256]
```

The channel grouping is arbitrary, not a claim about known feature structure.
One block transform is tested; no second block layer/permutation is added.
The operator can have full square rank despite the restricted block pattern.

The Hadamard operator uses a learned conditional adaptation of Fastfood's
diagonal/transform/permutation chain. Original Fastfood uses stored random
diagonal matrices for kernel features; our model does not reproduce that
construction or its approximation guarantees.
[Original Fastfood paper](https://proceedings.mlr.press/v28/le13.pdf).

```python
a, b, c_out = coefficient_head(c).chunk(3, -1) # each [B,32]
Bdiag = signs * (1 + a @ D_B + bias_B)        # [B,256]
Gdiag = 1 + b @ D_G + bias_G                  # [B,256]
Sdiag = c_out @ D_S + bias_S                  # [B,256]
y = H(h * Bdiag[:,None])                     # normalized Walsh-Hadamard
y = y[..., permutation] * Gdiag[:,None]
delta = H(y) * Sdiag[:,None]
```

Signs and permutation are fixed, independently seeded per layer using
`training_seed +200000+adapter_index`, without consuming the global RNG.
Both transforms are orthonormal. Butterflies accumulate in FP32 and cast
back at each transform boundary during BF16 training. The normal forward
constructs no dense256×256 operator. Dense matrices exist only in correctness
oracles and untimed magnitude diagnostics. No custom Triton kernel is used.

The square diagonal control uses a wider bottleneck96:
`delta = h * (head(c) @ decoder + offset)`. This extra diagonal can combine
algebraically with the existing AdaLN scale; it is a generator/capacity
control. Its96-output coefficient heads are initially identical to the
Hadamard model's3×32 heads, with equal head/decoder matrix-multiply work.
Hadamard adds mixing operations and4096 offset parameters across the model.

## Initialization, budgets and optimization

Every operator starts at zero delta. Low-rank uses nonzero U and zero V;
diagonal/block decoders and offsets start zero. Hadamard B/G start as signs/one,
their conditional deviations start zero, and S starts zero. Every coefficient
head starts nonzero so decoder routes can wake. Once S opens, the B/G routes
can learn; zeroing both generator stages would stall learning. All parameter
tensors must receive nonzero gradients by step10 in the preflight audit.

Initial internal Hadamard mixing is orthogonal before the zero output diagonal.
Learned diagonals can change sign and magnitude; no positivity, clipping,
orthogonality constraint or variance-preservation guarantee is imposed.
Gauges remain. Gains1/sqrt(8),1/sqrt(16),1 for low-rank,block,diagonal/Hadamard
are declared conventions, not matched function-update scales or a muP rule.

| Design | Adapter parameters | Total parameters |
|---|---:|---:|
| Plain AdaLN |0|9,758,768|
| Rectangular low-rank r8/s32 |2,834,944|12,593,712|
| Square low-rank r8/s32 |1,212,928|10,971,696|
| Square diagonal s96 |396,032|10,154,800|
| Square block16/s32 |1,147,136|10,905,904|
| Square Hadamard,3×s32 |400,128|10,158,896|

Block and square low-rank have approximately matched adapter budgets; diagonal
and Hadamard also have approximately matched budgets. These are not exact
parameter or application-FLOP matches. Hadamard butterfly additions, scaling,
casts and memory traffic are excluded from reported matrix FLOPs and tracked
separately. Generated sizes alone do not establish peak training allocation.

All five adapter designs receive multipliers1/2 on fresh seeds128/129.
Plain AdaLN has no adapter group and runs once per seed:22 full30k quality
runs. Base LR8e-4, rank8, widths, initialization and gains stay fixed. Report
both rates and every endpoint, including finite poor quality. No automatic
selection, LR-grid expansion, resume, or best-intermediate checkpoint selection.
No static structured-operator controls in this first screen; a gain cannot
alone establish the contribution of conditioning versus parameterization.

## Training, evaluation and resources

Retain width256/depth8/heads4/patch4/MLP1024, batch128, AdamW betas0.9/0.999,
fused optimizer, weight decay0, clipping1, BF16/dynamic full-graph compilation,
EMA0.999, class dropout0.1,50-step warmup and30k cosine schedule with floor0.01.
Keep the45k/5k CIFAR-10 research split seed123, evaluation noise2027 and
sampling seed2028. Cached Inception and auxiliary classifier hashes remain
pinned. Validate raw/EMA every5k, generate CFG1 at10k/20k, and evaluate final
same-state/noise EMA CFG1/2 with DDIM50 and5k real/5k fake images.

Six compiled12-step preflights cover each architecture at the upper adapter
rate2 (plain rate1), including ragged128/128/72 batches, all-route gradients,
diagnostics, raw/EMA validation and guided evaluation. All must complete and
pass the source/config/initialization/stream/evaluator audit before quality.
Failure preserves evidence and withholds later work; compatibility preflights
do not establish long-run stability. Base/null parameters are paired across
all designs; every architecture starts identically across rates. Initial
Jacobians and many generator coordinates differ across architectures.

Primary output is final30k CFG2 FID; retain CFG1, class agreement/confusions,
both seeds, convergence, branch/function/gradient magnitudes, throughput and
whole-training allocation. Same-square-placement contrasts are separated from
cross-placement rectangular-reference gaps. Two seeds and reused holdout/noise
limit interpretation. Timings are concurrent observations; isolated speed
benchmarking remains deferred.

Use private supervisor and gpu-claim with at most two of our GPU workers or
claim waiters. Estimate10–14 aggregate GPU-hours,5–7 wall hours with two GPUs.
Check8GiB free before launch for the4GiB artifact bound and shared headroom.
Retain configs/seeds/splits, frozen sources/runtime, metrics/diagnostics/logs,
reports/audits/manifests, compiler caches expected below3GiB,22 final100-image
EMA CFG1 diagnostic grids and six preflight10-image grids until explicitly
removed. No model weights, periodic/final/recovery checkpoints, generated
datasets, evaluator downloads or official-test evaluation. No TT/FFT, shared
nonlinear trunks, retrieval or LM jobs are included.


## Additional interpretation and launch verification

The two Hadamard transforms and three condition-generated diagonals change
both channel mixing and condition dependence: their product can be cubic in
c, versus quadratic factor products for low-rank and affine operators for
the block/diagonal designs. Budget controls do not isolate those mechanisms.
This screen compares complete conditional architectures; a benefit cannot
be attributed to fixed mixing alone. Block16 and square rank8 generate the
same4096 values per sample and have equal leading application matmul work,
but kernels/layout and head costs differ.

Launched October16 at20:39 UTC. All six compiled preflights passed the
controller audit and independent frozen replay by20:46 UTC. Every adapter
tensor received gradients by step5. Ninety workspace and90 frozen CPU checks
passed. All28 configs,22 initial models/groups,43 frozen files and33 unchanged
reference sources were verified, with matching package versions. The22 full30k
quality runs are queued with two active workers. Final quality/audit pending.
[Preflight evidence](dit-structured-preflight.json) ·
[CPU validation](dit-structured-cpu-validation.json) ·
[Launch manifest](dit-structured-launch-manifest.json).
