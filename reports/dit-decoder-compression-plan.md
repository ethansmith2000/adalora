# CP factor-decoder compression: bounded first screen

The completed global-LR study supports broadcast class/timestep-conditioned
DPLR against plain and parameter-matched wider AdaLN. This follow-up targets
the generator's dominant decoder parameters. It is a parameter/quality trend
screen, with ordinary training-time and memory observations. It does not
launch the deferred dedicated speed benchmark or causal retrieval/LM work.

## Tensor design

The existing independent affine coefficient heads remain:

```python
a, b = Linear(256, 2*s)(c).chunk(2, dim=-1)  # each [B,s]
```

Dense U/V decoder bases contain `s*r*(d_in+d_out)` parameters per layer.
The CP alternative factorizes their three tensor modes:

```python
# U_channel: [d_in,p]; U_basis: [s,p]; U_rank: [p,r]
# V_basis: [s,p]; V_rank: [p,r]; V_channel: [p,d_out]
u_weights = (a @ U_basis)[..., None] * U_rank  # [B,p,r]
v_weights = (b @ V_basis)[..., None] * V_rank  # [B,p,r]
U = U_channel @ u_weights                    # [B,d_in,r]
V = v_weights.transpose(-1,-2) @ V_channel   # [B,r,d_out]
U = U + U_offset                            # static [d_in,r]
V = V + V_offset                            # static [r,d_out]
delta = (h @ U @ V) / sqrt(r)                # h: [B,T,d_in]
```

Factors are generated once per sample and reused across tokens, as before.
Neither `[d_in,s,r]` nor `[s,r,d_out]` dense decoder bases are reconstructed
during training. CP width `p`, coefficient bottleneck width `s`, and produced
transform rank `r` are distinct. The transform rank stays at most `r=8`.
The decoder parameter count becomes
`p*(d_in+d_out+2*s+2*r)`, plus unchanged offsets.

This changes the decoder function class and imposes fixed learned channel
subspaces plus the static factor offsets. It is not an exact compression of
an arbitrary trained dense decoder. No pretrained adapter is decomposed.

## Fixed designs and budgets

All models use width 256, depth 8, four heads, patch size 4, MLP hidden width
1024, transform rank 8, independent affine heads and learned factor offsets.

| Design | s | CP p | Total parameters | Adapter parameters | Decoder + offset parameters |
|---|---:|---:|---:|---:|---:|
| Dense reference | 32 | — | 12,593,712 | 2,834,944 | 2,703,360 |
| Dense wider control | 64 | — | 15,346,736 | 5,587,968 | 5,324,800 |
| CP small | 32 | 64 | 10,668,592 | 909,824 | 778,240 |
| CP larger | 32 | 128 | 11,364,912 | 1,606,144 | 1,474,560 |
| CP wider coefficients | 64 | 128 | 11,562,032 | 1,803,264 | 1,540,096 |

The last design spends some savings on larger s while retaining fewer adapter
parameters than the dense s32 reference. The dense s64 control distinguishes
widening the coefficient bottleneck from changing the decoder. These are not
equal-parameter, equal-FLOP or equivalent-function comparisons.

## Initialization and optimization

`U_channel` has independent normal entries with variance `1/d_in`, `U_basis`
variance 1, and `U_rank` variance `1/p`. Thus each effective dense U basis
entry has variance `1/d_in` in expectation, matching the reference's marginal
initial entry variance. Entries are correlated through shared factors.
`V_basis` and `V_rank` start nonzero; `V_channel` starts zero. Both factor
offsets remain zero. This preserves zero adapter delta while allowing the
output factor's gradients to wake first, followed by its companion factors.
AdaLN and residual-gate zero initialization remain unchanged.

CP introduces multiplicative scale gauges across its three factors. Marginal
variance matching does not establish variance preservation of the trained
operator, gauge control, equal functional updates, or a muP scaling rule.
This first screen keeps global LR and relative adapter LR fixed rather than
calibrating or selecting per-design rates. All gradient routes must wake
within the audited early steps. Fixed-condition factor magnitudes and
functional updates, live branch magnitudes, and clipped gradient groups are
retained to make optimization differences inspectable.

To preserve the exact reference head/global RNG trajectory at each s, model
initialization creates the old dense U/V temporarily on CPU and discards them
before moving the model to CUDA. CP draws use an isolated generator seeded
with `training_seed + 100000 + adapter_index`. Base/null embeddings match
across all designs; coefficient heads match within each s. Temporary CPU
initialization overhead is not counted as a training-memory saving.

## Training, evaluation and failure gates

Five designs × two new seeds (120/121) × 30,000 steps: ten complete quality
runs. Each has a fresh 30k cosine schedule, 50-step warmup, minimum LR ratio
0.01, global LR 8e-4, relative adapter LR 1, batch 128, fused AdamW
`(0.9,0.999)`, zero weight decay, clip norm 1, BF16, dynamic model compilation,
EMA 0.999 and class dropout 0.1. The LR came from the prior bounded search;
it is not established optimal for either dense or compressed designs.

Five 12-step compiled preflights cover every design, 128/128/72 ragged
training batches, gradient wake-up, diagnostics, raw/EMA validation, cached
FID/classifier evaluation and same-state CFG1/2. All preflights must complete
and pass independent evidence audits before any full quality run starts.
Failure preserves the evidence and withholds later runs; there is no implicit
resume or retry. Sources and configs are frozen before launch. The previous
DiT training/data helper snapshot is reused; only declared CP integration,
diagnostic/reporting changes and required source-provenance additions differ.

Use the same 45k/5k CIFAR-10 research split, split seed 123, validation noise
seed 2027, sampling seed 2028, DDIM50, 5k real and 5k generated images, and
hash-pinned Inception/auxiliary classifier. Raw/EMA denoising validation runs
every 5k. CFG1 generation at 10k/20k tracks convergence; final CFG1/2 share
one EMA state and sampling noise. Primary comparisons use final30k CFG2 FID;
all per-seed differences, CFG1 and class agreement are also reported.
No best-intermediate selection, automatic LR search or longer confirmation.

This is a two-seed screen on a reused research holdout/evaluation noise,
not independent test evidence or a claim of general superiority. Historical
quality cells are context only; both dense controls are newly trained.
Concurrent throughput and peak allocation are descriptive. CP can increase
decoder FLOPs despite reducing parameters. Broadcast factor memory still has
shapes `[B,d_in,r]` / `[B,r,d_out]`; this does not validate per-token memory.

## Resources and retained evidence

At most two of our GPU workers, including preflights and claim waiters.
Estimated 4–7 aggregate GPU-hours, about 2–3.5 wall hours with two available
GPUs. Check for active adalora jobs and at least 8 GiB free before launch.
The workflow has a 4 GiB storage bound: compact configs, source/runtime/seeds,
split indices, metrics, logs, diagnostics, reports/audits; compiler caches
expected below 3 GiB; ten final100-image CFG1 grids and five preflight10-image
grids. Retain these until explicitly removed. No periodic/recovery/final model
weights, generated datasets, official-test evaluation or evaluator downloads.
The fixed cached evaluators and historical evidence remain untouched.

CPU checks cover explicit dense-expansion outputs and every gradient, gradcheck,
absence of saved dense basis tensors, unchanged default dense training,
paired heads/global RNG, BF16 factor dtype, all-route gradient wake-up,
diagnostic trajectory invariance, optimizer ownership, config roundtrips,
finite grid/budgets and preflight failure gates. GPU audits additionally pin
source/evaluator hashes, observed data/noise/dropout streams and hardware claims.
