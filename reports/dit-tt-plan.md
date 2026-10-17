# Conditional tensor-train operator study — 2026-10-17

Implement the generated operator as a TT matrix, rather than applying TT only
to the generator's static weight. Keep the established class/timestep broadcast
CIFAR-10 diffusion task, width256/depth8, 64 image tokens and the existing
non-affine pre-norm AdaLN/gates. The square branch is:

```python
norm_h = norm_mlp(tokens)                   # [B, T, 256]
h_mod = norm_h * (1 + scale) + shift
latent = Linear(256, 32)(condition)         # [B, 32], shared across core heads
cores = [latent @ basis_i + offset_i for i in range(4)]
delta = tt_apply(norm_h, cores)             # [B, T, 256]
pre_act = mlp_in(h_mod + delta)             # one shared base projection
```

The channel axis is tensorized as `[4,4,4,4]`, with each core's paired input and
output modes `[4,4]`. For bond rank `k`, the per-image cores are:

```python
[B, 1, 4, 4, k]
[B, k, 4, 4, k]
[B, k, 4, 4, k]
[B, k, 4, 4, 1]
```

A TT bond rank is distinct from ordinary matrix rank: even bond1 can represent
a full-rank Kronecker product. The mode order imposes a channel structure;
there is no mode-order/permutation search in this study.

| Design | Adapter parameters | Total parameters |
|---|---:|---:|
| Square diagonal, latent96 | 396,032 | 10,154,800 |
| Square TT, bond6/latent32 | 420,608 | 10,179,376 |
| Square block16/latent32 | 1,147,136 | 10,905,904 |
| Square TT, bond11/latent32 | 1,180,928 | 10,939,696 |
| Square low-rank, matrix-rank8/latent32 per factor | 1,212,928 | 10,971,696 |
| Rectangular low-rank reference | 2,834,944 | 12,593,712 |

TT6 generates1,344 core values per image; TT11 generates4,224. Their single
condition-to-latent head has8,224 parameters per block. With decoder biases,
each generated core value contributes33 parameters. Across eight blocks the
TT head totals65,792 parameters; core decoders/offsets total354,816 (bond6)
or1,115,136 (bond11). This is an approximate parameter match, not equal compute,
activation memory, Jacobians, or optimization budgets.

`tt_apply` contracts one mode at a time. Its intermediate layout is
`[B,T,output_prefix,input_suffix,bond]`, bounded here by`B*T*256*k` values.
At batch128/tokens64 the largest state alone is24MiB (bond6) or44MiB (bond11)
in BF16. This excludes copies, autograd-saved states, gradients, base model,
optimizer/EMA, and compiler allocations. The generated cores are much smaller
but do not determine whole-training peak memory. Dense matrices are built only
in untimed diagnostics and test oracles. The reference is PyTorch plus compile;
no hand-written kernel or isolated speed benchmark is introduced.

The four TT cores are each affine in one shared latent, so the net conditional
operator is degree up to4 in the condition. Diagonal/block are affine and the
low-rank product is quadratic. Expressivity differences are not attributable
solely to channel mixing. Square output directions remain coupled to the base
MLP weight; the rectangular reference has different output-subspace freedom.

Initialization makes every core decoder zero and the last core offset zero.
The first three offsets are independent Gaussian with std
`1/sqrt(input_mode * left_bond)`. The head is nonzero affine. Only one core is
zero, so the end-core decoder/offset can open and other cores/head can receive
gradients after it opens; all tensor names must have observed nonzero gradients
by step10 before quality starts. This fan-in convention would preserve expected
input variance if all cores independently had that variance; the deliberate
zero end core changes that at initialization. There is no guarantee of trained
variance preservation, no normalization/rebalancing, and TT gauge freedoms remain.
Gain is1 for TT/diagonal,1/sqrt16 for block and1/sqrt8 for low-rank.

Use seeds130/131, adapter LR multipliers1/2, global LR8e-4 and the unchanged
30,000-step cosine schedule/warmup50, fused AdamW(.9,.999), zero weight decay,
BF16 dynamic compilation, batch128, EMA.999 and class drop.1. Six compiled
12-step preflights cover all architectures, including ragged final batches,
training/validation/generation and actual gradient wake-up. Quality is gated by
independent replay of preflight source/config/runtime/stream/group/evaluator
audits. Then24 full quality runs, every endpoint retained, no grid expansion or
automatic recipe selection. CFG1 at10k/20k and final CFG1/2 use the same EMA/noise,
5,000 generated/real holdout images,50 DDIM steps, existing hash-pinned evaluators.
The official test is unused; no evaluator downloads. Two training seeds on a
reused research holdout do not establish independent final-quality evidence.

Report both seeds, mean/sample SD (not confidence intervals), final CFG2 FID,
CFG1 FID, class agreement, fixed-function/update/gain diagnostics, optimizer group
ownership/rates, ordinary concurrent training throughput and whole-training peak
allocation. Keep same-square contrasts separate from the rectangular reference.
No fresh plain-AdaLN/Hadamard runs; historical results are contextual only.

At most two adalora GPU workers or claim waiters, using GPU claims and private
supervisors. Minimum8GiB free;4GiB retained-artifact bound, estimated12–20 aggregate
GPU hours/6–10 wall hours with two GPUs (TT execution remains unmeasured). No
periodic/final/recovery model checkpoints. Retain compact configs, sources,
provenance, metrics, diagnostics, logs, reports, audits and manifests. Declared
artifacts:24 final100-image EMA CFG1 grids, six10-image preflight grids, compiler
caches expected below3GiB. Retain until explicitly removed; prior runs/evaluators
unchanged. Check capacity before launch. No rectangular TT, FFT, shared nonlinear
trunk, toy retrieval, token-self-conditioning or new LM launch in this finite study.

The TT-matrix construction is from [Novikov et al. (2015)](https://papers.nips.cc/paper/2015/file/6855456e2fe46a9d49d3d3af4f57443d-Paper.pdf).
Their operators are static; the condition-generated cores here are our adaptation.
The reference workflow is the completed audited structured screen
`runs/dit-structured-20261016T203614Z`; preserve its frozen training/evaluation
helpers and copy only declared TT/integration overlays.

## Verified launch status



All six TT-study CUDA preflights passed both the controller audit and an
independent frozen-runner replay. Every adapter tensor received a nonzero
gradient by step5, including all80 tensors in each TT variant. Source/config,
paired streams/initializations, actual optimizer groups/rates, claim identity
and evaluator budgets verified. Short-preflight whole-run peak allocations
were1.91GiB (TT6) and2.37GiB (TT11); these are not sustained quality-training
resource results. The24 full30k quality runs are now open under the two-worker
cap. Final quality and audit remain pending; no model weights/checkpoints.
[TT preflight evidence](dit-tt-preflight.json) ·
[TT startup status](dit-tt-startup.json).
