# External conditioning: equal global-LR tuning

The earlier 60k-step class/timestep-conditioned DiT comparison favored DPLR
over a parameter-matched wider AdaLN model in both seeds at CFG1 and CFG2.
Mean guided FID was 30.08 versus 32.33, with two paired guided differences
−2.49/−2.00. The later factor-offset comparison was modestly positive, and
offsets are the user's forward research default. Neither comparison gives
all architectures an equal global-LR search at the current model size.
The ViT baseline's improvement after tuning makes this control worthwhile.

Use the **original frozen DiT training/model/data/diagnostic snapshot** from
`runs/dit-factor-offset-20261010T193609Z/runner`. Keep every training source
byte unchanged, including its old ViT/data helper dependencies. Recent token
conditioning changes remain outside this workflow. Only the new orchestrator,
its tests, the shared supervisor helper's optional phase metadata, and the
reporting summarizer overlay the runner. No prior quality cell is reused.
The original factor-offset reporting failure and successful reporting recovery
remain preserved; its final audit and trained records supply source provenance.

| Method | MLP hidden width | Parameters | Factor offsets |
|---|---:|---:|---|
| Plain AdaLN-Zero |1024|9,758,768|No adapter|
| Wider AdaLN-Zero |1715|12,594,632|No adapter|
| DPLR r8/s32 |1024|12,593,712|Enabled|

Wider matches offset-enabled DPLR within **920 parameters (0.0074%)**.
This replaces the old hidden 1696 match for the no-offset parameter budget.
All models already initialize base Linear weights with Xavier. No new
initializer, nonlinear/shared projector, input normalization, gradient-routing
intervention, cross-attention or rank/width search is introduced.

Each method receives **three global LRs: 2e-4, 4e-4, 8e-4**, on two new
tuning seeds 114/115. All **18 tuning runs train for 30,000 steps**, using a
fresh 30k cosine schedule. Select one rate per method by **mean final CFG2
EMA FID**, with ties broken by final EMA noise MSE, closest unit multiplier,
then lower multiplier. Best intermediate states and classifier agreement do
not select rates. Require the exact completed audited grid, freeze selection
and its hash, then run **12 confirmation cases on fresh seeds 116–119**.
The training horizon and schedule are identical in tuning and confirmation.
There are no duplicate/reused confirmation cells or automatic grid expansions.

The condition is a learned class-plus-timestep vector normalized by non-affine
LayerNorm, `[B,256]`, broadcast across the 64 image tokens. All models retain
the same AdaLN scale/shift/residual gates and non-affine pre-norms. DPLR uses
independent affine coefficient projectors, r8/s32, normalized value input,
parallel MLP-input delta, factor offsets and gain 1/sqrt(8). Relative adapter
LR multiplier remains 1, so changing global LR scales all optimizer groups.

Keep width 256/depth 8/heads 4/patch 4, batch 128, fused AdamW(.9,.999), weight
decay 0, clip 1, warmup 50, cosine floor .01, compiled BF16/dynamic compilation,
10% class dropout, in-memory EMA .999, 1000 diffusion timesteps and squared
cosine betas. Use the full 45k/5k CIFAR10 research split, split seed 123,
evaluation seed 2027, sampling seed 2028 and deterministic DDIM 50.

Track raw/EMA denoising validation every 5k; record CFG1 FID/class agreement
at 10k/20k. Final CFG1 and CFG2 evaluate the same EMA state and initial noise.
Use **5,000 real and generated images**, balanced requests and the existing
frozen Inception weights/auxiliary classifier. Report both scales and all
per-seed method-minus-control differences. Prior 60k or 30k endpoints remain
context, rather than schedule/seed-controlled new outcomes.

Diagnostics retain zero initial gates/deltas, all adapter-gradient wakeup,
offset/base/coefficient magnitudes and fixed function updates, raw/EMA MSE,
wrong-class sensitivity, observed input/noise/timestep/effective-label traces,
optimizer rates, split/runtime/source hashes and claimed hardware identity.
Plain and DPLR share narrow base initialization. Wider has its own shape-
dependent initialization, paired across rates. All methods/rates share data,
noise and label-dropout streams within each seed. Each worker's canonical
config is audited rather than ignoring unexpected method-specific fields.

Two 12-step compiled real-data preflights (wider and offset-enabled DPLR)
exercise ragged 128/128/72 batches, gradient wakeup, full 5k validation and
20-image short-DDIM CFG1/2 evaluation. Both audits must pass before tuning;
the full tuning audit must pass before frozen selection and confirmation.

Four fresh seeds improve the assessment of seed variation, but tuning and
confirmation reuse the research holdout/evaluation noise. SD is not a confidence
interval. Finite-sample 5k held-out-train FID is not published 50k test FID;
classifier agreement is an auxiliary proxy. Equal tuning budget does not
establish globally optimal rates or matched function updates. Parameter counts
do not equate FLOPs/activation shapes. The condition is one external vector;
sequence-valued conditioning and cross-attention remain a separate question.

Resource/retention declaration before launch: **at most two of our GPUs**
across all stages, including queued workers, under private supervisor and
cooperative claims. **32 workers: 2 preflights +18 tuning +12 confirmation**.
Estimate **10–15 aggregate GPU-hours, about 5–8 wall hours** with two available
GPUs. Require 8 GiB shared free headroom; retain at most 4 GiB for this workflow.
Retain compact source/config/seed/runtime/split/metric/diagnostic/log/audit/
selection/report records; compiler caches have a compatibility/performance
purpose and expected size at most 3 GiB. Retain thirty final 100-image CFG1
diagnostic grids and two preflight 10-image grids. These remain until explicitly
removed; prior source/evaluation inputs and their retention stay unchanged.
**No saved training weights, periodic/recovery/final checkpoints, new evaluator
weights/downloads or official-test evaluation.** Required existing classifier
and Inception weights are reused and hash-pinned.

[Implementation](../dit_external_lr_study.py) ·
[Protocol tests](../test_dit_external_lr_study.py) ·
[CPU checks](dit-external-lr-tests.log) ·
[Live plan](dit-external-lr-plan.json).

CPU validation: **54 regression/protocol checks passed**, plus **8 protocol
checks inside the frozen snapshot**. All 20 prepared configs round-trip through
the frozen trainer; all parameter counts and zero-output class/null-class
functions are verified without a CUDA context. GPU preflights remain pending.
Raw workflow: `runs/dit-external-lr-20261013T070213Z`.

Both compiled GPU preflights passed their full evidence audits. Wider and
offset-enabled DPLR completed the ragged data budget and real CFG1/2 evaluator
paths; the 18 full-horizon tuning runs are now underway. Confirmation remains
gated on the complete audited grid and frozen selection.
