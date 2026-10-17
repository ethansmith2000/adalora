# ViT capacity control: initializer and learning-rate study

The wider static MLP underperformed plain ViT at 30 epochs, with lower training
accuracy as well as lower validation accuracy. Its initial MLP output RMS was
1.379× the narrow model under the same fixed `std=0.02` initialization. This
study checks that control before stronger conclusions about DPLR capacity.
[Diagnostic](vit-capacity-diagnostic.md) · [Candidate rationale](vit-capacity-initializer-plan.md).

Each of plain ViT, wider plain ViT and DPLR s24 gets the same **12 candidates**:
four initialization policies × three global learning rates. Each candidate
runs for the complete **30-epoch / 10,560-step schedule** on paired tuning
seeds 104/105. Short prefixes do not select recipes: the earlier ten-epoch
screen showed that the wider model learns more slowly. The research split is
the same 45,000 training / 5,000 validation images, split seed 123.

| Method | MLP hidden width | Parameters |
|---|---:|---:|
| Plain |768|2,690,122|
| Wider plain |1483|4,341,772|
| DPLR s24 |768|4,342,378|

All use width 192, depth 6, heads 3, patch 4, batch 128, AdamW(.9, .999), fused
optimizer, compiled BF16, warmup 5%, cosine LR floor 0.01, clip 1, weight decay 0.05
except biases/norms/factor offsets, and label smoothing 0.1. DPLR transform
rank 8, projector width 24, gain 1/sqrt(8), factor offsets enabled, normalized-token
self-conditioning and parallel MLP input delta remain unchanged. Relative
affine and low-rank LR multipliers stay 1. Global LRs are
**2.5e-4, 5e-4, 1e-3**; this scales base and conditional groups together.

The initializer intervention touches only the two base MLP weight matrices.
Attention, embeddings, norms, biases and classifier follow their historical
initialization. Conditional scale/shift, generated V bases and factor offsets
remain zero; U bases and coefficient projectors remain nonzero. The new helper
rescales the existing random draws after base initialization and before
adapter attachment, consuming no extra RNG and adding no parameters or
runtime operations. The legacy policy is exactly unchanged.

| Candidate | fc1 rule | fc2 rule |
|---|---|---|
| `legacy` |std 0.02|std 0.02|
| `fanin_gelu` |gain 1 / sqrt(input width)|explicit output gain / sqrt(hidden width)|
| `xavier_gelu` |gain 1 × sqrt(2/(input+hidden))|explicit output gain × sqrt(2/(hidden+output))|
| `mixed_gelu` |explicit sqrt(2) / sqrt(input width)|explicit output gain × sqrt(2/(hidden+output))|

For the three fan-aware policies, derive the output gain using deterministic
96-point Gaussian quadrature of the actual GELU activation. At reference
hidden width `4*dim`, the expected output second moment matches the legacy
narrow MLP; hold that gain fixed when changing hidden width. This prevents
default fan-aware scales from introducing a large unrelated overall residual
gain. The first projection's preactivation scale still differs between
families, deliberately. Fan-in keeps its width invariance; Xavier/mixed
depend on fan-out too and are measured rather than assumed invariant.
The mixed recipe explicitly tests a ReLU-style sqrt(2) input gain before GELU,
with the output gain calibrated using the actual GELU second moment. It is an
approximation to test, not a claim about GELU's optimal gain. Using gain1 in
this mix would duplicate the fan-in recipe at the narrow reference after
output anchoring; the distinct input gain avoids that wasted tuning budget.
These are normal-family std rules applied to existing truncated-normal draws
whose bounds are ±100 std; they retain that truncation rather than calling
new RNG-based initializers.

Add two **diagnostic-only** wider runs at global LR 5e-4 with
`fc1_std=.02; fc2_std=.02*sqrt(768/1483)`. This `reference_fanin` correction
changes only the width-induced initial output gain and is exactly the legacy
rule at the narrow reference width. It **cannot enter recipe selection**;
each method still has precisely 24 eligible tuning runs. Tuning stage total
is 72 eligible plus 2 diagnostic runs.

Two new compiled GPU preflights (12 steps, 328 training images) exercise wider
Xavier and DPLR fan-in initialization. Both must pass complete evidence audits
before tuning. Select one initializer/LR per method by mean **final** validation
accuracy over seeds 104/105; tie-break on mean final validation loss, closest
unit LR, lower LR and declared initializer order. Best intermediate-epoch
accuracy is not used. All 74 records must be complete and audited before
selection. Freeze selection and its hash before confirmation; no automatic
grid expansion or selection based on confirmation results.

Confirmation uses four fresh paired seeds 106/107/108/109. Run each method's
selected recipe and its legacy-LR 5e-4 anchor on those same seeds. If selected
already equals the anchor, use that record for both roles without duplicating
training. This produces 12–24 confirmation runs, determined only by frozen
tuning selections. Record paired selected-minus-anchor and cross-method
differences, means/sample SD, training/validation curves and costs. The static
low-rank baseline and s48 are not retuned here; this cohort specifically checks
the wider control with equal budgets for plain, wider and DPLR s24.

Diagnostics include a CPU-only untrained probe at `[2,65,192]` for initial
preactivation RMS, post-GELU second moment and MLP output RMS in both widths;
actual fixed-token **complete MLP** function updates during training in addition
to the existing low-rank delta updates; live branch magnitudes; parameter/
gradient RMS; logged gradient clipping norms; optimizer membership/rates/decay;
initialization and observed image/label stream hashes. Probe frames are
0/1/2/3/5/10/50/100/250/500/800/1200/2000/3520/7040/10560.

Audit every worker's frozen sources, runtime, configs, optimizer groups and
schedule, seed/split, parameter and image counts, zero initialization/gradient
wakeup, finite function measurements, validation denominators, successful exit
and GPU claim identity. Plain and DPLR share identical base initializations
within each initializer/seed. Width changes constructor RNG consumption, so
the wider base is not identical to the narrow base. Observed augmentation
streams must match within each seed across all candidates. No old GPU results
are reused as new workers.

Four new confirmation seeds address seed variation, not research-holdout
overfitting: tuning and confirmation reuse the same holdout. Gaussian second-
moment anchoring is an initialization model, not exact data-distribution or
training invariance. LR selection need not preserve function-update magnitude;
diagnostics measure it without enforcing the earlier low-rank matching gate.
SD is not a confidence interval. The wider control matches DPLR within 606
parameters, with similar leading matmul FLOPs; this does not equate architectures.

Resource/retention declaration before launch: at most **two of our GPUs**
across all stages; **10–16 aggregate GPU-hours**, approximately 5–8 wall hours
with two available GPUs. Maximum 100 workers: 2 preflights +74 tuning/diagnostic
+24 confirmation. Retain at most 4 GiB with 8 GiB minimum free-space headroom;
shared storage has approximately 309 GiB free. Preserve compact sources/configs/
runtime/seeds/splits/initializer/optimizer/metrics/diagnostics/logs/audits/reports
and declared compiler caches (expected ≤3 GiB) until explicitly removed. No
model weights, periodic/recovery/final checkpoints, sample grids, official-test
evaluation or evaluator weights. Existing evidence and retention are unchanged.

[Implementation](../vit_initializer_study.py) ·
[Initialization helper](../vit_initialization.py).

CPU verification: 72 passed, one CUDA-specific exact-layer check skipped.
The tests verify four distinct initial MLP functions for each method, exact
legacy equivalence, RNG/zero-init rules, std/gain calculations, complete grid
selection and confirmation anchors. The corrected frozen workflow is
`runs/vit-initializers-20261012T070919Z`. Preparation remained CPU-only.

The preliminary workflow `runs/vit-initializers-20261012T070149Z` was stopped
before any tuning after its CPU probe revealed that the original gain1 mixed
recipe duplicated fan-in at narrow width. Both preliminary preflight workers
finished and passed their audits; their source/protocol/logs/results are kept
with a correction record. The corrected mixed profile has explicit sqrt(2)
input gain and an output gain derived using actual GELU moments. New GPU
preflights will verify the corrected source snapshot; old results are not reused.
No tuning budget was spent on duplicate candidates. Preliminary artifacts are
retained under their original bound; the corrected workflow has its declared
4 GiB bound.

[Test log](vit-initializer-tests.log) · [Initialization probe](vit-initializer-probe.json) · [Live plan](vit-initializer-study-plan.json).

Both corrected compiled GPU preflights passed the frozen evidence audit.
The 74 full-duration tuning/diagnostic workers are scheduled, with two
currently active; fresh confirmation is gated on the completed audited
selection grid. [Preflight evidence](vit-initializer-preflight.json).

## Completion record

Completed all 74 tuning/diagnostic and 24 confirmation workers; all seven final audit checks passed. Plain/wider/DPLR selected confirmation means are 79.440%/78.635%/79.350%. See [completed results](vit-initializers.md). Private supervisors are closed and our workers are inactive; no weights or test evaluation.
