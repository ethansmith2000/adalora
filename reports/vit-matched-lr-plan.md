# ViT matched Xavier and global-LR diagnostic

The initializer study left tuned plain ViT and DPLR effectively tied. Its two
tuning seeds showed a learning-rate interaction under Xavier initialization:
plain/DPLR means 79.13%/79.64% at `2.5e-4`, versus 79.62%/78.86% at `5e-4`.
This follow-up completes those two fixed LR comparisons on seeds 106–109.

| Method | Global LR | Existing cells reused | New runs |
|---|---:|---:|---:|
| Plain |2.5e-4|0|4|
| Plain |5e-4|4|0|
| DPLR r8/s24 |2.5e-4|4|0|
| DPLR r8/s24 |5e-4|0|4|

Reuse only the eight matching completed cells in
`runs/vit-initializers-20261012T070919Z`. Its 24 quality records are reaudited
read-only; the original source, protocol, report, audit, selection, stage
records and worker evidence stay unchanged. New workers run the byte-identical
training source snapshot. Before launch, require matching Python, PyTorch,
CUDA wheel, package versions and trainer/model source hashes. Successful
compiled preflights and full-budget cases already exercise these methods and
LRs; no extra GPU preflights or tuning workers are introduced.

All 16 combined cells use the same Xavier profile, width192/depth6/heads3,
MLP hidden768, patch4, batch128, 30 epochs/10,560 steps, split seed123 and
45,000 training/5,000 validation images. Keep AdamW(.9,.999), fused optimizer,
compiled BF16, warmup5%, cosine floor.01, clip1, matrix weight decay.05 and
label smoothing.1. DPLR keeps normalized-token self-conditioning, parallel
MLP input delta, r8/s24, gain1/sqrt(8), factor offsets enabled and relative
affine/low-rank LR multipliers1. Global LR scales all groups together.

Freeze the exact eight missing cases before launch. There is no selection,
early stopping, extra grid cell or retuning based on follow-up endpoints.
Primary results are paired final-validation **DPLR minus plain** at each LR
and the interaction:

```python
gap_low[seed] = acc_dplr_lr_low[seed] - acc_plain_lr_low[seed]
gap_high[seed] = acc_dplr_lr_high[seed] - acc_plain_lr_high[seed]
interaction[seed] = gap_high[seed] - gap_low[seed]
```

Report every seed, mean and sample SD in percentage points. Retain complete
training/validation curves, fixed normalized-token complete-MLP and low-rank
interval updates, live branch magnitude, gradient/parameter RMS, clipping,
optimizer groups, source/runtime/split and observed image/label stream hashes.
Audits require identical base initialization across methods/LRs for each seed,
identical DPLR adapter initialization across LRs, identical probe inputs and
observed training streams across cohorts. All 16 cells must finish at the
same horizon with 5,000-example validation denominators.

These confirmation seeds and the research holdout were already inspected
before proposing this follow-up. It is an exploratory interaction diagnostic,
not independent confirmation. Same nominal LR does not mean matched function
updates or the optimal LR for each architecture. Four seeds, 30 epochs and a
single split limit the claim; SD is not a confidence interval. Static low-rank,
diagonal-only, wider MLP and regularization studies are outside this launch.

Resource and retention declaration before launch: **at most two of our GPUs**,
including queued workers, under private supervisor and cooperative GPU claims.
Expected **1–2 aggregate GPU-hours**, roughly **30–60 minutes** with two GPUs
available. Require 8 GiB free headroom and retain at most 4 GiB for this workflow.
Retain compact configs, seeds, source/runtime provenance, metrics, logs,
diagnostics, reports and reference hashes; compiler caches have the stated
compatibility/performance purpose and expected size at most 3 GiB. Keep these
until explicitly removed. **No weights, periodic/recovery/final checkpoints,
official-test evaluation or evaluator downloads.** Original retention is unchanged.

CPU verification: 58 checks passed across the new matrix/pairing tests and
existing initializer, split-LR and common executor tests. They cover the exact
missing complement, independent interaction arithmetic, rejection of duplicate
or incompatible reuse, and withholding GPU work when reference verification fails.

[Implementation](../vit_matched_lr_study.py) ·
[CPU checks](../test_vit_matched_lr_study.py) ·
[Live plan](vit-matched-lr-plan.json) ·
[Previous completed study](vit-initializers.md).

## Completion record

All eight new runs completed October 12 at 17:13 UTC and all seven final audits passed. Plain/DPLR means are 79.465%/79.350% at 2.5e-4 and 79.440%/79.525% at 5e-4. Method gaps remain small; the large tuning-seed LR interaction did not reproduce. Private supervisors are closed and our workers inactive. [Completed results](vit-matched-lr.md).
