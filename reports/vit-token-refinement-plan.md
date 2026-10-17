# Bounded ViT calibration refinement

The original two GPU preflights and all28 calibration pilots completed and
passed their individual evidence audits. The frozen calibration gate failed:
s12 has no tested rate keeping every update ratio in0.5–2, and the static
rank8+diagonal control also lacks a valid rate. The original failed protocol,
worker outputs and diagnostic summary stay intact. No quality run was started.

At s12×1, the smallest ratio is0.471; at×4, the largest is2.504.
For the static control,×4 is too low (ratios0.230–0.340), and×16 crosses the
upper limit (maximum2.241). s48×1 and detached s24×1 have valid ratios.

Run exactly eight new500-step pilots, retaining the original seeds96/97,
3520-step schedule horizon, training/model/diagnostic sources, configs except
adapter rate/output path, split, normalization, initialization and observed
augmentation streams:

| Method | Additional adapter-rate multipliers | Runs |
|---|---|---:|
| s12 |2|2|
| s48 |0.5|2|
| Static rank8 + diagonal |8 and12|4|

Rates are frozen from update bracketing, not accuracy. Merge the audited28
original pilots with these eight; select the closest geometric update magnitude
among rates whose six ratios (two seeds×three probes) all remain in0.5–2.
Bounds and probe steps100/250/500 are unchanged. Retain all candidate diagnostics
even if the refined gate fails. No unbounded expansion or relaxed guard.

If the refined gate passes, launch the originally planned16 ten-epoch quality
runs: eight families on fresh seeds98/99, with selected rates frozen before
launch. Offsets remain enabled in all dynamic methods. Hyper LR changes the
scale/shift heads as well as low-rank parameters; this calibration does not
match their updates or establish globally optimal learning rates. Unit-rate
pilot accuracies are preliminary and are not used for recipe selection.

No GPU preflight is repeated: reuse the original preflights only after full
source/runtime/config/initialization/evaluation audit. A separate frozen runner
and protocol records the old failure and protects all referenced evidence.

Resource declaration before launch: at most two GPU workers across refinement
and quality, estimated2–4 aggregate GPU-hours (about1–2 wall hours when two GPUs
are available),4GiB retained-artifact bound and8GiB minimum free-space headroom.
Current shared free space is approximately312GiB. Preserve compact sources,
configs, runtime/seed/split records, metrics, diagnostics, logs and audits/reports;
compiler caches are expected≤3GiB. Retain until explicitly removed. No model
weights, periodic/recovery checkpoints, grids, official-test evaluation or new
evaluator weights. Existing artifacts and retention remain unchanged.

[Original gate diagnostic](vit-token-coarse-calibration.md) ·
[Original exact ratios](vit-token-coarse-calibration.json).

Launched in `runs/vit-token-refinement-20261011T185940Z`. Both original
preflights and all28 original pilots were re-audited before launch; training
source hashes and the old failed protocol/summary hashes are pinned. Nineteen
focused CPU checks passed. Eight new calibration pilots are running with two
workers. The original16 quality runs remain gated, with no model weights saved.
[Live plan](vit-token-refinement-plan.json).

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

[Audited calibration selection](vit-token-refinement-calibration.json).

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

[Completed comparison and limitations](vit-token-refinement.md).
