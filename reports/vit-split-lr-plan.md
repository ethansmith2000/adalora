# ViT: independent affine and low-rank learning rates

The ten-epoch comparison finished with DPLR s24 at 69.82% versus diagonal
conditioning at 69.04%. s48 reached 70.05%, with 26.7% more parameters than s24.
The static rank 8+diagonal control at a shared 12× adapter rate developed
combined/base activation RMS of 11.69–19.68 across blocks/seeds. The shared
multiplier changed the scale/shift heads as well as low-rank factors.
[Completed evidence](vit-token-refinement.md).

This study isolates that rate coupling and extends training to 30 epochs.
The model remains width 192, depth 6, heads 3, patch 4, transform rank 8. The
condition remains each normalized token; the low-rank delta receives that
normalized input in parallel with the modulated base MLP input projection.
Ordinary self-attention and static affine LayerNorm are unchanged. No AdaLN
residual gate is added. Dynamic U/V factors retain zero-initialized offsets.

The optimizer assigns every parameter exactly once:

```python
base_lr = 5e-4
affine_lr = base_lr * affine_lr_multiplier
lowrank_lr = base_lr * lowrank_lr_multiplier
# affine: condition -> scale/shift heads
# lowrank: coefficient projector, learned factor bases, U/V offsets;
#          or static U/V for the static control
# base: original model parameters, including learned LayerNorm affine
```

Both optional multipliers independently fall back to the legacy shared
`hyper_lr_multiplier`. If both are unset, optimizer group topology and updates
remain unchanged. Bias/norm/factor-offset decay remains zero; matrix weights
retain decay 0.05. Split-group names, parameter membership, counts, initial LRs
and weight decay are recorded, and every logged group LR is audited against
the schedule.

| Quality family | Affine LR × | Low-rank LR × | Purpose |
|---|---:|---:|---|
| Plain | — | — | Original model |
| Diagonal |1|—| Conditioned scale/shift control |
| Static capacity |—|—| Wider MLP within 606 parameters of s24 |
| DPLR s24 |1|1| Reference/provisional default |
| DPLR s48 |1|Calibrated| Wider projector trend |
| Static unit |1|1| Untuned static rank 8+diagonal control |
| Static split |1|Calibrated| Update-matched low rank with fixed affine rate |
| Static joint12 |12|12| Frozen diagnostic for the previous rate coupling |

Two new compiled GPU preflights exercise s48 and static split on 328 training
examples, 12 steps each. The changed optimizer and source snapshot require
fresh preflights; no old GPU records are reused.

The calibration is exactly 20 fresh 1200-step pilots on paired seeds 100/101:

| Family | Low-rank multipliers | Pilots |
|---|---|---:|
| DPLR s24 reference |1|2|
| DPLR s48 |0.25,0.5,1|6|
| Static split |4,8,12,16|8|
| Static unit |1|2|
| Static joint12 |12|2|

Each pilot is a prefix of the 10560-step/30-epoch schedule, including 528 warmup
steps. Probe low-rank function updates at 500/800/1200 on fixed FP32 normalized
tokens, average block update RMS, and divide by the paired s24 reference.
Every selected ratio across both seeds/three probes must be in 0.5–2. Among
valid candidates, select the minimum mean absolute log ratio, tie-breaking
toward the lower rate. Accuracy does not enter selection. The static unit and
joint12 controls keep their declared rates, even if they miss these bounds;
they are diagnostic controls, not update-matched methods. Keep all candidate
ratios, including failures. A failed gate withholds quality, without relaxing
bounds or automatically expanding the grid.

If both preflights and calibration audits pass, freeze rates and run exactly
16 fresh quality runs: eight families × seeds 102/103, 30 epochs each. These
are fresh initializations with a new schedule; they do not resume the prior
ten-epoch models or isolate duration alone. The 45k/5k research split remains
seed 123, with batch 128, AdamW(.9, .999), fused optimizer, compiled BF16,
LR 5e-4, warmup 5%, cosine floor 0.01, gradient clip 1 and label smoothing 0.1.
All static-rate variants share their base and adapter initialization within
each seed. Common-shape models share base initialization and observed training
streams; the wider static MLP has different shapes/initialization. The audit
checks source/runtime/split/config, step/image counts, zero start and gradient
wakeup, optimizer membership and every group schedule, validation denominators,
GPU claim identity, fixed rate selection and retained artifacts.

Track final and per-epoch validation, low-rank update transfer, live branch
magnitudes and parameter/gradient RMS. Near parameter matching does not tune
the wider static MLP's learning rate. Larger s has a larger budget. Scalar
low-rank update matching does not match affine or whole-network updates or
guarantee equal gains. Two seeds and a reused research holdout are exploratory;
sample SD is not a confidence interval. Concurrent throughput/memory telemetry
is not an isolated performance benchmark. Detached conditioning remains an
earlier diagnostic; it is not repeated in this bounded longer study.

Resource and retention declaration before launch: at most two of our GPU
workers across preflight/calibration/quality; estimated 3–5 aggregate GPU-hours
(about 1.5–2.5 wall hours with two available GPUs). Retain at most 4 GiB, with
8 GiB minimum free-space headroom; shared free capacity is approximately 310 GiB.
Retain frozen sources/configs/runtime/seeds/splits/optimizer records, metrics,
diagnostics, logs, audits and reports. Compiler caches are expected ≤3 GiB.
Retention is until explicitly removed. No periodic/recovery/final weights,
sample grids, official-test evaluation or new evaluator weights. Existing
workflows and retention are unchanged.

Prelaunch CPU checks: 58 passed, one CUDA-specific exact-layer check skipped.
The two new GPU preflights cover the DPLR/static paths used by this study.
[Test log](vit-split-lr-tests.log) · [Implementation](../vit_split_lr_study.py).

Launched in `runs/vit-split-lr-20261011T214507Z`. The frozen sources match
the tested workspace, and CPU-only preparation verified that CUDA was not
initialized. Two claimed GPU workers are running the new preflights.
[Live plan](vit-split-lr-plan.json).

Both compiled GPU preflights finished and passed the frozen evidence audit.
All applicable adapter parameters received nonzero gradients by step3. The
20 fresh calibration pilots are now underway; the 16 longer quality runs
remain gated on the declared update bounds. No sources or recipes changed
after launch.

[Audited preflight evidence](vit-split-lr-preflight.json).

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

[Recovery plan and preserved evidence](vit-split-lr-recovery-plan.md).

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

[Completed comparison and diagnostics](vit-split-lr.md).
