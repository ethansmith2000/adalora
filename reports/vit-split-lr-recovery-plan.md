# Recover the interrupted 30-epoch ViT comparison

The original study is `runs/vit-split-lr-20261011T214507Z`. Its two preflights
and all 20 calibration pilots finished and passed the original frozen auditor
again on October 12. Calibration selected s48 low-rank LR 0.5 and static split
low-rank LR 16, with affine LR 1; the static joint12 diagnostic retains both
rates at 12. No selection or training recipe changes are made.

At the October 12 status check, the controller socket refused connection, all
old worker PIDs were dead, and container PID 1 started at 01:21:47 UTC. The last
recorded quality progress was October 11 at 22:12:57 UTC: four completed, six
submitted, two active. This establishes that the study was interrupted before
the restart; it does not establish the exact interruption time or its cause.
The original protocol still says `quality`, because its controller was killed
before it could record failure. Preserve that raw protocol and every original
log/source/config/output rather than rewriting the historical record.

Four fully completed quality cases can be reused only after full evidence
audit. The remaining 12 cases restart from their originally declared seeds,
not from checkpoints; no model weights were saved.

| Completed case | Final validation accuracy |
|---|---:|
| Wider static capacity, seed 102 |77.40%|
| Wider static capacity, seed 103 |78.14%|
| Static split, seed 102 |78.72%|
| Static joint12, seed 103 |78.68%|

The dynamic methods have no completed endpoint yet. These incomplete-cohort
readings do not establish a method ranking; the single static split and joint
results use different seeds and low-rank rates. The wider static mean is 77.77%,
but its previous ten-epoch mean was from other seeds and another schedule.

Use a separate recovery workflow with a byte-identical copy of the original
trainer/model/orchestrator sources, plus the recovery controller and its tests.
Pin the original protocol/calibration/preflight/calibration-summary/quality
screen/claim/exit-log/progress hashes. A declared audit view limits the original
quality screen, claim map and protocol count to the four completed workers;
it points to their unchanged original output paths and original frozen source
snapshot. The original worker files are never copied over or overwritten, and
the stale original sockets/PID/claim files are left intact.

Only final results with 10560 steps, a final `complete` metric event and a
successful worker exit are reusable. Audit sources/configs/runtime/splits,
parameter counts, initial weights, zero start and gradient wakeup, optimizer
group membership/decay/rates, every logged schedule, observed data streams,
evaluation counts and the no-weights policy. Verify the current runtime and
trainer source hashes before launch. Re-audit the old pilots and update-only
selection; do not rerun them or change rates. At final completion, combine the
four original and 12 replacement cases, require all 16 unique declared method/
seed pairs, and audit source/runtime/split/initialization/observed stream pairing
across the pre-interruption and replacement cohorts.

Resource declaration before launch: at most two of our GPU workers, estimated
2–4 aggregate GPU-hours (about 1–2 wall hours when two GPUs are available).
Retain at most 4 GiB in the new workflow with 8 GiB minimum free-space headroom;
shared capacity is approximately 309 GiB free. Retain compact source/config/
runtime/seed/split/optimizer/metric/diagnostic/log/audit/report evidence and
declared compiler caches (expected ≤3 GiB) until explicitly removed. No weights,
periodic/recovery/final checkpoints, grids, official-test evaluation or new
evaluator weights. Existing workflow retention is unchanged. No broad cleanup
of stale claims or other projects' processes/artifacts is authorized or needed.

Thirty-two recovery and split-rate CPU checks passed, including strict completion
criteria, exact remaining-case selection, immutable rates, failure-before-launch
guards and cross-cohort pairing failures. This recovery does not add methods,
seeds or a quality budget to the original 16-case study. It repeats the two
interrupted attempts from seed and runs the ten cases that had not started.

[Original protocol](vit-split-lr-plan.md) ·
[Recovery implementation](../recover_vit_split_lr.py) ·
[Test log](vit-split-lr-recovery-tests.log).

Launched in `runs/vit-split-lr-recovery-20261012T013129Z`. Four old cases
passed their full audit; twelve remaining cases are scheduled, with two
currently active. The frozen source snapshot and hashes of original partial
records are preserved. [Live state](vit-split-lr-recovery-plan.json) ·
[Interruption and completed readings](vit-split-lr-interruption.json).

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
