# Wider MLP control: optimization and initialization diagnostic

The parameter-matched wider MLP underperforms plain ViT by 0.68 validation
percentage points on average, with seed-wise differences −1.28/−0.08. This is
an untuned control under one shared optimizer recipe; its underperformance
weakens capacity-based interpretation of the DPLR result. A parameter match
leaves initialization gain and preferred optimizer settings different.

| Model | Last-epoch augmented training accuracy | Final validation accuracy |
|---|---:|---:|
| Plain |85.43%|78.45%|
| Wider MLP |83.78%|77.77%|

Training metrics cover all 45,000 images in the final epoch, weighted by logged
window sizes. Both train and validation accuracy are lower in the wider model.
This suggests an optimization/scaling issue worth checking rather than a
simple case of stronger training fit with worse generalization. It does not
establish the cause: augmentation and regularization affect training, and
only two seeds were measured. Training uses label-smoothed loss while
validation uses unsmoothed cross-entropy, so their losses must not be compared
directly. Within the training setup, mean smoothed loss is
0.8287 plain versus
0.8639 wider.

The model initializes every Linear with fixed `std=0.02`, including both MLP
projections. Increasing hidden width changes the output projection's fan-in:

```python
# plain
fc1 = Linear(192, 768)
fc2 = Linear(768, 192)

# parameter-matched wider control
fc1 = Linear(192, 1483)
fc2 = Linear(1483, 192)

# Both retain the same weight-init std=0.02.
# Under similarly distributed hidden activations:
initial_output_rms_ratio = sqrt(1483 / 768)  # 1.390
```

A CPU-only FP32 probe feeds identical normalized random tokens `[2,65,192]`
to the isolated untrained MLP branches across all six blocks and seeds 102/103.
The observed average wide/plain output RMS is
**1.379×**. Temporarily multiplying
only the wider `fc2.weight` by `sqrt(768/1483)` (0.7196)
changes that average to 0.993×.
This supports the existence of an initial gain difference. It does not show
that rescaling improves training: inputs are synthetic, the layers are
untrained, and no training run used this change. Wider-model shapes also
change random-number consumption, so the wider and plain/DPLR models do not
have identical shared base initialization in the original study.

Revised next priority: run a bounded plain/wider control check with common
seeds/data/schedule, a small learning-rate bracket and explicit initial MLP
gain choices (legacy fixed std versus a declared reference-gain correction).
Separate LR effects from initialization effects, retain the untouched recipe
as a control, then compare selected controls to DPLR on fresh confirmation
seeds. All model families should receive a declared, comparable tuning budget
for a stronger claim. This takes priority over interpreting the matched
wider MLP gap as evidence against static capacity or merely adding DPLR seeds.
DPLR's paired improvement over diagonal conditioning remains measured; the
capacity interpretation needs the stronger control.

No trainer/model source, experiment weights or historical results were changed.
No GPU job was launched. CPU preparation/probing kept CUDA uninitialized.
[Structured diagnostic](vit-capacity-diagnostic.json) ·
[Reproduction script](../diagnose_vit_capacity.py) ·
[Original comparison](vit-split-lr.md).
