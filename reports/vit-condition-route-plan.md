# ViT condition magnitude and gradient-route diagnostic

The current tuned ViT comparison gives no clear DPLR validation advantage.
Condition input is already post-LayerNorm, and an older ten-epoch detached
control underperformed full-gradient conditioning. Extra normalization or
gradient attenuation therefore has no established benefit. This small check
tests the proposed mechanism before a longer ablation or regularization grid.

Run three fixed DPLR r8/s24 recipes on **two new seeds, 110/111**, for
**1,200 steps of the existing 10,560-step schedule**:

| Recipe | Condition input | Condition gradient scale |
|---|---|---:|
| Current reference |Post-LayerNorm `h`|1|
| Condition RMS |Parameter-free RMS normalization of `h`|1|
| Half condition gradient |Post-LayerNorm `h`|0.5|

Both interventions apply to scale/shift and low-rank coefficient generation;
their contributions are measured separately. The value input stays `h`.
Keep Xavier initialization, width 192/depth 6/heads 3/MLP hidden 768, batch 128,
global LR 5e-4, relative affine/low-rank rates 1, DPLR gain 1/sqrt(8), factor
offsets enabled, fused AdamW(.9,.999), compiled BF16, weight decay .05,
label smoothing .1, clip 1, warmup 5%, cosine floor .01 and split seed 123.
The full 45k/5k research split remains in use. No old worker result is reused.

```python
c = h                                          # reference
c = h * rsqrt(mean(h.float() ** 2) + 1e-6)      # RMS over final feature axis
c = h.detach() + 0.5 * (h - h.detach())         # same forward, reduced d/dh
```

RMS is parameter-free, preserves direction, and removes magnitude information.
Accumulation is FP32 and the reciprocal scale is cast back to the input dtype.
The default `none`/gradient-scale 1 path returns the same input directly,
introduces no parameters or new random draws, and preserves old forward and
backward behavior. Existing historical training snapshots remain frozen.

At diagnostic steps 0/1/2/3/5/10/50/100/250/500/800/1200, record the existing
fixed-token full-MLP/low-rank updates and live branch magnitudes, plus a
**task-loss gradient decomposition on four fixed validation images**. Use
raw-model FP32 eager eval-mode with the same label smoothing as training.
Capture the true loss cotangent at each MLP input projection, then locally
replay with independent value, affine-condition and low-rank-condition leaves:

```python
grad_h = grad_value + grad_condition_affine + grad_condition_lowrank
```

The value route includes the modulated projection and low-rank delta with
conditions held fixed; it is not the gradient of a plain model. Condition
Jacobians include the input normalization. Store both raw and applied condition
gradients, so the explicit factor 0.5 is not mistaken for improved alignment.
Verify the reconstructed output and gradient against the actual full-model
task-loss gradient at the shared normalized hidden input.

Retain per-block gradient RMS, condition/value norm ratio, global cosine,
mean token cosine, valid-token count and negative-token fraction. Zero-route
alignment is undefined and stored as `null`, never as apparent cosine 0.
Also record input/condition RMS, coefficient a/b RMS and token RMS p95, and
LayerNorm affine weight/bias RMS. Existing basis parameter norms, branch
RMS and fixed function updates remain recorded alongside these measurements.

These are local normalized-hidden-state VJPs on fixed evaluation images,
not training-batch or whole-network conflict. Opposing gradients can be useful;
negative cosine is descriptive, not evidence of an optimization defect.
Inspect per-block and per-seed observations rather than only an aggregate.
No magnitude/cosine threshold selects a candidate. Prefix validation curves
are retained for integrity and context, **not early quality selection**.

Two compiled 12-step GPU preflights (current reference and combined RMS+0.5)
must pass gradient reconstruction, source/runtime/config/data/optimizer and
gradient-wakeup audits before all six prefixes. Final audits require paired
initialization, fixed probe images/labels, observed training streams, finite
route measurements and unchanged parameter counts (4,342,378 each).
There is **no automatic long-horizon training or recipe selection**.

Resource/retention declaration before launch: at most **two of our GPUs**,
including queued workers, under private supervisor/cooperative GPU claims.
Eight workers total: 2 compatibility preflights +6 diagnostic prefixes.
Expected 0.2–0.6 aggregate GPU-hours, roughly 10–20 wall minutes including cold
compilation. Require 8 GiB free headroom; retain at most 4 GiB of compact sources,
configs, seeds/runtime/splits, metrics, logs, audits and diagnostics. Compiler
caches have a compatibility/performance purpose, expected at most 3 GiB.
Keep declared artifacts until explicitly removed; historical retention is
unchanged. **No weights, checkpoints, official-test evaluation or downloads.**

[Implementation](../vit_condition_route_study.py) ·
[Route tests](../test_vit_condition_routes.py) ·
[CPU test log](vit-condition-route-tests.log) ·
[Live plan](vit-condition-route-plan.json).

CPU validation: **97 checks passed**, including exact short-training trajectory
equality with diagnostics enabled/disabled. GPU compatibility audits remain
pending and gate all six prefixes. Raw workflow: `runs/vit-condition-routes-20261013T010835Z`.

## Completion record

All two preflights and six prefixes completed October 13 at 01:17 UTC; all seven final audits passed. Condition RMS stayed near one, and reference condition/value global cosine was positive in all 36 blocks/observations at steps 100/500/1200. Extra RMS normalization slowed early validation learning in both seeds; half-gradient scaling was mixed. These local early probes give no clear mechanism-based reason for longer intervention training. Defaults stay unchanged. Private supervisors are closed and our workers inactive. [Completed measurements](vit-condition-routes.md).
