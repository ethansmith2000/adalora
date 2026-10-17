# Fresh-seed channel/dense/CP adapter-LR follow-up

The completed channel-factor screen found a useful parameter/quality tradeoff:
channel s64 at adapter multiplier4 reached34.64 CFG2 FID, versus33.59 for
dense at1, using61.8% fewer adapter parameters. Channel gains and the wider
coefficient bottleneck helped, but the higher rate severely degraded one
dense seed and one CP seed. Those finite poor endpoints remain in the results.
This follow-up adds an intermediate rate and repeats both endpoints on fresh
training seeds, before changing projector architecture or declaring a default.

## Prespecified grid

| Design | Adapter parameters | Total parameters | Adapter LR multipliers |
|---|---:|---:|---|
| Dense s32 |2,834,944|12,593,712|1,2,4|
| CP s32/p128 |1,606,144|11,364,912|1,2,4|
| Channel s64 |1,082,368|10,841,136|1,2,4|

Every cell trains for a fresh30k steps on seeds126/127: eighteen quality runs.
The same three-point search budget applies to every architecture. Multiplier2
is the geometric midpoint between1 and4; base/global LR stays8e-4. The adapter
group starts at8e-4,1.6e-3 or3.2e-3 before warmup/schedule scaling. All adapter
parameters, including coefficient heads, decoder factors/scales and offsets,
receive the multiplier; common base/AdaLN parameters keep the base rate.

The channel width was chosen from the prior screen. Fresh seeds therefore
replicate training behavior, while architecture selection and the reused
research holdout/evaluation noise are not independent confirmation. Static
and channel-s32 controls are not repeated in this focused LR study. The prior
modulation-over-static evidence remains a separate finding.

All training/model/diagnostic/helper files are copied byte-for-byte from
`runs/dit-channel-factors-20261014T175748Z/runner`. Only this study's controller
and tests are added. Preserve transform rank8, coefficient widths, static
offsets, normalized value input, parallel MLP-input delta and gain1/sqrt(8).
No new normalization, gain, initialization, activation or muP rule is tested.

## Training, evaluation and checks

Keep width256/depth8/heads4/patch4/MLP1024, batch128, fused AdamW betas0.9/0.999,
weight decay0, clipping1, BF16/dynamic compilation, EMA0.999, class dropout0.1,
50-step warmup and30k cosine schedule with minimum ratio0.01. Base/null weights
and data/noise/dropout streams are paired within seed. Every design has
identical initial parameters across all three rates; dense/CP heads are
paired at s32. Dense/CP Jacobians are not matched to the channel design.

Three compiled12-step preflights at multiplier4 cover every architecture,
including128/128/72 ragged batches, diagnostics, actual optimizer group rates,
all-route gradient wake-up, raw/EMA validation and guided generation. Every
preflight must complete and pass its independent evidence audit before the
quality queue opens. Preflights check compatibility; they cannot establish
long-run optimization stability. Numerical/worker failure preserves evidence
and withholds later work. Severe but finite poor quality remains an endpoint.

Use the same45k/5k CIFAR-10 research split seed123, evaluation noise seed2027,
sampling seed2028 and cached hash-pinned Inception/auxiliary classifier. Raw
and EMA validation every5k; intermediate CFG1 generation at10k/20k; final
same-state/noise EMA CFG1/2 with DDIM50 and5k real/5k fake images. Primary
comparison is final30k CFG2 FID. Retain CFG1, class agreement/confusion,
per-seed gaps, convergence, update/magnitude diagnostics and every endpoint.

Report all nine design/rate aggregates with sample SD, same-rate design
contrasts, and within-design2-minus1,4-minus2 and4-minus1 paired FID changes.
Separately report the prespecified channel4-minus-dense1/CP1 operating-point
gaps; these compare different rates. No best-intermediate selection, automatic
recipe promotion, longer confirmation, pooled historical replacement, grid
expansion, dedicated speed benchmark, shared/nonlinear projector or causal
retrieval/LM launch is authorized by this finite protocol.

## Resources and artifacts

At most two of our GPU workers or GPU claim waiters, managed through private
supervisor and `/workspace/bin/gpu-claim run`. Expected7–10 aggregate GPU-hours,
approximately4–5 wall hours with two available GPUs. Require8GiB free before
launch for the4GiB artifact bound and shared headroom.

Retain frozen source/version provenance, configs, seeds/splits, runtime,
metrics/diagnostics, logs, audits, summaries and manifests. Compiler caches
are expected below3GiB. Deliberately retain eighteen final100-image EMA CFG1
diagnostic grids and three preflight10-image grids. Keep these declared
artifacts until explicitly removed. No periodic/final/recovery checkpoints,
model weights or generated datasets. Existing evaluator weights are reused
without downloading or changing retention; no official-test evaluation.


## Launch verification

Launched October15 at06:14 UTC. All three compiled higher-rate preflights
passed the controller audit and independent frozen-runner replay by06:18 UTC.
Every adapter tensor received gradients by step5. Thirty-six workspace and36
frozen CPU checks passed; all21 configs and18 initial models/optimizer routes
were verified without CUDA. All36 historical source files remain unchanged
inside the38-file frozen runner. The eighteen full30k quality runs are queued
with two active workers. Final quality and audits remain pending.
[Preflight evidence](dit-channel-lr-preflight.json) ·
[CPU validation](dit-channel-lr-cpu-validation.json) ·
[Launch manifest](dit-channel-lr-launch-manifest.json).
