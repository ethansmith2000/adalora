# ViT condition magnitude and gradient-route results

The early probes do not show an obvious condition-magnitude or opposing-gradient problem. Condition RMS is already near one, and the reference condition/value gradients have positive global cosine in every measured block at steps 100/500/1200. Extra condition normalization changes the gradient geometry and slows early learning; half-gradient scaling produces mixed early validation changes. Keep the existing default and defer longer intervention training.

Completed 2026-10-13T01:17:25.664975+00:00: two compiled GPU preflights and six 1,200-step diagnostic prefixes on seeds 110/111. All seven audit checks passed on replay from the frozen runner. The sources/runtime/configs/optimizer/data/observed streams and paired initialization/fixed probes are audited. No recipe selection, full-horizon quality runs, weights, checkpoints or official-test evaluation.

All prefixes keep the 10,560-step LR schedule, Xavier base MLP initialization, global LR 5e-4, DPLR r8/s24, gain 1/sqrt(8), factor offsets, affine/low-rank multipliers 1, width 192/depth 6/heads 3/MLP hidden 768, batch 128, fused AdamW(.9,.999), compiled BF16, warmup 5%, cosine floor .01, matrix decay .05, label smoothing .1 and clip 1. The same split seed 123 gives 45k training/5k validation images. Every recipe has 4,342,378 parameters.

Gradient measurements use the true smoothed task-loss cotangent on four fixed validation images, with raw-model FP32 eager eval-mode. Three independent input leaves separate value, affine-condition and low-rank-condition VJPs. The value route includes the modulated projection and low-rank delta with conditions held fixed; it is not a plain-model gradient. Both intervention paths affect affine and low-rank heads.

Local output and gradient reconstruction checks passed; maximum relative gradient reconstruction error is 7.34e-8. The CPU tests also verify unchanged optimizer gradients/parameters/mode/RNG/hooks and unchanged short training trajectories with probes enabled/disabled. Zero-gradient alignment is explicitly undefined.

| Recipe, step 1200 | Condition RMS mean | Applied condition/value gradient RMS ratio | Condition/value cosine mean |
|---|---:|---:|---:|
| Current DPLR | 0.976 | 0.614 | +0.297 |
| Condition RMS | 1.000 | 0.649 | +0.310 |
| Half condition gradient | 0.979 | 0.321 | +0.297 |

The table averages 12 per-block records (six blocks × two seeds), not the cosine or norm ratio of one concatenated network gradient. Inspect the per-seed, per-block records in the JSON for heterogeneity.

Current condition RMS starts about 1.000, averages 0.993 at500 steps, and 0.976 at1200; endpoint block values span 0.966–0.987. Input normalization is already effective in these probes. Coefficient RMS at 1200 ranges 0.389–0.901 for a and 0.351–0.865 for b. These input/coefficient observations do not bound transform spectral gain or eliminate learned-basis gauge freedom.

Condition gradients are substantial: the reference mean condition/value norm ratio rises from 0.179 at 100 steps to 0.496 at 500 and 0.614 at 1200. Endpoint block ratios span 0.258–1.012. However, global condition/value cosine is positive in all 36 reference records at 100/500/1200 (range +0.110 to +0.454); all 12 endpoint block cosines are positive. Some token gradients oppose the value route: the mean endpoint negative-token fraction is 3.46%, and one block reaches 18.46%. This gives no evidence of widespread conflict in the measured early fixed-image setting, and does not prove that opposition is absent elsewhere.

At 1200, low-rank condition gradients have greater RMS than affine-condition gradients in the first three blocks for both seeds; affine-condition gradients have greater RMS in the last three. Low-rank-only global alignment is weaker than affine alignment, but remains positive in every reference endpoint block. A single total-gradient statistic would hide this variation.

RMS normalization forces condition RMS to 1.000 but leaves the average applied condition/value ratio at 0.649 and cosine at +0.310. It does not visibly reduce the proposed route burden in these probes. Even an almost-unit-RMS input can have a changed Jacobian when normalized again; this intervention also removes condition magnitude information.

Half-gradient scaling gives an applied ratio 0.321 while the raw ratio is 0.641. Its endpoint average cosine +0.297 is essentially the same as the reference, as expected from positive scalar scaling at fixed vectors. The trained states differ, so measured ratios are not exactly a half of reference. Forward identity at fixed weights does not imply unchanged learning.

| Recipe | Step1200 validation accuracy, seed 110/111 | Mean | Paired change versus current |
|---|---|---:|---|
| Current DPLR | 55.64%, 55.52% | 55.58% | Reference |
| Condition RMS | 49.50%, 51.74% | 50.62% | -6.14, -3.78 points |
| Half condition gradient | 54.48%, 56.34% | 55.41% | -1.16, +0.82 points |

These are schedule-prefix observations, not final-quality comparisons or candidate selection. Condition RMS slows early learning on both seeds (mean −4.96 points); half-gradient scaling is mixed (−1.16/+0.82, mean −0.17 points). Two seeds and early accuracy cannot establish long-horizon ranking or generalization. The planned mechanism measurements and absence of a clear problem do not motivate automatic longer training.

Combined/base activation RMS at the endpoint spans 1.20–2.44 for current, 1.17–2.49 for RMS-normalized and 1.12–2.55 for half-gradient. Similar ranges reinforce that neither intervention establishes output variance preservation or controls the learned transform’s gain. All underlying branch and basis/parameter records remain retained.

Scope limits: four fixed evaluation images, FP32 raw model, six blocks, two seeds and early schedule prefixes. These are local gradients at normalized hidden inputs, not complete upstream/network gradients or BF16 stochastic-training batch gradients. Positive cosine does not prove beneficial conditioning; negative cosine does not prove harmful conflict. Later training and external conditioning may differ.

Next priority: retain the default self-conditioning path and return to external-conditioning comparisons with strong AdaLN and static-capacity controls. This does not establish that external conditioning is better; it is a separate architectural question. No additional experiment is launched by this report.

The entire workflow finished in about 7.2 wall minutes, using 0.178 aggregate worker-hours. It retained about 260 MiB within the 4 GiB bound. All private supervisors are closed and our GPU workers are inactive. Historical evidence remains unchanged.

[Protocol](vit-condition-routes-protocol.json) · [Per-seed/per-block measurements](vit-condition-routes.json) · [Evidence index](vit-condition-routes-evidence.json) · [Audit](vit-condition-routes-audit.json) · [Export manifest](vit-condition-routes-manifest.json)
