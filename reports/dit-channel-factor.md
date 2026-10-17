# Completed channel-modulated static-factor and adapter-LR study

Channel modulation improves on static low-rank factors, especially at the higher adapter LR, but does not beat the dense reference. The s64/rank8 model at adapter multiplier4 reaches34.64 CFG2 FID, versus33.59 for dense at multiplier1, while using61.8% fewer adapter parameters (13.9% fewer total parameters). These are different operating rates within the same two-point search budget; they are not a same-rate or parameter-matched comparison. No new default, automatic selection or follow-up launch.

| Design | Adapter parameters | LR multiplier1: CFG2 FID | LR multiplier4: CFG2 FID |
|---|---:|---:|---:|
| dense_s32 | 2,834,944 | 33.59 ± 0.73 | 54.08 ± 28.76 |
| cp_s32_p128 | 1,606,144 | 34.25 ± 0.38 | 219.06 ± 261.07 |
| channel_static | 163,840 | 38.35 ± 0.49 | 37.04 ± 0.28 |
| channel_s32 | 623,104 | 38.04 ± 0.28 | 35.61 ± 0.36 |
| channel_s64 | 1,082,368 | 37.72 ± 0.38 | 34.64 ± 0.19 |

Final30k EMA, CFG2/DDIM50, 5k real/5k generated images. Means ± sample SD over fresh training seeds124/125, not confidence intervals. The same research holdout, evaluation noise and evaluators are reused. CFG1 results, all individual seeds, class agreement, timings, allocation and convergence remain in the linked evidence.

At multiplier4, s32 modulation improves over static by1.88/0.97 FID and s64 by2.73/2.06 on seeds124/125. Widening s32 to s64 improves FID by0.85/1.09. Moving from multiplier1 to4 improves static/s32/s64 FID in both seeds, by mean1.32/2.43/3.08. At multiplier1, modulation also improves on static in both seeds, but the gaps are smaller:0.46/0.16 for s32 and1.24/0.02 for s64. This is evidence that conditioning helps these compact factors; it is not evidence that their initial Jacobians, function-update scales or optimal LRs match dense/CP.

Compared with dense at multiplier1, s64 modulation at multiplier4 has paired FID gaps+1.44/+0.67. Compared with CP at multiplier1, its gaps are−0.004/+0.80, so no consistent guided-quality advantage over CP. CFG2 class agreement averages85.20% for s64 at4, versus88.58% dense at1,87.39% CP at1 and81.13% static at4. Its CFG1 FID47.51 is close to dense47.17, but this does not close the primary guided-FID/class-agreement gap.

The higher adapter LR is unsafe as a general replacement for the original rate in this grid. Dense seed125 ends at FID74.42 and44.92% class agreement (other seed33.75); CP seed124 ends at403.66 and10.18% agreement (other seed34.46). Both workers completed with finite evidence. CP EMA noise MSE rises from0.06305 at5k to0.88477 at10k and ends0.80451. The affected dense seed has0.74768/0.87440 at5k/10k, then partially recovers to0.06279. The 12-step preflights establish compatibility and early gradient wake-up, not long-run optimization stability. Keep these poor endpoints in the aggregate; no failed-quality seeds are discarded. This study does not isolate a gauge, gain or factor-magnitude mechanism for the degradation.

Observed training throughput at the relevant operating points is2888 images/s dense at1,2661 CP at1,2872 channel s32 at4 and2777 channel s64 at4. Static at4 reaches3319. Whole-training peak allocation is1.17GiB dense,1.15GiB CP,1.13GiB channel s32 and1.14GiB channel s64; the static control is higher at1.34GiB. Parameter savings therefore do not translate directly into whole-training allocation savings. These are concurrent training observations, not isolated speed or activation-memory benchmarks.

All25 workers completed: five compiled preflights plus twenty full30k runs. All nine final audits passed in the controller and independent frozen-runner replay, covering sources/configs, paired initialization/streams, actual optimizer ownership/rates, gradients, evaluation budgets, exits/claims and storage. The workflow ran October14,18:05–22:40 UTC, about4h35m; about1.32GiB is retained within the4GiB bound. Private supervisors are closed and adalora claims released. No weights, checkpoints, official test data or evaluator downloads.

The compact channel design is worth retaining as a quality/parameter tradeoff; dense remains the quality reference. More seeds and a bounded intermediate-rate comparison would be needed before promoting a compact recipe. Wider coefficient heads, shared/nonlinear projector trunks and causal sequence conditioning remain separate roadmap choices. No extra GPU work was launched during this review.

[Comparison](dit-channel-factor.json) · [Evidence](dit-channel-factor-evidence.json) · [Audit](dit-channel-factor-audit.json) · [Protocol](dit-channel-factor-protocol.json) · [Original plan](dit-channel-factor-plan.md)
