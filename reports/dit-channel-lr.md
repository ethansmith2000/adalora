# Completed fresh-seed channel/dense/CP adapter-LR follow-up

The follow-up reproduces different learning-rate responses across parameterizations. Dense at multiplier1 remains the lowest observed mean FID in the grid. CP benefits modestly from the intermediate multiplier2. Channel s64 improves with higher adapter LR, but2-to4 brings only a small additional gain and does not close its quality gap. Multiplier4 again severely degrades one dense seed and one CP seed. No new default or automatic follow-up is selected.

| Design | Adapter parameters | LR multiplier1: CFG2 FID | LR multiplier2: CFG2 FID | LR multiplier4: CFG2 FID |
|---|---:|---:|---:|---:|
| dense_s32 | 2,834,944 | 33.65 ± 0.44 | 34.65 ± 1.29 | 46.55 ± 18.25 |
| cp_s32_p128 | 1,606,144 | 34.18 ± 0.46 | 33.90 ± 0.68 | 216.81 ± 257.24 |
| channel_s64 | 1,082,368 | 36.82 ± 0.55 | 35.78 ± 0.37 | 35.58 ± 0.22 |

Final30k EMA, CFG2/DDIM50,5k real/5k generated images. Means ± sample SD over fresh training seeds126/127, not confidence intervals. Both CFG scales, classifier agreement/confusions, every individual seed, costs and convergence are retained in the comparison/evidence JSON.

The prescribed channel4-minus-dense1 paired FID gaps are+2.08/+1.77, mean+1.92; channel4-minus-CP1 gaps are+1.56/+1.23, mean+1.40. Channel s64 retains61.8% fewer adapter parameters and13.9% fewer total parameters than dense, but this fresh-seed quality gap is wider than the prior screen's1.05 mean. Historical means are context, not substituted for any current result.

The new intermediate point helps CP in both seeds:2-minus1 FID−0.44/−0.13, mean−0.28. Its multiplier2 mean33.90 is descriptively0.25 higher than dense at1 (33.65), using43.3% fewer adapter parameters and9.8% fewer total parameters. Choosing these observed operating points after looking at the grid is exploratory; this is not an independently confirmed optimum or an equivalence test. Dense2-minus1 is mixed (+2.22/−0.23).

Channel FID improves in both seeds for1-to2 (−1.70/−0.39) and2-to4 (−0.09/−0.31). The last step gives only0.20 mean additional improvement. Its CFG1 FID worsens slightly in both seeds from2 to4 (+0.06/+0.19), while guided class agreement improves from83.76% to85.04%; dense at1 reaches88.85%. Thus guided-FID gains should not be read as a uniform improvement in all quality measures or a calibrated gain/variance-preservation result. No transform normalization, initialization or function-update matching was changed.

At multiplier4, dense seed126 ends at FID59.46/65.56% class agreement, versus33.65/87.94% for seed127. CP seed126 ends at398.71/10.08%, versus34.92/87.40% for seed127. The affected dense EMA noise MSE rises from0.06365 at5k to0.82732 at10k and partially recovers to0.06123; affected CP reaches0.82235 at10k and ends0.78671. All workers completed with finite evidence. Poor finite endpoints are retained in every aggregate. This repeats the previous qualitative high-rate sensitivity, but four endpoint seeds across two screens do not establish a precise failure probability or prove a gauge/magnitude mechanism. Twelve-step preflight success cannot establish long-run optimization stability.

| Design/rate | CFG2 FID seed126 | CFG2 FID seed127 |
|---|---:|---:|
| dense_s32_a1 | 33.342 | 33.968 |
| dense_s32_a2 | 35.566 | 33.741 |
| dense_s32_a4 | 59.460 | 33.648 |
| cp_s32_p128_a1 | 33.858 | 34.507 |
| cp_s32_p128_a2 | 33.421 | 34.381 |
| cp_s32_p128_a4 | 398.706 | 34.918 |
| channel_s64_a1 | 37.210 | 36.432 |
| channel_s64_a2 | 35.513 | 36.042 |
| channel_s64_a4 | 35.420 | 35.736 |

Observed concurrent training throughput is2899 images/s dense at1,2582 CP at2 and2801 channel s64 at4. Whole-training peak allocation is about1.17/1.15/1.14GiB respectively. CP's parameter savings do not improve its observed throughput. These are descriptive concurrent measurements, not isolated speed or activation-memory benchmarks.

Twenty-one workers completed: three compiled preflights and eighteen full30k runs. All nine final audits passed in the controller and independent frozen-runner replay, including all38 snapshot files,36 unchanged reference sources, configs, actual optimizer ownership/rates, initialization/stream pairing, gradient wake-up, evaluator/guidance budgets, exits, claims and storage. The workflow ran October15,06:14–10:28 UTC, about4h14m. About1.12GiB is retained within the4GiB bound. Private supervisors are closed; adalora claims released. No model weights/checkpoints, official test data or evaluator downloads.

Fresh training seeds do not make the reused holdout/noise or the prior choice of channel s64 independent. Static and channel s32 controls were not repeated here; modulation-over-static remains prior-screen evidence. Both seeds/all rates remain visible; no best-intermediate selection or pooled historical replacement. These results support retaining dense as the quality reference and CP/channel as distinct parameter/quality/speed tradeoffs. They do not justify automatically expanding the LR grid or promoting a compact default. No new experiments launched during this review.

[Comparison](dit-channel-lr.json) · [Evidence](dit-channel-lr-evidence.json) · [Audit](dit-channel-lr-audit.json) · [Protocol](dit-channel-lr-protocol.json) · [Original plan](dit-channel-lr-plan.md)
