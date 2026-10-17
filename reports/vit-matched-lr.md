# ViT matched Xavier and learning-rate results

Plain ViT and DPLR remain effectively matched at both fixed global LRs. The large architecture/LR flip on the two tuning seeds did not reproduce in these four seeds. There is no clear final-validation benefit that justifies DPLR’s additional parameters/compute in this 30-epoch setting.

Completed 2026-10-12T17:13:29.508162+00:00: eight new full-budget runs plus eight audited existing cells from `/workspace/adalora/runs/vit-initializers-20261012T070919Z`. All seven audit checks passed on replay from the frozen runner. Training sources, Python/packages/CUDA wheel, configs, optimizer groups, split, initialization, probe inputs and observed image/label streams match across cohorts. Original evidence remains unchanged.

Each method uses the same `xavier_gelu` initialization, width 192/depth 6/heads 3/MLP hidden 768, r8/s24 for DPLR, 30 epochs/10,560 steps, batch 128, 45k/5k split seed 123, compiled BF16 and fused AdamW(.9,.999), weight decay .05, label smoothing .1, clip 1, warmup 5% and cosine floor .01. Global LR scales base and conditional groups together; DPLR affine and low-rank multipliers remain 1. Factor offsets remain enabled. There is no new recipe selection or grid expansion.

| Global LR | Plain accuracy, mean ± sample SD | DPLR accuracy, mean ± sample SD | Mean DPLR − plain |
|---:|---:|---:|---:|
| 0.00025 | 79.465 ± 0.250% | 79.350 ± 0.270% | -0.115 points |
| 0.0005 | 79.440 ± 0.343% | 79.525 ± 0.287% | +0.085 points |

All endpoint accuracies use the same 5,000-image research holdout. Paired differences are in percentage points, seed order 106,107,108,109.

| Contrast | Per-seed differences | Mean ± sample SD |
|---|---|---:|
| DPLR − plain, global LR 0.00025 | -0.440, +0.000, -0.240, +0.220 | -0.115 ± 0.287 |
| DPLR − plain, global LR 0.0005 | -0.460, -0.260, +0.440, +0.620 | +0.085 ± 0.525 |
| Interaction: gap at 5e-4 − gap at 2.5e-4 | -0.020, -0.260, +0.680, +0.400 | +0.200 ± 0.420 |

DPLR at the lower LR has one win, one tie and two losses; at the higher LR it has two wins and two losses. The paired mean interaction is +0.200 points with sample SD 0.420; its direction splits two/two. This does not support a stable LR-dependent DPLR benefit. In contrast, the two original tuning seeds suggested DPLR gaps +0.51 points at the lower LR and −0.76 at the higher LR. Those tuning observations were too unstable to guide an architectural conclusion.

Parameter totals are 2,690,122 plain versus 4,342,378 DPLR (+61.4%). This comparison controls nominal initializer and LR, rather than function-update magnitude or optimal per-architecture hyperparameters. A small positive mean at one LR is not a new selected winner.

| Method | Global LR | Final augmented-training window accuracy | Validation cross-entropy |
|---|---:|---:|---:|
| plain | 0.00025 | 87.55% | 0.6619 |
| plain | 0.0005 | 88.58% | 0.6646 |
| dplr | 0.00025 | 92.33% | 0.6937 |
| dplr | 0.0005 | 90.83% | 0.6664 |

Each final training window has 7,624 augmented examples within epoch 30; these are logged windows rather than whole-epoch or clean-training evaluations. DPLR fits training better at both LRs without clear validation-accuracy gains. At the lower LR, DPLR also has higher validation cross-entropy than plain. This keeps generalization/regularization as a plausible next question; it does not establish overfitting or a causal explanation. Complete curves and fixed-token MLP/low-rank interval diagnostics remain in the audited worker metrics.

The eight new runs used 0.852 aggregate worker-hours and finished in about 27.6 wall minutes. Only two of our GPU workers were active at any point; the private supervisors are now closed, and our workers are inactive. About 140 MiB of compact records and declared compiler cache is retained, below the 4 GiB bound. No weights, checkpoints, official-test evaluation or new GPU launch for this status/report.

These seeds and the research holdout were already inspected before this follow-up was proposed. This is an exploratory diagnostic, not independent confirmation. Four seeds, one split and 30 epochs limit the claim, and sample SD is not a confidence interval. The broader external-conditioning/DiT question remains separate.

Next priority remains a bounded generalization or stronger-control experiment rather than expanding projector capacity. Compare regularization with equal tuning budgets and retain tuned plain; an equally tuned diagonal/static-low-rank control can isolate whether conditional low-rank structure adds value. No additional experiment is selected or launched by this report.

[Protocol](vit-matched-lr-protocol.json) · [Results](vit-matched-lr.json) · [Evidence index](vit-matched-lr-evidence.json) · [Audit](vit-matched-lr-audit.json) · [Export manifest](vit-matched-lr-manifest.json)
