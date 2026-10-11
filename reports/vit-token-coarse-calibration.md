# ViT coarse calibration

All 28 pilots and two GPU preflights completed and passed their individual evidence audits. The calibration gate failed; zero ten-epoch quality runs started. Both private controllers and worker supervisors are closed. No model weights were saved.

| Family | Adapter LR multiplier | Minimum update ratio | Maximum update ratio | All ratios within0.5–2 |
|---|---:|---:|---:|---|
| s12 | 0.25 | 0.130 | 0.207 | no |
| s12 | 1 | 0.471 | 0.625 | no |
| s12 | 4 | 1.479 | 2.504 | no |
| s48 | 0.25 | 0.488 | 0.583 | no |
| s48 | 1 | 1.667 | 1.845 | yes |
| s48 | 4 | 6.958 | 10.398 | no |
| s24_detached | 0.25 | 0.258 | 0.386 | no |
| s24_detached | 1 | 0.908 | 1.199 | yes |
| s24_detached | 4 | 3.837 | 6.262 | no |
| static_r8_diagonal | 0.25 | 0.019 | 0.022 | no |
| static_r8_diagonal | 1 | 0.063 | 0.083 | no |
| static_r8_diagonal | 4 | 0.230 | 0.340 | no |
| static_r8_diagonal | 16 | 1.199 | 2.241 | no |

s12 at unit LR is borderline low (minimum0.471); its4× rate overshoots. The static control at4× remains low, while16× crosses the upper bound (maximum2.241). s48 and detached s24 have valid unit-rate candidates. The original fixed grid was coarse; no bounds are relaxed.

The eight additional pilot cases are s12×2, s48×0.5, and static×8/×12, each on the same two pilot seeds. They are chosen from update bracketing, not held-out accuracy, and use unchanged training sources/settings and the same observed streams. Reuse the original pilots only after audit. A separate protocol retains the original failed gate and limits this refinement to eight new cases before the originally planned16 quality runs.

At unit rates, 500-step mean validation accuracy is37.46%/37.30%/38.85% for s12/s24/s48 and35.63% for detached s24. These early values are not matched-update quality comparisons, and they do not select rates. The first ten-epoch results remain pending.

[Exact ratios and provenance](vit-token-coarse-calibration.json) · [Raw pilot evidence](../runs/vit-token-20261011T070503Z/calibration/summary.json)
