# ViT token self-conditioning results

All 16 ten-epoch runs completed. Final validation accuracy on the 5,000-image research holdout; paired fresh seeds 98/99. Means ± sample SD, not confidence intervals. Transform rank `r=8`; `s` is the projector bottleneck/basis width. Dynamic methods enable factor offsets.

| Method | Parameters | Adapter LR multiplier | Accuracy (%) | Training peak MiB |
|---|---:|---:|---:|---:|
| plain | 2,690,122 | 1 | 65.96 ± 0.14 | 427.3 ± 45.1 |
| diagonal | 3,134,794 | 1 | 69.04 ± 1.39 | 478.7 ± 56.7 |
| static_r8_diagonal | 3,180,874 | 12 | 66.64 ± 0.85 | 495.8 ± 50.7 |
| static_capacity | 4,341,772 | 1 | 63.16 ± 0.31 | 615.5 ± 45.2 |
| s12 | 3,761,626 | 2 | 68.40 ± 0.76 | 515.5 ± 32.6 |
| s24 | 4,342,378 | 1 | 69.82 ± 0.96 | 536.8 ± 26.3 |
| s48 | 5,503,882 | 0.5 | 70.05 ± 0.16 | 595.5 ± 26.8 |
| s24_detached | 4,342,378 | 1 | 68.72 ± 0.17 | 538.8 ± 30.4 |

DPLR s24 improves over diagonal conditioning by 0.48/1.08 percentage points in seeds 98/99 (mean +0.78), with 38.5% more total parameters. The wider s48 adds 26.7% parameters and approximately 10.9% mean observed peak allocation over s24 for +0.23 points on average; its paired changes are −0.56/+1.02 points. This suggests diminishing returns, without establishing a preferred width conclusively. Detaching the condition lowers s24 accuracy by 1.90/0.30 points (mean 1.10); this changes gradient flow, not the forward conditioning source.

The larger static MLP matches s24 total parameters within 606, but scores 63.16%. Its different MLP shape/initialization and untuned learning rate make this a capacity control under this recipe, not proof that widening is intrinsically worse. The static rank8+diagonal control scores 66.64% at adapter multiplier12; low-rank update matching does not make its entire optimization comparable.

The original 28-pilot coarse gate remains failed and preserved. Eight additional fixed-rate pilots produced a valid 36-case selection without changing the 0.5–2 bounds or selecting on accuracy. All selected methods also remain within the early bounds on fresh quality seeds at steps100/250/500. This transfer result does not match diagonal updates or the complete learning trajectory.

Endpoint combined-to-base activation RMS on fixed live images ranges from 11.69–19.68 across blocks/seeds for the static rank8+diagonal control, versus 1.45–3.67 for s24. The adapter multiplier changes its scale/shift parameters as well as low-rank factors. This is an optimization confound to investigate, not evidence that the low-rank operation diverged; all recorded results are finite.

Recommended next comparison: hold scale/shift LR fixed while calibrating the low-rank group, retain the diagonal and static controls, and extend s24 plus key controls to a longer schedule with fresh seeds. s24 is a reasonable provisional default; no new GPU jobs were launched by this status check.

All eight final workflow audit checks passed when replayed from the frozen runner. Both old preflights and all36 calibration pilots remain source/config/runtime/split/initialization/stream audited. All private supervisors are closed, our GPU claims are released, and no weights, checkpoints or official-test evaluations were produced. Retained raw artifacts are about524MiB, within the declared4GiB bound. Recorded elapsed time totals0.644GPU-hours for quality and0.097 for the eight added pilots; these include worker startup/evaluation, are concurrent telemetry, and are not isolated speed benchmarks.

Limits: Two fresh quality seeds and reused research holdout; sample SD is not a confidence interval. Offsets are the forward default by user preference; this study does not repeat the offset ablation. s24 has the near-parameter-matched static-capacity control; s12/s48 are width trends with different budgets. All norms retain static affine parameters across controls; attention remains ordinary self-attention. Factor offsets are treated as biases with zero weight decay; learned factor bases retain weight decay .05. No AdaLN residual gate is introduced into this existing ViT; identity is the zero adapter and zero initial classifier. The adapter is parallel to the MLP input projection; normalized h drives both the branch and condition. Activation mixing avoids per-token generated factors but costs O(B*T*s*r) activations and more token-wise GEMM work. Calibration matches only early low-rank function updates; diagonal and full-network optimization may differ. 500-step pilots use a prefix of the 3520-step schedule; quality uses a complete ten-epoch schedule. This is a short trend study, not an asymptotic quality comparison; concurrent throughput is not an isolated speed benchmark. Eight additional rates are frozen after observing coarse-grid updates; no quality-based selection or relaxed bounds. The original two preflights and 28 pilots are reused only after full audit and paired source/runtime/initialization/stream checks.

[Comparison and diagnostics](vit-token-refinement.json) · [Evidence](vit-token-refinement-evidence.json) · [Audit](vit-token-refinement-audit.json) · [Protocol](vit-token-refinement-protocol.json) · [Calibration](vit-token-refinement-calibration.json)
