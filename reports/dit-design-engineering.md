# DiT adapter design engineering

500 steps per case, seed 42, one sequential claimed GPU; 4,096 cached train images sampled with replacement. This is an engineering screen, not a quality ranking.

| Case | Extra parameters | Peak training MiB | Mean block delta/base RMS | Train images/s |
| --- | ---: | ---: | ---: | ---: |
| adaln | 0 | 560.6 | 0.000 | 4,574 |
| bottleneck_r2 | 13,836 | 582.3 | 0.068 | 4,263 |
| bottleneck_r4 | 27,672 | 582.8 | 0.084 | 4,068 |
| basis_r2_normalized | 221,376 | 587.4 | 0.321 | 4,124 |
| basis_r2_modulated | 221,376 | 587.4 | 0.304 | 4,636 |
| basis_r4_normalized | 405,696 | 592.4 | 0.476 | 3,788 |
| basis_r4_modulated | 405,696 | 592.4 | 0.447 | 4,165 |
| dplr_r8_reference | 774,336 | 603.4 | 0.690 | 4,243 |

Peak training allocation excludes diagnostic forwards and EMA. The gain column is the arithmetic mean of each block's fixed-image delta/base RMS ratio at the final step, not an operator-norm bound. Training steps include cached CPU preparation and are individually synchronized.

| Case | Broadcast saved MiB | Token saved MiB | Broadcast eager / compiled ms | Token eager / compiled ms |
| --- | ---: | ---: | ---: | ---: |
| adaln | 15.95 | 27.77 | 1.961 / 1.022 | 1.393 / 0.990 |
| bottleneck_r2 | 19.07 | 33.86 | 3.474 / 3.182 | 3.161 / 1.956 |
| bottleneck_r4 | 19.13 | 33.96 | 3.499 / 2.549 | 3.437 / 2.923 |
| basis_r2_normalized | 19.58 | 35.40 | 2.653 / 2.255 | 4.223 / 2.461 |
| basis_r2_modulated | 19.58 | 35.40 | 4.310 / 3.107 | 4.012 / 1.486 |
| basis_r4_normalized | 20.14 | 36.52 | 2.457 / 2.537 | 2.833 / 2.920 |
| basis_r4_modulated | 20.14 | 36.52 | 4.333 / 2.999 | 4.033 / 3.284 |
| dplr_r8_reference | 21.25 | 38.76 | 2.088 / 2.210 | 2.471 / 1.462 |

Projection timings are host-wall-clock medians of repeated complete forward/backward loops; repeat values and first-call latency are in the JSON. Compiler disk caches can be reused, so first-call latency is not necessarily a cold-cache build. Other projects share host resources. These values do not establish a stable speed ordering or end-to-end compiled training gain.

Saved MiB count unique eager nonparameter backing storage, including inputs and autocast weight copies. They describe the first-block projection with the common six-output modulation calculation, not adapter-only activation storage or compiled peak memory. Token probes are synthetic extensions of trained broadcast weights, not trained self-conditioned models.

The bottleneck reduces adapter parameters by about 93–94% relative to matched-rank basis mixing. Its output gain is substantially smaller under the shared LR/initialization recipe. Gain and functional-update controls are needed before attributing quality differences to capacity. No quality winner is selected from the short held-out noise-MSE diagnostic.

All cases preserve zero initial deltas and open their adapter gradients. The audit verifies shared streams, paired initialization, frozen source, budgets, finite evidence, and compiled numerical checks. No model weights/checkpoints or official test evaluation.

[Full report](dit-design-engineering.json) · [Audit](dit-design-engineering-audit.json) · [Design notes](../DESIGN_NOTES.md)
