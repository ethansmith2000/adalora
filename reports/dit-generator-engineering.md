# DiT generator widths and conditional cores

500 steps/case, seed 42; 4,096 cached train images, batch 128, width 192, depth 6. One sequential claimed GPU. This screen measures learning and engineering costs, not quality rankings.

All adapters use normalized input, common AdaLN-Zero and a fixed 1/sqrt(outer rank) gain except the explicit raw basis control. Adapter LR multiplier is 1; these new recipes are not function-update calibrated. Warmup 50, base LR 0.0004, cosine horizon 30,000 steps. Raw validation noise MSE is a short-run learning diagnostic; no EMA/FID/class-agreement evaluation.

| Case | Total parameters | Added versus AdaLN | Peak training MiB | Train images/s | Final mean delta/base RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| adaln | 4,179,696 | 0 | 560.56 | 5,608 | 0.000 |
| ffn_r2_budget | 4,401,456 | 221,760 | 591.10 | 4,461 | 0.000 |
| basis_r2_s16 | 4,401,072 | 221,376 | 587.38 | 4,317 | 0.275 |
| basis_r8_s16 | 4,954,032 | 774,336 | 603.44 | 4,135 | 0.574 |
| basis_r8_s16_raw | 4,954,032 | 774,336 | 603.44 | 4,353 | 0.662 |
| basis_r8_q4 | 6,502,704 | 2,323,008 | 628.06 | 4,082 | 0.839 |
| basis_r8_q16 | 4,760,448 | 580,752 | 600.78 | 4,151 | 0.596 |
| basis_r8_q32 | 4,470,072 | 290,376 | 596.80 | 3,988 | 0.417 |
| basis_r16_q32 | 4,746,552 | 566,856 | 609.58 | 4,082 | 0.585 |
| bottleneck_r8 | 4,235,040 | 55,344 | 583.83 | 4,085 | 0.113 |
| core_r4 | 4,221,264 | 41,568 | 583.03 | 4,130 | 0.153 |
| core_r8 | 4,299,888 | 120,192 | 584.78 | 4,260 | 0.282 |
| core_r16 | 4,568,304 | 388,608 | 589.93 | 3,979 | 0.402 |
| dplr_core_r8_p2 | 4,272,096 | 92,400 | 585.03 | 4,063 | 0.134 |

The FFN control adds capacity through MLP hidden width, so its adapter parameter count is zero. At width 192/depth 6 its total parameter count exceeds basis r2 by only 384. Equal-shape variants have identical initial base hashes. The expanded FFN is seed-matched but changes initialization through different shapes; it does not have an identical base hash.

Peak training allocation includes model, gradients, AdamW, training activations and fixed diagnostic inputs, and excludes EMA/diagnostic forwards. Timing includes synchronized cached CPU preparation, H2D, noise generation, forward/backward and optimizer; diagnostics and validation are outside the timer.

| Case | Broadcast saved MiB | Token saved MiB | Broadcast eager / compiled ms | Token eager / compiled ms |
| --- | ---: | ---: | ---: | ---: |
| adaln | 15.95 | 27.77 | 2.214 / 1.949 | 2.193 / 1.875 |
| ffn_r2_budget | 15.99 | 27.80 | 2.119 / 2.354 | 2.356 / 2.193 |
| basis_r2_s16 | 19.58 | 35.40 | 2.128 / 1.588 | 2.004 / 2.898 |
| basis_r8_s16 | 21.25 | 38.76 | 4.541 / 3.224 | 4.255 / 3.219 |
| basis_r8_s16_raw | 21.25 | 38.76 | 4.367 / 2.189 | 4.246 / 1.817 |
| basis_r8_q4 | 21.76 | 48.25 | 4.734 / 2.600 | 2.840 / 3.227 |
| basis_r8_q16 | 21.19 | 37.58 | 4.534 / 1.528 | 1.997 / 3.098 |
| basis_r8_q32 | 21.10 | 35.80 | 4.394 / 3.424 | 4.201 / 2.238 |
| basis_r16_q32 | 23.18 | 37.63 | 4.468 / 3.453 | 4.544 / 3.120 |
| bottleneck_r8 | 19.27 | 34.16 | 3.560 / 2.930 | 3.521 / 3.062 |
| core_r4 | 19.14 | 34.15 | 3.137 / 2.803 | 3.511 / 2.576 |
| core_r8 | 19.30 | 35.05 | 3.738 / 1.968 | 4.029 / 3.087 |
| core_r16 | 19.69 | 38.39 | 3.729 / 2.278 | 3.942 / 1.931 |
| dplr_core_r8_p2 | 19.32 | 34.70 | 5.108 / 2.541 | 5.578 / 2.782 |

Projection timings measure the trained first-block pre-activation path, including non-affine normalization, common modulation and base projection, with complete forward/backward but no optimizer. Token conditions are synthetic extensions of broadcast-trained weights, not a token-conditioned quality experiment. Saved storage counts unique eager nonparameter backing storage including inputs and autocast copies; it is not a whole-model peak or compiled activation measurement.

Repeat timings, first-call latency and compiled output/gradient parity are retained in JSON. Compilation may reuse disk caches. Other projects share host resources even though this worker has one exclusively claimed GPU. A single order/seed and short training window do not establish a stable wall-clock winner or equal-time quality result.

| Case | Final mean fixed-input adapter-delta update RMS | Raw validation noise MSE |
| --- | ---: | ---: |
| adaln | 0 | 0.081712 |
| ffn_r2_budget | 0 | 0.081604 |
| basis_r2_s16 | 0.0002629 | 0.080563 |
| basis_r8_s16 | 0.000346258 | 0.080404 |
| basis_r8_s16_raw | 0.000807702 | 0.079920 |
| basis_r8_q4 | 0.000908873 | 0.079935 |
| basis_r8_q16 | 0.000290775 | 0.079572 |
| basis_r8_q32 | 0.000133556 | 0.080533 |
| basis_r16_q32 | 0.000153059 | 0.080551 |
| bottleneck_r8 | 0.000222268 | 0.081552 |
| core_r4 | 0.000696241 | 0.081647 |
| core_r8 | 0.000686084 | 0.081432 |
| core_r16 | 0.000765949 | 0.081864 |
| dplr_core_r8_p2 | 0.000619432 | 0.081922 |

The update column measures only the adapter delta, not the entire pre-activation or model. Zeros for AdaLN/FFN indicate an absent adapter; their base parameters still learn.

Zero initial deltas and adapter gradient wake-up are audited. Distinct core/generator parameterizations have different gauges and initial optimization geometry. Scalar gains do not ensure comparable function updates or variance preservation. Compare these diagnostics before calibrating a longer quality study.

No model weights, resume checkpoints, generated image datasets or official test evaluation. The evidence audit checks frozen source/configs, parameter/FLOP budgets, shared batches/noise, equal-shape initialization, numerical parity and successful completion.

[Full report](dit-generator-engineering.json) · [Audit](dit-generator-engineering-audit.json) · [Design notes](../DESIGN_NOTES.md)
