# Completed conditional structured-operator screen

Hadamard mixing improves on its nearly parameter-matched diagonal control at the higher adapter LR, but costs training throughput. Block structure offers a different quality/throughput/memory tradeoff against square low-rank. Neither new operator displaces the rectangular DPLR quality reference. These are first two-seed architecture results, not a new default or an established optimal learning rate.

| Design | Adapter parameters | Adapter LR1: CFG2 FID | Adapter LR2: CFG2 FID |
|---|---:|---:|---:|
| adaln | 0 | 39.43 ± 0.29 | — |
| rectangular_lowrank | 2,834,944 | 33.93 ± 0.10 | 33.59 ± 0.33 |
| square_lowrank | 1,212,928 | 35.20 ± 0.22 | 43.36 ± 11.01 |
| square_diagonal | 396,032 | 37.77 ± 0.74 | 38.38 ± 0.02 |
| square_block16 | 1,147,136 | 36.28 ± 0.33 | 35.63 ± 1.32 |
| square_hadamard | 400,128 | 37.83 ± 0.26 | 36.62 ± 1.15 |

Final30k EMA, CFG2/DDIM50,5k real/5k generated images. Means ± sample SD over seeds128/129, not confidence intervals. Plain AdaLN has no adapter LR and runs once per seed. Every rate and endpoint remains in the evidence.

## Same-placement comparisons

At LR2, Hadamard-minus-diagonal guided FID is−0.97/−2.56, mean−1.76. Both seeds improve. Class agreement is83.28% versus80.66%, improving2.06/3.18 percentage points. These controls have400,128/396,032 adapter parameters and identical initial96-output coefficient heads. At LR1, FID gaps are−0.64/+0.78, mean+0.07, while class agreement is82.15% versus79.93%. The FID advantage therefore depends on recipe; it is not a uniform win across rates. Hadamard's three generated diagonals also change condition dependence, so the result cannot isolate fixed mixing alone.

Block16 at LR1 has FID36.28 versus35.20 for square low-rank, worse by1.15/0.99 in the paired seeds. At LR2, block reaches35.63 versus43.36, but this contrast includes one poor square-low-rank endpoint. At the different observed operating points block2 versus square-low-rank1, FID gaps are+1.20/−0.35, mean+0.43; this is mixed rather than a consistent quality advantage. Their adapter budgets are1.15M/1.21M, generated values per sample and leading application matmul work match, but generator heads/layout, gains and initial Jacobians differ.

Square low-rank LR2 seed128 ends at FID51.15 and76.46% class agreement, versus35.58/86.32% for seed129. EMA noise MSE is0.10926 at5k and ends0.06007 for the affected seed, compared with approximately0.058 for the good recipes. It completed with finite evidence and is retained in the aggregate. The study does not isolate a gauge or magnitude cause. Preflight success and early gradient wake-up do not establish long-run optimization stability.

## Quality and resource tradeoffs

The established rectangular reference reaches33.93 at1 and33.59 at2; both are below all new square-design means. This comparison also changes placement, free output-factor directions and parameter count. Square low-rank1 uses57.2% fewer adapter parameters than rectangular DPLR and has35.20 FID. Hadamard2 uses85.9% fewer adapter parameters and has36.62 FID. Against plain AdaLN39.43, Hadamard2 improves FID in both seeds by1.79/3.82, but neither equal total capacity nor a static structured-operator control is present. Those gaps alone cannot identify the contribution of conditioning.

| Design/rate | Training images/s | Peak allocated GiB | CFG2 agreement |
|---|---:|---:|---:|
| adaln_a1 | 4003 | 1.270 | 78.66% |
| rectangular_lowrank_a1 | 2924 | 1.168 | 88.60% |
| rectangular_lowrank_a2 | 2907 | 1.168 | 88.48% |
| square_lowrank_a1 | 2898 | 1.119 | 85.91% |
| square_diagonal_a2 | 3337 | 1.280 | 80.66% |
| square_block16_a2 | 3154 | 1.332 | 84.30% |
| square_hadamard_a2 | 2599 | 1.129 | 83.28% |

Hadamard2 throughput is22.1% lower than diagonal2 (2599 versus3337 images/s), with lower recorded peak allocation1.129 versus1.280GiB. This PyTorch/compile implementation has no isolated speed advantage. Block2 throughput is8.8% higher than square-low-rank1 (3154 versus2898), but peak allocation is19.0% higher (1.332 versus1.119GiB). Whole-training allocation includes model/optimizer/EMA/activation execution effects; it is not a direct measure of generated-operator storage. These are concurrent observations, not isolated speed or activation-memory benchmarks. All per-rate cost distributions remain in JSON.

## Every final guided endpoint

| Design/rate | FID seed128 | FID seed129 |
|---|---:|---:|
| adaln_a1 | 39.224 | 39.629 |
| rectangular_lowrank_a1 | 34.001 | 33.861 |
| rectangular_lowrank_a2 | 33.821 | 33.358 |
| square_lowrank_a1 | 35.361 | 35.047 |
| square_lowrank_a2 | 51.145 | 35.579 |
| square_diagonal_a1 | 38.295 | 37.243 |
| square_diagonal_a2 | 38.397 | 38.371 |
| square_block16_a1 | 36.511 | 36.040 |
| square_block16_a2 | 36.565 | 34.698 |
| square_hadamard_a1 | 37.650 | 38.018 |
| square_hadamard_a2 | 37.430 | 35.808 |

## Evidence and limits

All28 workers completed: six compiled preflights plus22 full30k runs. All nine final audits passed in the controller and independent frozen-runner replay, covering43 frozen files,33 unchanged reference sources, all configurations, actual optimizer ownership/rates, paired initialization/buffers/data/noise/dropout, gradient wake-up, guidance/evaluator budgets, successful exits, claims and storage. The workflow ran October16,20:39 to October17,01:38 UTC, about4h59m. About1.51GiB is retained within the4GiB bound. Private supervisors are closed, adalora GPU claims released. No weights/checkpoints, official test or evaluator downloads.

Both training seeds use the same reused research holdout, evaluators and sampling noise. Generator/output-operator parameterization, conditional polynomial degree and initialization/gain conventions differ; rates1/2 are not calibrated function-update matches or established optima. The square operators use one shared base projection and therefore constrain output directions relative to the rectangular reference. Diagnostic FP32 function probes for square methods include shared MLP-weight motion; BF16 projection rounding is logged separately. These limits prevent a general claim that full-rank structured operators beat low-rank or diagonal conditioning.

The first operator-space comparison gives useful measured tradeoffs and a recipe-specific Hadamard-over-diagonal signal. Keep the rectangular model as the quality reference and retain square low-rank, block and Hadamard as different cost/quality candidates. No automatic promotion, extra LR grid, FFT/TT implementation, new task or GPU run was launched during this review.

[Comparison](dit-structured.json) · [Evidence](dit-structured-evidence.json) · [Audit](dit-structured-audit.json) · [Protocol](dit-structured-protocol.json) · [Original plan](dit-structured-plan.md)
