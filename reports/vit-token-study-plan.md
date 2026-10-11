# ViT token self-conditioning study

Proceed with factor offsets enabled as the user-selected forward research default.
Preserve explicit off ablations and historical frozen protocols. The prior DiT
offset result was positive in both seeds, but small; this choice does not claim
significance or a universal improvement.

Use normalized tokens as both adapter input and condition at the first MLP
projection. Preserve ordinary self-attention and static LayerNorm affine settings
in every control. Detaching the condition blocks gradients through both its
scale/shift and factor coefficient generators while preserving forward values.
The existing ViT has no AdaLN residual gate; adapters start at zero and the
classifier remains zero-initialized.

| Family | Parameters | Role |
|---|---:|---|
| Plain | 2,690,122 | Original ViT |
| Diagonal | 3,134,794 | Token-generated scale and shift |
| Static rank8 + diagonal | 3,180,874 | Static factor control |
| Larger static MLP | 4,341,772 | Within606 parameters of dynamic s24 |
| DPLR s12 | 3,761,626 | d/16 generator-width trend |
| DPLR s24 | 4,342,378 | d/8 reference |
| DPLR s48 | 5,503,882 | d/4 generator-width trend |
| Detached DPLR s24 | 4,342,378 | Same forward, detached conditioning |

All dynamic families use transform rank8, gain1/sqrt(8), affine coefficient
heads and zero-initialized U/V factor offsets. Offsets add46,080 parameters
across six blocks and use zero weight decay as biases. Learned bases use0.05
decay. Activation mixing uses `[B,T,s,r]` intermediates and rank-sized offset
contractions, without materializing token-sized full factors.

Two12-step compiled BF16 GPU preflights cover s48 and the large static MLP,
including128/128/72 ragged batches and the complete5k holdout.

Twenty-eight500-step calibration pilots use separate paired seeds96/97.
Each retains the3520-step training horizon and5% warmup. Compare s12, s48,
detached s24 and static rank8+diagonal to s24 at unit adapter LR. Candidates
are0.25/1/4; static rank8 also tries16. Require every ratio at100/250/500
in both seeds to stay within0.5–2, then choose the closest geometric update
magnitude. These are fixed-token low-rank interval updates; diagonal or full
network updates are not matched. Quality metrics do not select the rates.
No unbounded search: failure withholds the quality stage.

If the gates pass, run all eight families on fresh seeds98/99 for ten epochs,
3520 steps each, with frozen selected rates. Common settings: CIFAR-10 train
split45k/validation5k, split seed123; width192/depth6/heads3/patch4; hidden768
except the matched MLP; batch128; AdamW(.9,.999), base LR5e-4, cosine floor0.01,
clip1, label smoothing0.1; fused optimizer and compiled BF16. Use per-epoch
validation and raw-model diagnostics. The two-seed screen gives trends rather
than confidence intervals or asymptotic quality evidence.

Before launch, check shared disk capacity (minimum8GiB). Budget at most two
GPUs across all stages,3–6 aggregate GPU-hours, and4GiB retained artifacts.
Retain configs, frozen sources, runtime/seed/split records, metrics, diagnostics,
logs, audits/reports and compiler caches (expected≤3GiB) until explicitly removed.
No model weights, periodic/recovery checkpoints, test evaluation, sample grids
or new evaluator weights. A one-layer FP32 CPU saved-storage probe accompanies
whole-model CUDA allocation; it does not measure compiled activation storage.
Concurrent throughput is recorded for audit, without an isolated speed claim.

Launched in `runs/vit-token-20261011T070503Z`; both GPU preflight audits passed
and calibration is running with two workers. Forty-one prelaunch CPU checks
passed; seven later protocol regression checks passed. A row-assembly bug was
fixed in a separately frozen recovery controller, with no repeated preflight
or altered training source. A preparation-only CPU probe briefly initialized
CUDA contexts on all devices; it did no model computation there and its process
exited. The helper now explicitly specifies no CUDA devices. Both exceptions
are preserved in [launch validation](../runs/vit-token-20261011T070503Z/launch_validation.json).

Rate multipliers apply to all existing conditional parameters, including
scale/shift. Matching low-rank function updates does not match diagonal updates.
The static-capacity control has different base shapes/initialization; exact
base pairing applies to the common hidden768 controls, while observed training
augmentation streams are paired across all families.

[Live state and protocol links](vit-token-study-plan.json).
