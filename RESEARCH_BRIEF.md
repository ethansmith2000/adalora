# Condition-Generated Low-Rank Transforms — Research Brief

*Oct 6, 2026 · companion code: `condlr.py` (reference implementations + self-tests; `python condlr.py`)*

**2026-10-07 design update:** [DESIGN_NOTES.md](DESIGN_NOTES.md) is the PyTorch/tensor
notation companion for the current implementation and subsequent discussion.
It separates generator factorization from generated-transform structure and
evaluation order; records normalized/modulated/static-affine adapter inputs,
additive versus composed placement, zero-init/gain/gauge concerns, and proposed
optimization controls. A bounded engineering suite now implements normalized/
modulated adapter inputs and the simpler conditional bottleneck; other proposed
alternatives remain unimplemented. Completed quality confirmations are recorded
below and in the linked reports; they do not test the broader proposed variants.
The diffusion model retains common non-affine LayerNorm and AdaLN-Zero heads,
adding a rectangular low-rank delta at MLP input from the unmodulated normalized
token. It uses `r=8,K=16` with factors generated once per image. Earlier ViT runs
used per-token self-conditioning and retained static LayerNorm affine parameters.
Completed results and their limits are in [README.md](README.md).
The [completed gain/update screen](reports/dit-gain-engineering.md) calibrates
fixed `1/sqrt(rank)` gains and adapter rates without quality-based selection.
Comparable fixed-input function updates still leave much smaller live outputs
for conditional bottlenecks than basis mixing; branch magnitude and optimizer
movement need separate diagnostics. Probes use full FP32 multiplication,
independently of BF16 training. This is one-seed engineering evidence, not a
variance-preserving scaling rule or quality confirmation.
The subsequent [fresh-seed quality screen](reports/dit-gain-quality.md) tests
five calibrated recipes and four controls for 5,000 steps on full training data.
It finds a FID/class-agreement tradeoff and imperfect transfer of the subset
update calibration; the smaller rank-2 variants warrant longer paired-seed
checks. This short screen does not establish an architectural winner or address
sequence-valued conditioning.
The supplied original setup conversation is reconciled in the design notes:
each factor contraction is bilinear, but the complete self-conditioned delta
with two bias-free linear factor generators is cubic in the normalized token.
The original CPU memory table is historical; compilation/fusion and hardware
speed assertions are engineering hypotheses rather than established guarantees.
Use PyTorch/tensor notation for new explanations; the original notation below
is retained as the historical brief.

**2026-10-08 generator/gauge clarification:** The detailed tensor explanations
and budget ledger are in [the basis-generator notes](DESIGN_NOTES.md#what-basis-mixing-factorizes)
and [the generator alternatives](DESIGN_NOTES.md#generator-alternatives-and-diagonal-structure-2026-10-08).
The current linear basis generator is exactly two factorized affine projections:

```python
# c: [B,q], h: [B,T,d]; s=K per factor generator.
a = c @ A_U + bias_U                # [B,s], signed coefficients
b = c @ A_V + bias_V                # [B,s]
U = (a @ D_U).reshape(B,d,r)         # D_U: [s,d*r], learned basis decoder
V = (b @ D_V).reshape(B,r,o)         # D_V: [s,r*o], learned basis decoder
delta = torch.bmm(torch.bmm(h, U), V)
```

Each decoder row is one basis matrix after reshaping. These are ordinary
learned projection weights shared across samples, not free extra objects or
condition-specific tensors. `r` bounds the update rank; `s/K` limits each
linear generator's projection rank. Separate heads cost
`2*q*s + s*r*(d+o)` before biases. A shared latent head costs
`q*s + s*r*(d+o)` but couples the condition features used by U and V.

Candidate generator widths are near `q/4,q/16,q/32`: 48/12/6 at the current
condition width 192, and 256/64/32 at width 1024. Current `K=16` is per side.
The rank-8 decoder bases contain 122,880 of the current 129,056 adapter
parameters/block, so compressing only the small coefficient head saves little.
Reducing s can fund larger r, with narrower condition dependence and increased
broadcast factor storage/application work. For example, `r=16,s=6` costs
94,476 parameters/block versus 129,056 at `r=8,s=16`; these budgets now have
engineering screens and the separate 30k gain/LR confirmation recorded below.

Learned generators have coordinate gauges: `a @ Q` paired with
`solve(Q,D_U)` preserves U, and `U @ S` paired with `solve(S,V)` preserves
the update. These do not change the forward function but can change optimizer
steps, weight decay and factor balance. Parallel placement does not remove
them; orthogonal initialization does not permanently fix them. Track function
updates and branch magnitudes alongside factor/encoder/decoder norms. Keep
zero-delta initialization compatible with any normalization or constraints.

Diagonal plus low-rank is a useful candidate for square replacement maps.
The rectangular condition-to-flattened-factor projection needs an explicit
coordinate mapping or structured skip. A linear square diagonal encoder
followed by compression to s still has projection rank at most s. A particularly
natural alternative is a square conditional core between static factors:
`G(c) = diag(g(c)) + L(c) @ R(c)`, generating `r+2*r*p` values when the core
factors have rank p. Channel-modulated static factors and further CP/Tucker or
structured decoder factorizations are other unimplemented cost comparisons.
Nonlinear generators can enrich condition dependence at similar parameter
counts, but do not automatically improve memory or speed.

Generator compression does not by itself reduce the number of explicitly
generated factor values. With per-token conditioning, activation mixing
avoids full factors using intermediates shaped `[B*T,s,r]` and has comparable
leading matmul work to generating factors per token. With broadcast conditioning,
weight mixing generates once per sample and amortizes across tokens. Both
parameterization and execution must be declared; actual peak memory and speed
need measurement. This documentation update adds no implementations or runs.

**Subsequent 2026-10-08 implementation:** Dense `conditional_core` and
`conditional_dplr_core` are now available in the diffusion trainer. The latter
uses `--core-rank` for its inner factorization while `--rank` controls the
outer transform. [The engineering workflow](DESIGN_NOTES.md#generatorcore-engineering-workflow-2026-10-08)
compares them with fractional basis-generator widths, r16/s6 and the FFN
budget control over 14 cases. All 74 targeted tests and a complete CPU smoke
passed before launching the real 500-step screen. Quality confirmation is a
separate subsequent study described below; CP/Tucker decoder compression,
shared/nonlinear factor heads, channel modulation and separate factor LRs
remain unimplemented or untested as indicated in the notes.

The [completed generator/core screen](reports/dit-generator-engineering.md)
passed all 14 cases and its audit. Dense/DPLR cores at r8 save 84–88% of adapter
parameters versus basis r8, with only about 3% lower whole-training peak memory.
They have smaller live deltas but larger final fixed-input function updates
under the shared gain/LR recipe; calibration remains a separate requirement.
The fractional generator widths and r16/s6 trade show measurable parameter,
token-storage and broadcast-memory differences. Compiled BF16 parity passed,
but variable timing and short noise-MSE diagnostics establish no quality or
stable wall-clock winner. No weights/checkpoints or official test evaluation.

The subsequent [full-loader core/FFN protocol](DESIGN_NOTES.md#full-loader-core-calibration-and-fresh-seed-quality-confirmation)
calibrates on seeds 53/54 and confirms seven families on fresh seeds 55/56/57
for 30k steps. Dense/DPLR cores matched aggregate update tolerance after one
LR correction (multipliers 0.414517/0.472845); rates are frozen independently
of quality. All 21 confirmation runs completed and the independent audit passed.
The FFN capacity control is within 384 parameters of basis r2 but changes
initialization through different shapes. Compact evidence only; no checkpoints.

The [completed core/FFN comparison](reports/dit-core-confirmation.md) finds
basis-r2 mean FID 65.33 versus FFN 67.06, and class agreement 45.35% versus
40.96%. FID favors basis r2 in two of three paired seeds; agreement favors it
in all three. Dense/DPLR cores save 84–88% of adapter parameters versus basis
r8, but achieve FID 68.21/68.34, versus scaled/raw-r8 61.43/62.23, and lose on
both quality metrics to basis r2 and rank 8 in every paired seed. Whole-trainer
memory saves about 3.4%, including EMA. Core update ratios drift from near 1
early to about 0.43/0.45 of scaled r8 at the endpoint, with much smaller live
deltas; calibration does not ensure sustained functional or magnitude matching.

Scaled r8 has lower FID than raw r8 in all three new seeds, reversing the prior
cohort's FID ordering; raw r8 has slightly higher mean class agreement. Gain
choice remains recipe/seed-dependent. A useful next quality comparison is
fractional basis-generator widths (`s=6/12` at `q=192`) and `r=16,s=6`, which
retain condition-dependent outer factors. Restriction to static outer subspaces
is one possible reason for core degradation; optimization drift is another,
and this study does not isolate them. Isolated timing and sequence-valued
conditioning remain open. The report includes paired differences, sample SD,
update trajectories, source/evaluator audits and artifact hashes.

**Roadmap refinement:** The user has deferred dedicated isolated speed work.
The next generator search should establish joint rank/width trends, including
approximate decoder-budget comparisons, before longer fresh-seed confirmation.
Token self-conditioning is an explicit parallel application: use each
normalized contextual token as its condition, first in ViT and then a small
causal LM. Richer generators, external sequence-conditioning retrieval and
structured produced operators remain on the roadmap. The implementation
status, proposed grid, controls and memory distinction are recorded in
[the revised roadmap](DESIGN_NOTES.md#revised-roadmap-parameter-trends-and-token-self-conditioning).

**Completed rank/width and token-conditioning screen:**
[The 64-run two-seed report](reports/rank-width-search.md) crosses transform
rank and generator width on broadcast-conditioned diffusion, and renews the
ViT token pilot. At a similar decoder budget, ranks 2/4/8/16 give mean FID
141.28/132.10/128.78/127.27; class agreement favors middle ranks. Width beyond
16 at rank 8 gives no steady FID improvement. Generator width changes early
function-update scale substantially under a common LR and rank gain, so the
search does not isolate capacity from optimization. In ViT, rank-8/s16 gains
only 0.44 percentage points over diagonal conditioning for another 774,336
parameters and 92.5 MiB training allocation. Allowing condition gradients at
rank-8/s12 gains 2.36 points over detaching them. These short results motivate
paired rank/width and gain/LR sensitivity checks, plus full/detached/diagonal
controls for token conditioning; longer confirmation remains necessary.
No new checkpoints, isolated speed studies, LM or external-sequence jobs
were launched. [The completed notes](DESIGN_NOTES.md#completed-screen-trends-magnitude-and-next-comparisons)
record the protocol, optimization caveats, recovery and next comparisons.

**Completed narrower-generator gain/LR confirmation:**
[The audited 30k study](reports/dit-width-calibration.md) uses independent
update-only calibration seeds 60/61, then seven recipes on fresh paired seeds
62/63/64. All 18 calibration and 21 quality runs completed. r8/s12 with
calibrated LR gives FID 61.68 ± 0.35 versus r8/s16's 62.48 ± 0.55, with
25% fewer adapter parameters (3.91% fewer total parameters), but agreement
falls from 50.43% to 49.77%. r16/s6 with matched LR gives 61.38 ± 2.38 FID
and 47.52% agreement; the large FID spread and lower agreement in every seed
prevent treating it as a general improvement. Its smaller parameter count
also does not lower whole-trainer peak allocation.

The single alternate gain, `(16/s)/sqrt(r)`, worsens mean FID and agreement
relative to matched LR alone for both shapes. It nevertheless changes
optimization substantially: r16/s6's fresh early/final function-update ratios
are 0.990/0.981 versus 0.847/2.100 for LR matching alone. Final live delta/base
is similar, about 0.79–0.80. Early scalar calibration, later optimizer movement,
branch magnitude and EMA quality cannot substitute for each other. These
observations do not prove a gauge mechanism, variance preservation or μP scaling.
Keep r8/s16 and the affordable r8/s12 candidate for future controls; token
self-conditioning requires its own diagonal/full/detached comparison and
calibration. Dedicated speed work remains deferred. The
[completed design notes](DESIGN_NOTES.md#completed-gainlr-confirmation-quality-and-drift)
record exact rates, paired differences, resource use and the remaining roadmap.
All private supervisors are closed; no weights/checkpoints were saved.

**Subsequent projector-design discussion (proposed):** Share a larger low-rank
factor generator across all layers or two contiguous depth groups, or share
only its nonlinear condition trunk/decoder bases while retaining layer-specific
heads. At the current six-layer q=d192/o768/r8 dimensions, six independent
affine s16 projectors, two shared s48 projectors and one shared s96 projector
each have 774,336 adapter parameters. Two shared s16 projectors cost 258,112
and provide a fixed-width tying control. Identical broadcast conditions produce
identical transforms within a whole-projector group; nonlinear layer embeddings
or independent heads can restore some conditional specialization.

For broadcast class/time, generated factors can be reused within a group.
At equal affine budget, G*s is fixed and leading total generation FLOPs can
remain similar. Token self-conditioning changes c at each layer, so parameter
sharing does not imply factor reuse or lower activation storage. Wider s
increases per-token activation-mixing intermediates. Shared gradients require
new calibration and per-layer diagnostics; sharing does not remove gauges.

Direct SiLU coefficient heads add no parameters and alter condition dependence
without enlarging the decoder span. A small nonlinear MLP or a larger shared
trunk is another affordable generator alternative because decoder weights
dominate the current budget. Coefficient, factor and complete-transform
normalization constrain different magnitudes and preserve different symmetries;
none automatically ensures variance preservation. The notes include a rank-sized
Gram calculation of transform Frobenius norm and a smooth zero-compatible cap.
The proposed first comparison is the three equal-budget sharing arrangements
plus the fixed-width two-group control, followed by nonlinear heads and a
separate normalization ablation if warranted. The full-sharing/direct-SiLU
subset is now implemented in the subsequent screen; other variants remain
proposals. Dedicated speed work stays deferred. See
[the detailed tensor designs and revised roadmap](DESIGN_NOTES.md#proposed-wider-nonlinear-and-shared-projectors-2026-10-08).

**Shared/nonlinear implementation and screen:** The completed workflow
`runs/dit-projectors-20261008T205312Z` compares six recipes: independent
affine/SiLU s16, two-group affine/SiLU s48, fully shared affine s96 and a
two-group affine s16 control, all at r8. The first five match parameter counts.
Update-only calibration uses seeds65/66; selected rates freeze before fresh
seeds67/68 at5k steps of the30k schedule. Prelaunch checks pass78 targeted CPU
tests, exact parity with the frozen independent implementation and two claimed
GPU smokes of shared BF16 training/EMA/evaluator integration. Shared tensors are
counted once, factors reused within forward, and branch diagnostics remain
layer-specific. All 22 calibration and twelve quality runs passed the full audit.
No weights/checkpoints, output normalization, dedicated speed work or automatic
longer confirmation were enabled. The
[launch notes](DESIGN_NOTES.md#shared-and-nonlinear-projector-screen-implementation-and-launch)
record bounded selection, resource estimates and retention.

In this two-seed short screen, independent affine s16 has the best observed
mean FID (124.12 ± 3.34). Equal-budget two-group s48 affine (131.96 ± 6.33) and
fully shared s96 affine (136.49 ± 3.91) worsen FID and class agreement in both
paired seeds. Independent SiLU has no clear benefit; two-group SiLU improves
agreement but produces mixed FID with a large spread. Scalar early update
calibration does not guarantee fresh-seed or later matching: independent SiLU's
final update/reference rises to 2.839, while fully shared affine stays near0.990
despite worse quality. Shared bases with layer-specific heads, a shared trunk
with independent decoders, and small MLPs remain proposed next comparisons.
No longer jobs were launched. See the [report and figures](reports/dit-projector-screen.md)
and [interpretation and next comparisons](DESIGN_NOTES.md#completed-screen-quality-optimization-and-next-comparisons).


**Shared-basis/layer-head screen:** The completed workflow
`runs/dit-basis-heads-20261008T214617Z` keeps independent layer heads while
tying U/V bases in two groups or all layers. Wider s44/s77 recipes match total
parameters within0.1%; a fixed-width s16 tying control measures savings.
Independent hidden48 linear/SiLU MLP heads have exactly equal budgets and
initialization, adding0.56% whole-model parameters versus the affine reference.
Calibration seeds70/71 select bounded update-only rates before fresh paired
quality seeds72/73, six recipes at5k steps of the30k schedule. All stages use
at most two claimed GPUs, with no checkpoints or automatic longer confirmation.
CPU validation passed108 tests plus exact frozen-default parity; two GPU
compatibility checks passed. All26 calibration and twelve quality runs completed and passed the full audit. The [design and protocol notes](DESIGN_NOTES.md#shared-bases-with-layer-heads-and-small-mlps-two-gpu-screen)
record tensor shapes, ownership, near-matched budgets and artifact retention.


The completed screen favors independent bases over wider tied bases under this
recipe. SiLU MLP48 lowers FID versus its exactly matched linear MLP control in
both seeds (mean128.93 versus131.60), while lowering class agreement (27.18%
versus28.72%). Affine-reference FID is133.67, agreement27.96%. MLP calibration
passes on seeds70/71 do not transfer reliably to fresh seeds72/73: early update
ratios average0.805 for linear and0.632 for SiLU, with SiLU reaching2.663× at
step5k. These are short two-seed results and an FID/condition-use tradeoff,
not a default replacement. Retain the independent affine and both MLP controls;
strengthen update-transfer diagnostics and include actual learned class/time
condition probes before longer confirmation. All private workers are closed,
claims released and the two-GPU cap audited. See the [report](reports/dit-basis-head-screen.md)
and [findings and next comparisons](DESIGN_NOTES.md#completed-two-gpu-screen-findings-and-next-comparisons).

---

## 1. Summary

A condition vector `c` is projected into a low-rank transform `U(c) ∈ ℝ^{d_in×r}`, `V(c) ∈ ℝ^{r×d_out}`, applied as
`y = W x + x U(c) V(c)`. This generalizes AdaLN/FiLM (a condition-generated *diagonal*) to a condition-generated
*cross-channel* transform. Early results are promising; the blockers are **activation memory, parameter count,
and speed**, which currently pin the rank at r ≈ 2.

Central observations of this brief:

1. **Each factor contraction is a bilinear map** `z_k = Σ_ij x_i A_ijk c_j` with a 3-way tensor `A ∈ ℝ^{d_in×d_c×r}`.
   The complete two-factor delta is linear in x and quadratic in c for bias-free linear generators.
   "Make the projection low-rank" = choose a tensor decomposition of `A`. AdaLN, the current method, diagonal-plus-low-rank,
   and gated bottlenecks are all points in one family.
2. **Per-token conditions are the hard case.** The generated factors cost `r·(d_in+d_out)` floats per token per layer
   (4× the hidden state at d=1024, r=2), and autograd stores them. Gradient checkpointing does not fix this because it
   rematerializes the same full tensors within the layer.
3. **Two ways out:**
   - **Exact:** treat each contraction as a GEMM with a fused rank-1 prologue/epilogue, chunk over tokens (or fuse
     in Triton), and save only `x, c, z`. Memory problem solved; FLOP cost unchanged (≈4× a dense d×d linear at r=2).
   - **Factorized:** low-rank the hypernetwork (basis mixing / CP / Tucker). Activation mixing can avoid full
     generated factors; weight mixing reuses per-sample factors. Reduces generator parameters and FLOPs,
     potentially making r = 8–16 affordable, subject to generator width and execution choice.
4. **Diagonal + low-rank (DPLR)** is a natural target class: the diagonal does what AdaLN does well (per-channel gain),
   the low-rank part does what AdaLN cannot (cross-channel mixing).

Recommended order: exact chunked path at r=2 as the faithful baseline → basis-mixing at r=8–16 → DPLR → one
full-rank structured variant (Hadamard sandwich or block-diagonal).

---

## 2. Setup and notation

| Symbol | Meaning |
|---|---|
| `x ∈ ℝ^{d_in}` | token activation |
| `c ∈ ℝ^{d_c}` | condition; **per-sample** (broadcast over T) or **per-token** (`c_t`) |
| `W` | static base weight |
| `U(c), V(c)` | generated factors, rank `r` |
| `A ∈ ℝ^{d_in×d_c×r}`, `B ∈ ℝ^{d_c×r×d_out}` | the hypernetwork as tensors: `U(c)_ik = Σ_j A_ijk c_j`, `V(c)_ko = Σ_j B_jko c_j` |
| `K` | basis count per factor generator; bounds its linear projection rank; equivalent to separate-head bottleneck width `s` |

Nonlinear hypernetworks (`c → MLP → linear → U`) fit the same analysis: replace `c` with the MLP's output.

Code layout: `A` is stored as `(d_in, d_c·r)` with index `j·r + k`; `B` as `(d_c·r, d_out)`.

---

## 3. Cost model

Per layer, `d_in = d_out = d`.

| Variant | Extra params | Stored for backward / token | Fwd FLOPs / token (extra) |
|---|---|---|---|
| AdaLN (scale only) | `d_c·d` | `d` (if per-token) | `2·d_c·d` (per-token) |
| Full hypernet, naive (current) | `2·d_c·r·d` | **`2·r·d`** (U, V) + `r` | `≈ 4·d·d_c·r` |
| Full hypernet, exact chunked | `2·d_c·r·d` | `r` (plus x, c already held) | `≈ 4·d·d_c·r` (+1 recompute in bwd) |
| Basis mixing (K bases, rank r) | `2K·d_c + 2K·r·d` | `≈ 2K·r + 2K + r` | `≈ 4·d·K·r` |
| DPLR (basis-mixed low-rank + diag) | above + `d_c·d` | above + `d` | above + `2·d_c·d` |

Worked numbers at d = d_c = 1024:

- Full hypernet, r=2: **4.2M params**, fwd ≈ **8.4 MFLOP/token = 4× a d×d linear**.
- Basis mixing, r=8, K=16: **≈0.29M params**, fwd ≈ **0.5 MFLOP/token = 0.25× a d×d linear**, at 4× the rank.

**Historical CPU saved-activation figures from the original setup conversation**
(`saved_tensors_hooks`, parameters excluded; one layer, T=512, d=d_c=1024, fp32,
per-token c). These rounded figures are not whole-model GPU peaks or a new
measurement of the current source; see README for subsequent local profiles:

| Implementation | Saved for backward |
|---|---|
| `x` and `c` alone (unavoidable) | 4.0 MB |
| Naive full hypernet, r=2 | 20.0 MB |
| Naive full hypernet, r=8 | 68.0 MB |
| **Exact chunked**, r=2 or r=8 | **4.0 MB** |
| Basis mixing, r=8, K=16 | 4.6 MB |
| Basis mixing, r=16, K=16 | 5.1 MB |

The exact path does not save the full generated factors. Saved memory still includes
the small rank-r bottleneck, so it grows slightly with r; the rounded historical
table above hides that contribution. Its *transient* tile size is `chunk·d_c·r` (e.g. chunk=4096,
d_c=1024, r=2, bf16 → 16 MB), set by the chunk size.

**Takeaway:** with per-token c, the exact chunked path solves memory but not compute; the factorized path solves both.

---

## 4. The reframe: one bilinear family

`z_k = Σ_ij x_i A_ijk c_j` — every variant is a structure imposed on `A` (and on `B` for the output side).

### 4.1 Decompositions of the hypernetwork tensor

- **Full (current):** no structure; `d_in·d_c·r` parameters per side.
- **Basis mixing (rank-K projection):** `U(c) = Σ_k a_k(c) U_k`, with `a(c) = W_a c`. Equivalent to
  `W_u = U_basis · W_a`. This is the most direct reading of "the projection itself is low-rank".
- **CP, rank M:** `A_ijk = Σ_m P_im Q_jm R_km`, giving `z = Rᵀ((Pᵀx) ⊙ (Qᵀc))`.
  The layer becomes: project x → gate elementwise by a projection of c → mix. **AdaLN is the case P = I** (and no r bottleneck).
- **Tucker:** small core shared across the three modes; interpolates between CP and full.

Composing input and output sides gives `x → P → ⊙(Qc) → R(to r) → S → ⊙(Tc) → O`. The rank-r bottleneck sits
*between two gates*. Factorization can make that bottleneck cheaper to widen, but r remains a generator,
activation and application cost choice. Whether it should exist at all becomes an ablation (§8).

### 4.2 Where the known methods sit

| Method | Structure on the conditioned transform |
|---|---|
| FiLM / AdaLN | diagonal |
| This work (current) | rank-r, full hypernet |
| Basis mixing / CP | rank-r, low-rank hypernet |
| DPLR | diagonal + rank-r |
| Structured full-rank (§6) | block-diag / Kronecker / Hadamard-diagonal |

### 4.3 Per-sample vs per-token: which contraction order

- **Per-sample c, long sequences:** mix *weights* first. Build `U(c), V(c)` once per sample (`B·d·r`, small), then
  `bmm`. Saves a factor of K in token-side FLOPs. (`BasisMixLowRank(mix="weights")`)
- **Per-token c:** mix *activations*. Compute `x @ U_basis` as one static GEMM (can be concatenated onto `W`),
  combine with coefficients elementwise, then one static GEMM with `V_basis`. (`mix="activations"`)

Both are verified equal in the self-tests.

---

## 5. Implementation paths

### 5.1 Exact full hypernetwork (faithful baseline)

Each contraction is a dense GEMM plus a cheap rank-1 term:

| Pass | Computation | Shape note |
|---|---|---|
| in, fwd | `Y = x @ A_flat`; `z = Σ_j Y[:, j, :] · c[:, j]` | GEMM + reduction epilogue |
| out, fwd | `G = c ⊗ z`; `y = G @ B_flat` | outer-product prologue + GEMM |
| in, bwd | `G = c ⊗ dz`; `dx = G @ A_flatᵀ`; `dA = xᵀ @ G`; `dc = Σ_k (x@A_flat)[:, :, k] dz_k` | 2 prologue GEMMs + recompute GEMM |
| out, bwd | `G = c ⊗ z`; `dB = Gᵀ @ dy`; `H = dy @ B_flatᵀ`; `dz, dc` from H | 2 GEMMs + reductions |

The `(T, d_c·r)` intermediate is `U(c_t)` in disguise. `BilinearCondIn` / `BilinearCondOut` in `condlr.py`
implement this as custom `autograd.Function`s that loop over token chunks and save only `(x, c, A)` / `(z, c, B)`.
They pass `gradcheck` and match a materializing oracle exactly.

Shape choice matters: writing the input side as `(x ⊗ c) @ A'` gives K = d_in·d_c, N = r, which is a poor
tensor-core shape. The `x @ A_flat` form has N = d_c·r and K = d_in.

**Why checkpointing is not enough:** it rematerializes the full `(B·T, d·r)` factors inside the layer in both
forward and backward, so the per-layer peak barely changes. The chunked Function bounds it explicitly.

### 5.2 torch.compile vs Triton

The preference is PyTorch correctness first, then compile, then custom kernels
only when profiling justifies them. The original compiler and roofline claims
were predictions, not measurements or restrictions valid for every version.

- **Compile candidates:** static GEMMs, pointwise operations, and reductions in
  factorized variants and reusable exact-path chunk bodies. Test both passes;
  inspect graph breaks, recompilation, and actual generated execution.
- **Memory:** eager chunk scratch still reaches device memory. Chunking avoids
  whole-sequence generated factors and bounds tile dimensions; compilation and
  activation recomputation do not by themselves establish that explicit bound.
  Memory-budget settings are version-dependent private APIs; see DESIGN_NOTES.
- **Initial pattern:** keep the chunk loop eager and compile its reusable bodies.
  Fixed/padded tails or dynamic shape handling are candidates for limiting
  specialization. This is a starting strategy, not proof that other compilation
  scopes cannot work.
- **Timing:** favorable GEMM layouts do not establish a compute-bound workload
  or modest overhead. Measure complete backward, tile traffic, launches, peak
  allocation, and compile amortization on the target hardware.
- **Triton:** consider it if measured traffic, launch overhead, or reductions
  remain limiting. Fusion must explicitly handle tiling and accumulation of
  weight gradients across token tiles. A reference chunk is not automatically
  one Triton program, and SRAM-only transient execution remains an unimplemented
  kernel objective. Keep the exact PyTorch path as the numerical oracle.

See [PyTorch's checkpointing/compiler discussion](https://pytorch.org/blog/activation-checkpointing-techniques/)
and [compiler programming model](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.html).

### 5.3 Factorized

`BasisMixLowRank` and `DPLRCond` in `condlr.py`. Plain autograd + compile. Optional speed trick: concatenate
`U_basis` onto the base weight so `[Wx, x U_basis]` is one GEMM; similarly stack `V_basis` with the next GEMM's
K dimension.

---

## 6. Structured alternatives for the generated transform

Orthogonal to §4: any structure below can still have its generator low-ranked.

| Generated ΔW(c) | Generated params / sample-or-token | Apply cost | GPU fit | Notes |
|---|---|---|---|---|
| Low-rank U·V (current) | `2rd` | `O(Trd)` | good with §4/§5 | rank-limited |
| Block-diagonal, block b | `d·b` | `O(Tdb)` bmm | **good** (tensor cores) | no cross-block mixing; static W supplies it; shift block boundaries across layers instead of Monarch's permutation |
| `H·D₁(c)·H·D₂(c)` Hadamard sandwich | `2d` (= AdaLN cost) | `O(Td log d)` | ok, needs FWHT kernel | **full-rank conditioned transform at AdaLN cost**; ±1 entries, bf16-safe; Fastfood-style |
| Kronecker `A(c) ⊗ B(c)`, √d×√d | `≈2d` | two small matmuls via reshape | good | full-rank; sum of a few Kroneckers for expressivity |
| Circulant via FFT | `d` | `O(Td log d)` | meh | shift-equivariant bias over *channels* is odd; half-precision FFT constraints; prefer Hadamard |
| Tensor-train cores | small | chain of small contractions | poor unless fused | elegant, launch-heavy |
| r Householder reflections | `rd` | `O(Trd)`, sequential in r | ok | orthogonal, norm-preserving, very stable |

Caveat from the Monarch / Fastfood literature: at d ≈ 1–4k, structured ops often lose wall-clock to dense GEMM
(bandwidth-bound, many launches). Block-diagonal and Kronecker map best to tensor cores; Hadamard is the best
full-rank option.

**Per-token note:** generated-parameter count per token *is* the activation-memory cost. The Hadamard sandwich
(`2d`) and Kronecker (`≈2d`) are at AdaLN-level memory while being full-rank, which makes them strong per-token
candidates.

---

## 7. Diagonal plus low-rank (DPLR)

`T(c) = D(c) + U(c)V(c)` or product form `D(c)(I + U(c)V(c))`.

- **Division of labor:** diagonal = per-channel gain (AdaLN's strength); low-rank = cross-channel mixing (AdaLN's gap).
  The low-rank part is not generally an orthogonal rotation or norm-preserving map.
- **Algebra:** cheap inverse (Woodbury), cheap determinant (matrix determinant lemma), relevant if this is ever used in
  flows. Same structure family as S4's DPLR parameterization, so there is prior art on stable parameterization.
- **Sum vs product:** product composes cleanly with a bounded low-rank factor. With a Householder/Cayley low-rank
  part you get rotation then per-channel scale — an SVD-shaped conditional transform with the diagonal as singular values.
- **Diagonal placement:** input-side (AdaLN-style pre-scale), output-side (gating), or both (conditioned
  diagonal–dense–diagonal sandwich around static W).
- **Shared vs independent heads:** `s(c) = Σ a_k s_k` reuses the basis coefficients (fewer params, but ties gain to
  cross-channel mixing). Ablate against an independent `to_scale` head.
- **Memory:** per token, `s` costs `d`, same as per-token AdaLN; recompute from `c` in backward if it matters.
- **Init:** zero the scale head and `V`; **do not** also zero `to_coef`, or both sides get zero gradient forever.
  (`DPLRCond` follows this; self-test checks it reduces to the base layer at init.)

---

## 8. Other design levers

- **Cross-layer sharing:** one hypernetwork + per-layer embedding/offset (PixArt-α "adaLN-single" style). Likely the
  largest parameter saving available. With basis mixing: share coefficient heads, keep per-layer bases, or vice versa.
- **Quadratic scale:** `U(c)V(c)` scales as |c|², and so do its gradients. Zero-init the V side; normalize c; consider
  L2-normalizing `U(c)` columns (weight-norm on generated factors).
- **Application form:** additive `Wx + xUV`, multiplicative `W(I + UV)x`, or orthogonal rotation of x.
- **Placement:** QKV vs MLP-in vs output projections. One shared `U(c)` feeding Q, K, V jointly is cheap.
- **The r bottleneck after factorization:** with CP the rank-r stage sits between two gates; test whether removing it
  (going straight from gate 1 to gate 2) helps or hurts.
- **Self-conditioning:** each input contraction is bilinear in token and condition.
  If `c=h` and both factor generators are bias-free linear maps, the complete
  `delta = (h @ U(h)) @ V(h)` is cubic in h. Affine heads add lower-degree terms;
  nonlinear generators/normalization change this analysis. This still relates
  to bilinear / multiplicative MLPs (see Jayakumar et al. 2020), but the complete
  two-factor layer should not be labeled generally second-order.

---

## 9. Experimental plan

This is the original proposed order. Subsequent completed screens and
confirmations are recorded below. The FFN/core quality comparison is now
complete; isolated timing and generator-width/rank quality tradeoffs remain
follow-ups.
The expanded generator/diagonal alternatives are proposals rather than evidence
that these original hypotheses have been established.

### 9.1 Hypotheses

- **H1** Exact chunked full hypernet reproduces current results while saving approximately x+c
  plus a small rank-r bottleneck, without retaining full generated factors.
- **H2** Basis mixing at r=8–16 matches or beats full hypernet at r=2, at < 10% of the params and < 10% of the FLOPs.
- **H3** DPLR beats both AdaLN and pure low-rank at matched params.
- **H4** A full-rank structured variant (Hadamard sandwich or block-diag) at AdaLN-level cost beats low-rank at matched cost.
- **H5** Cross-layer sharing loses little quality for a large parameter reduction.

### 9.2 Runs, in order

1. **Correctness + profiling of exact path.** Confirm parity with current implementation; profile peak memory
   (`torch.cuda.max_memory_allocated`), tokens/s, and compiled vs eager. Sweep chunk size.
2. **Factorized sweep.** Basis mixing over K ∈ {4, 8, 16, 32} × r ∈ {2, 4, 8, 16}. Activation vs weight mixing where c
   is per-sample.
3. **DPLR.** Sum vs product; diagonal placement; shared vs independent diagonal head.
4. **Structured.** Hadamard sandwich and block-diagonal (b ∈ {16, 32, 64}) with low-ranked generators.
5. **Sharing.** Cross-layer hypernetwork sharing on the best variant from 2–4.

### 9.3 Baselines (needed to attribute gains)

- AdaLN / FiLM with matched params and matched FLOPs.
- Static LoRA of the same rank + AdaLN — isolates *condition-dependent* mixing from added capacity.
- Simply widening the model to the same parameter / FLOP budget.

### 9.4 Metrics, reported per run

Task quality · total params · extra params · peak training memory · tokens/s (train and inference) ·
FLOPs/token · stability (loss spikes, grad norms of hypernet params).

### 9.5 Fairness rules

Match on **both** params and FLOPs where possible; when they conflict, report two baselines. Same LR schedule, tune
LR for the hypernet parameters separately (their gradients scale with |c|).

---

## 10. Risks and open questions

- Does low-ranking the hypernetwork lose something the full tensor captured? H2 tests this directly; Tucker is the
  fallback if CP/basis is too restrictive.
- Exact path FLOPs (≈4× a dense linear at r=2) may dominate step time even with memory solved.
- Structured full-rank ops may underperform dense GEMM in wall-clock at this d.
- Training stability with per-token, quadratically-scaled transforms; normalization choices may matter more than structure.
- Interaction with mixed precision: bf16 accumulation in the j-reduction (keep accumulators in fp32 in any fused kernel).

---

## 11. Engineering notes

- Chunk size: pick to bound transient `chunk·d_c·r` and keep GEMMs large enough to saturate; start at 2048–4096 tokens.
- `torch.compile(fullgraph=True)` on the chunk bodies only; avoid compiling the Python loop.
- In any Triton kernel, accumulate `z` and `dA`/`dB` in fp32.
- Keep `NaiveCondLowRank` as an oracle on small shapes in CI; run `python condlr.py` after any kernel change.
- Measure saved-activation memory with `torch.autograd.graph.saved_tensors_hooks` (the method used for §3's table);
  measure peak with `torch.cuda.reset_peak_memory_stats` / `max_memory_allocated`.

---

## 12. References

- Perez et al., *FiLM: Visual Reasoning with a General Conditioning Layer*, AAAI 2018.
- Peebles & Xie, *Scalable Diffusion Models with Transformers* (DiT, adaLN-Zero), ICCV 2023.
- Chen et al., *PixArt-α* (adaLN-single), 2023.
- Ha, Dai & Le, *HyperNetworks*, 2016.
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021.
- Jayakumar et al., *Multiplicative Interactions and Where to Find Them*, ICLR 2020.
- Dao et al., *Monarch: Expressive Structured Matrices for Efficient and Accurate Training*, ICML 2022.
- Le, Sarlós & Smola, *Fastfood — Approximating Kernel Expansions in Loglinear Time*, ICML 2013.
- Gu, Goel & Ré, *Efficiently Modeling Long Sequences with Structured State Spaces* (S4, DPLR), ICLR 2022.
- Oseledets, *Tensor-Train Decomposition*, SIAM J. Sci. Comput. 2011; Novikov et al., *Tensorizing Neural Networks*, NeurIPS 2015.
- Dao et al., *FlashAttention*, NeurIPS 2022 (tiling/recompute pattern for the fused kernel).

### Longer rank-2 confirmation protocol (2026-10-08)

The short fresh-seed quality screen motivated a focused full-loader calibration
and longer paired-seed comparison. `dit_rank2_confirmation.py` calibrates basis
r2 and conditional bottleneck r2 against scaled basis r8 on the actual full
training loader, using seeds 48/49 and highest-FP32 fixed-input transform-update
RMS averaged across six blocks and steps 50/100/250/500. Calibration runs a
500-step prefix of the same 30,000-step LR schedule, enabled by the optional
`lr_schedule_steps` field. The bounded rate rule and ±25% tolerance were fixed
before launch; selection never uses validation or generated quality.

Both families matched at multiplier 1 without corrections (basis r2 0.964×,
bottleneck r2 1.034×). These global rates are frozen for fresh paired seeds
50/51/52. Confirmation runs 30k steps for basis r2, bottleneck r2, scaled/raw
r8, and AdaLN-192/240, reusing the frozen evaluators and split. Warmup 50 differs
from the historical 30k study, so only current controls match this recipe.
The comparison retains no model weights and makes no claim of equal updates
at every step, variance preservation, universal rank scaling, isolated speed
advantage or sequence-conditioned cross-attention performance. See the protocol
and current status in [the README](README.md#full-loader-calibrated-rank-2-confirmation).

### Completed longer confirmation

The full-loader calibration and 18 paired 30k-step runs are complete. Basis r2
improves over AdaLN-192 on FID and requested-class agreement in all three fresh
training seeds, but both rank-8 references perform better. The tiny conditional
bottleneck does not show a clear quality improvement over the base. Raw r8 has
the best observed mean FID/class agreement and beats the current AdaLN-240
control on both in all paired seeds, with fewer parameters and lower whole-trainer
peak allocation; concurrent throughput remains lower. These are recipe-specific
observations, not an isolated speed/wall-clock win or sequence-conditioning result.

Full-data calibration selected multiplier 1 for both rank-2 families without
quality selection. The match transfers only approximately: bottleneck seed 50
misses early tolerance, and its update ratio drifts lower later. All 62 tests
and the independent audit passed; no new model weights/checkpoints or official
test evaluations were saved/performed. The [report](reports/dit-rank2-confirmation.md)
contains means/SD, per-seed paired differences, update/magnitude plots, actual
memory/timing scope and limitations. The design notes include a proposed
small conditional core and a near-exact parameter-matched FFN control. The
subsequent generator/core engineering screen implements dense and
diagonal-plus-low-rank cores and exercises the FFN control; these remain
outside the longer quality-confirmation cohort above.

They now have a separate [calibrated three-seed 30k confirmation](reports/dit-core-confirmation.md),
summarized near the start of this brief. The new cohort strengthens the
basis-r2 versus FFN agreement result, while exposing a large quality loss from
the cheaper conditional cores and later function-update drift. The earlier
raw-r8 FID ordering does not reproduce on the new seeds. Historical cohorts
are kept separate rather than pooled or substituted for current controls.


**2026-10-16 objective and operator-space update:** The goal covers both making
conditional low-rank transforms practical and exploring alternative generated
operators. The next implemented study changes the operator itself: square
block structure and conditional Hadamard mixing, with approximately matched
square low-rank/diagonal controls and the established rectangular reference.
It retains broadcast diffusion, common AdaLN and a zero-delta parallel branch;
no toy retrieval or new LM benchmark is launched. The six-preflight/22-run
protocol, PyTorch shapes, gains, generation/application costs, limitations and
resource/artifact policy are in [the structured-operator plan](reports/dit-structured-plan.md).


**2026-10-17 structured-operator results:** The first22-run screen completed
with all nine audits passing. Conditional Hadamard at adapter LR2 improves
on its nearly budget-matched square diagonal control in both seeds, but
loses observed training throughput; block structure offers a different
quality/throughput/allocation tradeoff against square low-rank. Rectangular
DPLR remains the quality reference. No general operator winner or new default
is established. See [the complete structured-operator report](reports/dit-structured.md)
for every rate/seed, matched-placement gaps, costs and limitations.


Conditional tensor-train operators are now implemented for the next bounded
DiT study. Four paired channel modes `[4,4,4,4]`, TT bonds6/11 and one shared
affine latent32 give420,608/1,180,928 adapter parameters. TT bond rank is not
ordinary matrix rank. Training applies mode contractions without a dense
operator; token-axis contraction intermediates still require measurement.
A zero final core gives an exact zero delta, with nonzero earlier static cores
and audited gradient wake-up. The24-run plan uses diagonal, square low-rank
and block controls plus the contextual rectangular reference, two fresh seeds
and the existing two adapter rates. Two-GPU cap, no weights/checkpoints.
Validation and compiled GPU preflight must pass before quality begins.
[Conditional TT protocol](reports/dit-tt-plan.md).


All six TT-study CUDA preflights passed both the controller audit and an
independent frozen-runner replay. Every adapter tensor received a nonzero
gradient by step5, including all80 tensors in each TT variant. Source/config,
paired streams/initializations, actual optimizer groups/rates, claim identity
and evaluator budgets verified. Short-preflight whole-run peak allocations
were1.91GiB (TT6) and2.37GiB (TT11); these are not sustained quality-training
resource results. The24 full30k quality runs are now open under the two-worker
cap. Final quality and audit remain pending; no model weights/checkpoints.
[TT preflight evidence](reports/dit-tt-preflight.json) ·
[TT startup status](reports/dit-tt-startup.json).
