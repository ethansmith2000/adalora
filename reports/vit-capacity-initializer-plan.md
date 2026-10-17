# Initializer and learning-rate axes for the wider control

The user agrees that the wider-control diagnostic warrants checking initialization
and LR, and proposes Xavier, Kaiming, or a layer-specific mix. Record these as
candidate designs; no initializer is selected and no GPU cohort is launched.

For the current GELU MLP, the hidden width changes from 768 to 1483 while the
input/output width stays 192. Fan-in scaling provides a direct way to keep
the input projection scale fixed and reduce the output projection scale as
its number of inputs grows:

```python
fc1_std = gain_in / sqrt(192)
fc2_std = gain_out / sqrt(hidden_dim)
```

The gains remain explicit choices. The first projection precedes GELU;
the second feeds the residual stream. A ReLU gain is not an exact GELU rule.
PyTorch's `calculate_gain` has no GELU entry. `kaiming_normal_(..., mode="fan_in",
nonlinearity="linear")` implements gain 1 fan-in normalization; it is a useful
baseline with any activation/residual gain declared separately.

Xavier normal is another candidate:

```python
fc1_std = gain_in * sqrt(2 / (192 + hidden_dim))
fc2_std = gain_out * sqrt(2 / (hidden_dim + 192))
```

This scales with both fan-in and fan-out. Widening therefore also changes
the first projection's preactivation distribution, despite its unchanged fan-in.
Together with GELU, it does not guarantee invariant full-MLP output RMS.
A mixed recipe, such as fan-in scaling before GELU and Xavier on the output
projection, should be a named, measured candidate rather than an implicit
choice. Both width ratios and absolute residual-branch RMS matter: default
fan-aware scales can be much larger than our legacy std 0.02, so fixing width
dependence alone can introduce a new overall-gain difference.

The installed PyTorch 2.11 implementation and official documentation agree on
the formulas/modes:
[Xavier](https://docs.pytorch.org/docs/2.11/nn.init.html#torch.nn.init.xavier_normal_) ·
[Kaiming](https://docs.pytorch.org/docs/2.11/nn.init.html#torch.nn.init.kaiming_normal_) ·
[Activation gains](https://docs.pytorch.org/docs/2.11/nn.init.html#torch.nn.init.calculate_gain).

Candidate axes for a bounded follow-up:

1. Retain the exact legacy fixed-std recipe as an anchor.
2. Retain the narrow-reference fc2 correction as a diagnostic of width-induced
   gain, independent of changing initialization family.
3. Compare explicit fan-in and Xavier/layer-specific alternatives after probing
   initial preactivation RMS, post-GELU second moment and residual-branch RMS
   at both widths. Declare activation and residual gains.
4. Cross the retained candidates with a small LR bracket. Initial scale does
   not determine functional update scale: optimizer steps and clipping can
   still differ with hidden width. Track early updates and clipping as well
   as training and validation curves.

Apply the common initializer policy to the base MLP in plain, wider plain and
DPLR comparisons with a declared comparable tuning budget. Keep attention,
embeddings and norms unchanged in this MLP-focused intervention, and preserve
the zero classifier/scale/shift/V-factor/offset rules that define the original
starting functions. Do not zero both generated low-rank sides or their
coefficient projector. Document the exact scope and source snapshot before
launch; unchanged historical runs remain anchors, not rerun results.

The GPU grid, gains, seeds, selection criterion and compute/artifact budget
remain to be frozen before any training launch. The two-GPU cap and no-default-
checkpoint preference continue to apply. No training source, default or
historical experiment has changed in this discussion.

[Measured wider-control diagnostic](vit-capacity-diagnostic.md).
