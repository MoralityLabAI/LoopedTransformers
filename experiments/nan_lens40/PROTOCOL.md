# NAN-LENS-40 phase-conditioned readout

Goodfire’s Inkling 2 Mini study found that a tuned lens recovered information
about the model’s final output at sites where its raw head struggled. Nanbeige
shares decoder weights across passes and has a common output head. Use the
archived Nanbeige sites from each physical pass and compare the raw head,
a constant calibration shift, and a site-specific linear residual correction.
Nanbeige has no Inkling-style coda in this run. [Inkling lens study](https://www.goodfire.com/research/interpreting-inkling-2-mini/#decoding-loops-logit-lens-tuned-lens-coda-lens)

## Inputs and frozen calibration

Reuse the hashed COMP14 NF4 and BF16 audits and READPATH37 head closure.
The roster is 8 CAL and 16 exposed DEV pointer roots, both worlds at each
root, and focused, literal-copy, and graph-erased prompts. Read last-prompt-token
residuals at zero-based layers 5, 11, 17 and 21 in each pass. The final
phase-2 layer-21 logits restricted to the same 16 bare labels A–P are the lens
target. Recompute those logits from the frozen RMS norm and head. At both
layer-21 sites independently replay the archived selected logits and
unrestricted winner logit within 0.125 logit. Bind the pinned archive,
checkpoint and tokenizer, and abort on any closure failure.

For each backend and each nonfinal site fit on the CAL focused and copy
rows only. Subtract each row’s mean across the 16 output logits. The raw
lens is this centered direct head readout. The constant calibration shift
adds the CAL mean final-minus-raw vector. The linear lens adds the ridge
prediction of the centered remainder. Features are unit-L2 residuals
centered by the CAL feature mean. The fixed penalty is 0.1 in the unit-L2
linear kernel. Repeat the fit with coordinate 2279 dropped before feature
normalization, while using the same unedited raw head for both fits.
This measures the contribution of that coordinate to the fitted correction.
Use the final site itself as an identity control. No tuning on DEV or
selection of a winning site. The lens objective never takes oracle labels.

## Endpoints and custody

Report final-output fidelity (mean restricted KL and restricted top-1 agreement)
separately from restricted task-oracle correctness and from the archived
unrestricted native choice. Keep ties explicit. Count both-world solves,
and paired repairs, regressions and unknowns relative to the same-site raw
head. Report CAL, DEV, task arms, stages, and precisions separately. The erased
twins are an input-identity control kept outside the fit. Scores cover the
registered 16 labels, with archived unrestricted winner support reported
separately. Keep raw logits, per-outcome records and named roots local.

Each independent unit is one pointer root. Both worlds, arms, sites, and
precisions sharing that root are repeated observations. These are observational
calibration and exposed development diagnostics on one pretrained checkpoint.
Physical dose and matched sham floor do not apply to a post-hoc observer.
No inference tests, causal threshold, learned stopping policy, model forwards,
or sealed ARG-03 data. A fresh source-level holdout is needed for a prospective
lens comparison. A causal threshold and matched sham are needed before using
the lens to support a mechanism claim. [EvalMath FSI-01](https://github.com/MoralityLabAI/EvalMath/blob/main/paper/finite_sample_inference.md)

Run from the checkout using the pinned Nanbeige environment and an empty
local output directory. The existing CPU lease requires 5 GiB available for
admission, reserves 4 GiB and applies a 0.75 GiB private cap, below-normal
priority, one numerical thread and a 600-second limit.

```powershell
python -B -m experiments.nan_lens40.run --snapshot <pinned-snapshot> --output runs/nan_lens40/attempt01
```
