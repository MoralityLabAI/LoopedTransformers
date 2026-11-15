# RAM-12R protocol | bounded state-conditioned lookahead from native historical proof traces

**Frozen before the first RAM-12R model fit, 2026-11-15.** This is a *retrospective* training study using previously selected MathCan old-TRAIN theorem roots. The registered RAM-08 PEF and RAM-10 prospective tests remain unscored and are excluded entirely. Historical RAM-11F scripts have already influenced the question and protocol, so this is not a preregistered independent generalization test.

## Immutable input and trust

Native RAM-12 source run [43716751350](https://github.com/patrickdugan/MathCan/actions/runs/43716751350), artifact ID 14162833338, ZIP SHA256 `327dbd1382aca1a1c3c790e434341d582aafb85e5ad1b4f9b70ca84f21cae924`. All 76 historical original theorem scripts were compiled twice with in-line read-only Lean goal instrumentation, for 152 complete source compiles, 282 action occurrences, 176 distinct compiler goal contexts and **230 unique positive state-action records**. Original MathCan source `c84e2025334e9a967c1297425a490f98d5b3d189`, worklist SHA256 `f5876539862a0f4f913dc3af620be0d038555efcdd3611754e002dc353c3b2aa`. Native state labels file SHA256 `f744f779012c04d50a41798ee3805c91fbb8da922a174b09ccbde3310eb95e8f`. Independent local audit covers all 76 source pairs, compiler log hashes and action-to-prefix provenance.

Original immutable RAM-08 TRAIN-only action archive `ram08-frozen-audited-train-43689874169.zip`, SHA256 `a2ac363547858871c4e066c96bac67253dfeb7ff4e2a0050bb5886c8ca9edcfc`, 20,930 independently replayed Lean effects, with its original nine CPU checkpoints. Original selected models:
- loop32/seed29 `4a88bc80456743a89a70882911060cc90e76785a5ae1d435a0e46a8f8c09778f`;
- loop64/seed43 `5cbd327904714ac698897e1c2333cae9ec4125aeb70e7c56cd6236db8a737dac`;
- flat64/seed29 `2aa101edbd56f6563ebb2edcfc8c61c07dcc027559ea4cdc551187d6ec5303d8`.

Root-level TRAIN/DEV/HOLDBACK split is fixed by SHA256(`RAM12-FRESH-STATE-20261115/` + source_file + ':' + theorem) mod6: 14/3/5 original roots, with 137/31/62 certified positive pairs and 105/24/47 exact-prefix states. These roots were all eligible for original RAM-08 training, so internal RAM-12 holdback is **not** independent of original frozen model exposures.

## Censor-aware new supervision

A complete Lean-proved suffix certifies a **positive** first action. An alternative action with no observed complete suffix is *unknown*. Only use a negative when RAM-08's original compiler independently recorded a **failed/unchanged** action at the **same exact theorem root and tactic prefix**; never join on printed-goal hash alone. Of 137 TRAIN positives, 83 have at least one matching independent original compiler-failed action; the other 54 contribute positive-only demonstrations, with no fabricated failures. Deduplicate RAM-11F proof scripts on original theorem+prefix+action and preserve exact original next-action spelling.

The auxiliary *length preference* uses two different **certified positive** next actions at exactly the same prefix, with a different remaining action count in their supporting complete proof. The shorter observed certified suffix is a preference target (cost), and the longer positive action remains valid. This does not assert the shorter witness is globally optimal.

## Bounded residual experiment

Frozen original 85K/207K/182K parameter trunks, no initial weight updates, exactly three residual heads. For each model the original RAM-08 scalar tactic score is
[
r_0(s,a)=3p_3+1.5p_1+0.3p_2-0.6p_0+0.4\operatorname{clip}(\hat\Delta,-3,3)
]
and the new score is `r(s,a)=r0(s,a)+0.35*tanh(w^T z0(s,a)+b)`. Its rank-preservation theorem is unchanged: an original winner with >0.70 score margin over every competitor cannot lose the top position under any residual-head weights.

Fit exactly **150** CPU AdamW optimizer updates, batch32 per model, LR0.003 weight decay0.01, seed20261115, 2 CPU threads, precision float32, deterministic algorithms. Each batch samples verified positive vs original verified rejected pairs from TRAIN, balanced by rejected family where possible; the extra shorter-vs-longer **positive** comparisons receive weight 0.2 if nonempty. Add 0.02 squared tanh-residual regularization. No DEV-based tuning, model resampling or early stopping. Run the complete same budget on loop32, loop64, flat64. Save hashes of each head and immutable original model hash.

## Frozen evaluation and interpretation

Report before/after on all original DEV and HOLDBACK exact-prefix contexts that have at least one fully source-verified positive plus at least one original *verified rejected* action, using **top-1 selected positive among only these observed actions**, candidate counts and overall positive-vs-failed pairwise win rate. Separately report minimal observed full-proof suffix preference among contexts with two or more certified positive actions. Report original frozen score, unmodified global family-frequency and shape-frequency as symbolic historical controls, with identical candidate pools. Disclose counts of censored positive-only contexts.

The descriptive success target is at least 2/3 models improving holdback top-1 certified-positive selection with no worse-than-one-state degradation on any other model. A failed target is reported as negative. No simulated "proof gain" credit: only a fresh native Lean search with a full third axiom-gated source replay can support a theorem-level capability claim. Distinguish selected observed actions from actual re-execution of an entire proof-search policy.

No new original PEF or RAM-10 test checkout, no paid compute, deployment or merge before code/audit review.
