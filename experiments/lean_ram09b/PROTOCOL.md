# RAM-09B frozen exploratory protocol: balanced family-hard negatives

**Protocol committed before RAM-09B fitting/evaluation, November 15, 2026.** This is a prospective *implementation step* with a predeclared algorithm and evaluation. Its evidence remains **retrospective** because the old RAM-08 original MathCan TRAIN source and the same 15-root internal holdback were already examined in RAM-09 stage 0. It is not a newly independent theorem-family transfer and cannot establish RAM-08's separate PEF hypothesis.

## Frozen inputs

- Original source `patrickdugan/MathCan@c84e2025334e9a967c1297425a490f98d5b3d189` only, archived 20,930 independently replayed compiler observations.
- The original `ram08-frozen-audited-train-43689874169` archive (SHA256 `a2ac363547858871c4e066c96bac67253dfeb7ff4e2a0050bb5886c8ca9edcfc`) and transitions (SHA256 `f92d5654fdb8fc1d4bd43e7fb4c0b1f03b7a27c1d7335dd86e6ec947b582542f`).
- Unchanged RAM-09 stage-0 deterministic split: SHA256(`RAM09-GRAPH-20261115/` + `file:theorem`) modulo five, holdback when zero. Forty-two fit roots, 15 old internal holdback roots, plus five original RAM-08 DEV roots that were previously used to select checkpoints.
- Three exact frozen RAM-08 best checkpoint SHA256s: loop32/seed29 `4a88bc...09778f`, loop64/seed43 `5cbd327...737dac`, flat64/seed29 `2aa101e...0303d8`.
- No PEF source checkout or theorem goal access. The original 120-case PEF source-family test remains zero-scored. This study does not generate a kernel certificate.

## Graph labels and sampling

The original RAM-09 `ObservedClosureGraph` reconstructs only genuine compiler-observed transitions. For each fit state with at least one *observed* full closure continuation of four or fewer tactics, select its minimum-distance recorded first action. For equal distance, prefer the original frozen model's high-scoring first action, tie broken by registered candidate order. Every candidate with positive effect but unobserved ultimate closure remains **censored** and is excluded from this loss.

For each eligible state, group genuinely rejected/unchanged original Lean actions by registered tactic family. Within each failure family, retain the highest-original-score failed script. Each optimizer update draws a positive tactic family uniformly, an eligible state uniformly within that family and a failed tactic family uniformly within that state's observed failed tactic families. This removes the earlier `apply_local` frequency weight. If a fit state has no compiler-rejected action, it is omitted. There is **no reference-proof prefix, synthetic Lean tactic, unknown-as-failure relabeling, or PEF test data**.

## Frozen training budget and comparator

Three separate frozen trunks, trainable **linear residual score head** only; all original model parameters remain frozen. CPU PyTorch float32 with deterministic algorithms and two threads; seed 20261115 for sampling and head initialization. 200 optimizer updates, batch size 32, AdamW LR 0.003 weight decay 0.01, regularizer weight 0.02. No early stopping, no holdback-based hyperparameter choice.

Let `r_0(s,a)` be original fixed RAM-08 scalar tactic rank score and `z_0(s,a)` its final frozen feature vector. Train

[
r_B(s,a) = r_0(s,a) + 0.35\tanh(w^\top z_0(s,a)+b).
]

A pairwise softplus loss `softplus(r_B(failed) - r_B(witness))` with squared tanh-residual regularization trains the head. The bound `|r_B-r_0| <= 0.35` implies any base winning margin strictly above 0.70 remains a winning margin. Check this property per original evaluated state.

**Comparators:** original frozen rankers and already-published RAM-09G frozen-trunk residual pilot; both use the exact same old holdback. Report all model families, even failures.

## Evaluation, fixed before run

Primary diagnostic: top-1 *observed witness-first-action* among 39 graph-reachable old holdback states. Also report overall useful, immediate closure, witness-first and high-margin preservation across all 191 old holdback states. Report the old five-root RAM-08 DEV as previously-selected, exploratory only. Report model-specific distribution of selected positive and negative families and representation of rare `exact_typed`, `rcases_local`, `induction_list` in the fit data.

**Preregistered descriptive target:** at least two of the three balanced residual models improve over their original frozen old holdback witness-first-action count, with no model degrading more than one state. Any failure, flat-model-only benefit or rare-family oversampling must be disclosed.

Any future new PEF theorems, alternate roots, or new positive exact collections require a new prospective protocol with a source and goal-state freeze before access, and independently compiled complete Lean theorem certification. No merge, hardware spending or new workflow dispatch.
