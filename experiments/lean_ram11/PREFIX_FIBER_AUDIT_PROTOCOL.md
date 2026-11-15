# RAM-11 | Prefix-fiber witness integrity audit

**Status:** Retrospective correction to the posthoc RAM-09 and RAM-10W graph analyses, November 15, 2026. The discrepancy motivating this work was identified before this document was written. It is not a new preregistered model test. Original RAM-08 and RAM-10 prospective theorem outcomes remain unscored.

## Frozen evidence

Read only the archived original RAM-08 train/dev observations from [original Actions run 43689874169](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43689874169), artifact 14152063191, ZIP SHA256 `a2ac363547858871c4e066c96bac67253dfeb7ff4e2a0050bb5886c8ca9edcfc`. The 20,930 compiler observations have transitions SHA256 `f92d5654fdb8fc1d4bd43e7fb4c0b1f03b7a27c1d7335dd86e6ec947b582542f`, a published separate Lean-replay audit. Preserve the original TRAIN/DEV split. No TEST/PEF source, reference script, new Lean invocation, weight fitting or CI runner is needed to reproduce the new diagnostic.

## Mathematical issue: a goal-quotient edge can splice incompatible prefixes

Every recorded row has original theorem root `r`, exact prior tactic-prefix `p`, Lean-printed state `x_p`, candidate tactic `a`, and a compiler-replayed effect `T(x_p,a)`. RAM-09 used `h(x)`, a canonical SHA of printed goals, as its graph vertex key. A quotient edge goes from `h(x_p)` to `h(x_q)` when the candidate's after-state printed goals match, even if `q != p + [a]`. This creates a *counterfactual splice* whose concatenated script was not necessarily exercised.

**Strict prefix-fiber graph** instead keys each node by `(source_file, theorem, tuple(prefix))`. An accepted positive nonterminal tactic `a` at `p` may connect only to a recorded node with prefix **exactly** `p + [a]`. Require full equality of its before/after printed goal lists and sorted visible definitions. Any other positive successor is **censored**. Terminal label3 actions become terminal edges. Intra-prefix consistency and source SHA must hold across all original candidate rows. Preserve order, failure labels, and the 20930-row original source.

This graph supplies evidence that *each action in a reconstructed witness was recorded as an actual next prefix*, with no state-hash join. Still, these historical compiler effects alone are NOT a third independent complete original theorem-source compilation or a kernel certificate; complete source replay must precede any solved-theorem count.

Let `R_prefix` be the set of states with a terminal path in the prefix-fiber graph. Let `R_quotient` be the original visible-goal quotient graph reachability. On the unchanged collection, every strict prefix edge induces a valid quotient edge:

[
R_{\mathrm{prefix}}\subseteq R_{\mathrm{quotient}}.
]

The difference consists of abstract goal-quotient routes without a complete, recorded exact-prefix path. An original theorem may still be provable by an uncollected tactic route. There is **no** justified inclusion of `R_quotient` in actual kernel-certified theorems. Require the independent Lean compiler for that claim.

## Planned immutable audit

1. Verify original archive SHA, transitions SHA and independent replay receipt; enforce exactly 20,930 records.
2. Build both graphs separately for original TRAIN and DEV, confirming each unique prefix yields consistent original source bytes, state signature, goals and definitions.
3. Count successful terminal edges, strict matching positive successor edges, censored successors, shortest witness depths, strict-reachable states and quotient-reachable states.
4. Count quotient-only states and unsafe quotient transitions. Show examples, with source/theorem/prefix hashes but without copying gold proof bodies.
5. Verify `R_prefix subseteq R_quotient` directly. Run unit tests that construct an explicit counterfactual splice from equal visible goals and distinct script prefixes.
6. Reevaluate *frozen original* RAM-08 selected checkpoints on both retrospective graphs, with exactly the same action-scoring formula and state rows. No model updates or selection.
7. If implementation and local tests pass, commit the source, exact receipt, and audit note on a distinct branch; keep previous reports intact and label their 248/714 coverage as **goal-quotient** reachability.

Previous exploratory RAM-09 pairwise and RAM-09B balanced models were trained using the goal-quotient positive labels. Treat their recorded success-targets as **abstract witness targets** until matching exact-prefix path or complete kernel replay is available.

No spending, paid resource calls, merges or changes to the original held-out theorem-test registrations.
