# RAM-10W: historical progress labels versus observed closure witnesses

**Data:** Retrospective analysis only of the immutable original RAM-08 old-MathCan
TRAIN and previously used DEV corpus. Original source-only Actions run
`43689874169` produced **20,930** independently replayed compiler effects,
verified in 62 fresh Lean sessions. Original ZIP SHA256
`a2ac363547858871c4e066c96bac67253dfeb7ff4e2a0050bb5886c8ca9edcfc`;
original transitions SHA256
`f92d5654fdb8fc1d4bd43e7fb4c0b1f03b7a27c1d7335dd86e6ec947b582542f`.
This analysis never opens RAM-10 TEST or PEF proof files and never adds old labels
to RAM-10 TRAIN. The preregistered positive family gate remains unchanged.

## The methodological issue

An accepted `induction xs` often splits one pending goal into multiple goals.
That is genuine Lean progress in the registered four-label convention, even
when its eventual ability to finish the theorem is unknown. Treating every
accepted branch-creating tactic as a terminal-proof-positive action would
amplify the behavior seen in RAM-09's first unsuccessful lookahead training.

The **recorded** old-source TRAIN effects show:

| Family | Compiler actions tested | Accepted progress | Immediate closures | Reach an observed terminal through recorded successors | Unobserved child states | Observed child with no *recorded* completion |
|---|---:|---:|---:|---:|---:|---:|
| `exact_typed` | 2,533 | 5 | 4 | **5** | 0 | 0 |
| `exact_local` | 2,116 | 3 | 2 | **3** | 0 | 0 |
| `rw_local` | 648 | 218 | 2 | **29** | 118 | 71 |
| `rcases_local` | 11 | 11 | 0 | **0** | 8 | 3 |
| `induction_list` | 290 | 290 | 0 | **9** | 201 | 80 |

The last two columns partition the accepted actions lacking a witnessed
closure, after excluding immediate goal closures. The number of reachable
witnesses is a property of the **finite sampled graph** rather than a
probability of eventual theorem success. Every unlabeled successor remains
open-world and censored. This analysis therefore makes **no** claim that
`rcases` failed eleven proofs or that the other 281 induction results were
unproductive. They simply lack a *recorded* continuation to a terminal state.

## A monotone lower-evidence theorem

Fix an underlying Lean compiler transition relation \(T\) on original goal
states, together with the goal-closed terminal set \(F\). Let \(G\subseteq T\)
be a finite audited subgraph whose edges are original compiler-accepted tactic
scripts. Define

\[
W_G(s,a)=\mathbf{1}\{\text{the observed edge }(s,a,s')\in G
\text{ reaches }F\text{ by a finite path entirely inside }G\}.
\]

For any two such audited graphs \(G\subseteq G'\subseteq T\),

\[
 W_G(s,a)\le W_{G'}(s,a)\le W_T(s,a).
\]

**Proof.** If \(W_G(s,a)=1\), its finite path consists of authentic compiler
edges in \(G\). Inclusion carries each edge into \(G'\) and then into the
full relation \(T\), so the identical path remains available. For a zero
value the inequality follows because the indicator is nonnegative. This
requires no probabilistic assumptions, and a replay disagreement invalidates
the graph's audited-subset premise.

When the set of previously accepted source actions is **fixed**, the fraction
with recorded closure witnesses can only rise as we explore and replay more
authentic successors. Adding brand-new actions to the denominator voids that
fraction's monotonicity. This is a useful training property: recorded witness
positives may safely persist across gym growth; unwitnessed positive
transitions should continue to carry an **unknown** continuation label.

Each certified finite witness is evidence for a compiler route from a recorded
goal state. A theorem-level score still needs a complete, separately compiled
original Lean proof including all pending goals, the allowed axiom inventory,
and an independent byte-identical recompile. The audit does **not** confer
that certificate on any observed state path.

## Reproduction

From the stacked RAM-10 branch, with Python and the immutable RAM-08 ZIP:

```bash
python experiments/lean_ram10/progress_witness_audit.py \
  ram08-frozen-audited-train-43689874169.zip \
  --out out/ram10w_original_history.json
python -m unittest discover -s experiments/lean_ram10 -p 'test_progress_witness_audit.py'
```

The original archival SHA and independent Lean replay receipt are checked
before loading one compiler label. Results are recorded as an unscored
historical diagnostic in the research-gym; no source reference proof, RAM-10
held-out state or new hypothesis test was involved.

A future separate RAM-11 registration could train a *censor-aware*
continuation head and compare it with one-step four-outcome CE, including
native complete Lean proof search and stronger symbolic controllers. The
present preregistered RAM-10 training objective is preserved.
