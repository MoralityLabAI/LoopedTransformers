# RAM-09 stage 0: observable Lean proof paths and bounded lookahead

**Status:** Retrospective, read-only use of the **frozen original RAM-08 TRAIN/DEV archive**. The prospective 12-theorem PEF test remains **unscored**. This is a separate branch stacked on RAM-08 and does not modify its checkpoints, preregistration, search or test scripts.

## Input and reproducibility

Original RAM-08 training run [43689874169](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43689874169), artifact ID 14152063191, archive ZIP SHA256 `a2ac363547858871c4e066c96bac67253dfeb7ff4e2a0050bb5886c8ca9edcfc`. The immutable `data/transitions.jsonl` SHA256 is `f92d5654fdb8fc1d4bd43e7fb4c0b1f03b7a27c1d7335dd86e6ec947b582542f`. The same 20,930 original compiler effects were independently checked in RAM-08. This research uses no fresh Lean evaluations, no PEF proof source, and no external reference proofs.

Original old-MathCan TRAIN has 57 theorem roots, 640 states, 18,658 observations; original DEV has five separate roots, 74 states, 2,272 observations.

## Observed proof graph

For each real original `(source_file,theorem,state_signature)`, use the source-visible action and the compiler-replayed original successor state to form a directed transition. Edges with closed goals point to terminal. Positive transitions whose successor was never collected are **censored**. Rejected or unchanged compiler actions are known failures, but an unobserved successful continuation never becomes a negative training label.

The shortest *observed* closure distance obeys the finite graph Bellman identity

\[
d_G(s)=\min_{a\,\text{with observed successor}}\bigl(1+d_G(T(s,a))\bigr),
\]
with terminal successor value zero and with absent paths assigned \(+\infty\) *inside the finite observed graph only*. Reverse BFS computes these distances exactly. A witness path records compiler-accepted effects; the original full-theorem Lean kernel-certificate requirement remains separate.

| Original split | States with observed path | Length 1 | Length 2 | Length 3 | Length 4 |
|---|---:|---:|---:|---:|---:|
| TRAIN (57 roots) | 213/640 | 134 | 56 | 16 | 7 |
| DEV (5 roots) | 35/74 | 14 | 13 | 7 | 1 |
| Total | 248/714 | 148 | 69 | 23 | 8 |

Original TRAIN graph: 347 closing candidate observations, 1,514 matched nonterminal transition edges, 1,648 censored positive successors; 33 roots with a known route. Original DEV graph: 29 closing observations, 207 matched edges, 190 censored successors; three roots with a known route. A missing path means only that this bounded *recorded* collection lacks one.

## Pilot A: full-network pairwise lookahead fine-tuning

The original 57 TRAIN roots were split deterministically using SHA256(`RAM09-GRAPH-20261115/` + `source_file:theorem`) modulo 5: 42 fit roots, 15 internal holdback roots. No goal-state SHA or original theorem root overlaps. Fit roots supply 174 reachable states and 694 pairs, each pairing the shortest observed completion action with a recorded Lean-rejected tactic. The internal holdback has 39 reachable states among 191. Original RAM-08 DEV has 35 reachable states and is reported only as reused, exploratory diagnostic data; it previously influenced RAM-08 checkpoint selection.

All three frozen original selected policies were *copied* into separate weights, then fine-tuned for 300 steps on 32 paired actions with AdamW LR 3e-4 and logistic ranking loss. Original checkpoints and RAM-08 selection never changed.

| Model | Internal holdback witnessed-first-action before | After |
|---|---:|---:|
| Looped32 seed29 | 26/39 | 19/39 |
| Looped64 seed43 | 18/39 | 8/39 |
| Flat64 seed29 | 26/39 | 20/39 |

Thus lower pairwise training loss **failed** to preserve previously useful first-action rankings. Looped64's old RAM-08 DEV useful-first-action count also fell 19/74 to 0/74 after fine-tuning. This is a negative result, with no theorem-level claim.

## Pilot G: bounded residual guard, chosen after observing Pilot A failure

Keep the original recurrent/flat trunk *fully frozen*. Train only a small linear residual head on the same 694 pairs and fixed fit split, 200 AdamW steps. The score is

\[
r_G(s,a)=r_0(s,a)+\alpha\tanh(h_\phi(z_0(s,a))),\qquad \alpha=0.35.
\]

For every action, \(|r_G-r_0|\le\alpha\). If the original winning score exceeds every competitor by more than \(2\alpha\), its first-place rank is preserved **for every possible residual-head weight**. This is an algebraic guard proven by subtracting the two bounded perturbations, and checked across all observed states by the code. This pilot was **posthoc**, not preregistered.

| Model | Internal holdback witnessed-first-action before | Guarded |
|---|---:|---:|
| Looped32 | 26/39 | 25/39 |
| Looped64 | 18/39 | 18/39 |
| Flat64 | 26/39 | 28/39 |

Old RAM-08 DEV witness-first-actions were Looped32 23/35 to 23/35, Looped64 15/35 to 11/35, Flat64 24/35 to 25/35. The small Flat64 gain and Looped64 regression offer **no recurrent capability advantage**. The guard is a rank-stability certificate, not a theorem-proof certificate.

## Reproduce

Use Python with PyTorch CPU and the existing checked-out RAM-08 source code:

```bash
python experiments/lean_ram09/offline_graph.py ram08-frozen-audited-train-43689874169.zip --output out/ram09_graph.json
python experiments/lean_ram09/offline_fit.py ram08-frozen-audited-train-43689874169.zip --out out/ram09_pairwise
python experiments/lean_ram09/guarded_probe.py ram08-frozen-audited-train-43689874169.zip --out out/ram09_guard
python -m unittest discover -s experiments/lean_ram09 -p 'test_*.py'
```

All outputs require a new empty path. A separate RAM-09 prospective protocol can be registered after RAM-08 PEF has completed, using truly fresh theorem roots, a stronger control and native complete-proof certification. These retrospective 74 development states cannot constitute new prospective evaluation.
