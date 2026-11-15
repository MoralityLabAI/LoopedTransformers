# AF-13: asynchronous Horn closure without a prescribed internal clock

AF-12's local experiment found that forcing the student to match every exact
synchronous closure wave could improve per-step atomic correctness yet damage
final Q0/Q1 correctness on deeper unseen graphs. AF-13 separately tests a
**temporally flexible**, causally grounded objective on prospectively unused
physical graph layouts 20–25.

All scientific thresholds and objective coefficients live in [PROTOCOL.json](PROTOCOL.json).
The parent source and scored evidence are unchanged. The original AF-06
qualified seed-17/29 warm-up checkpoints and exact original TypedLoop source
are still required. The experiment remains an inference task over finite
synthetic Boolean-Horn DAGs; Lean certifies the host's oracle, not the
student's learned theorem proving.

## Finite asynchronous closure theorem (mathematical proof)

Let \(U\) be a finite set, \(R\) a set of definite Horn clauses, and
\(L=\operatorname{lfp}(\Phi_R)\) above initial facts \(S_0\). A valid
asynchronous update chooses any subset \(E_t\) of enabled rule heads and sets
\(S_{t+1}=S_t\cup E_t\).

**Soundness.** All prefixes satisfy \(S_t\subseteq L\). At \(t=0\), this
follows from the definition of \(L\). If \(S_t\subseteq L\), any head enabled
by premises in \(S_t\) must also occur in the closed set \(L\); therefore
\(S_{t+1}\subseteq L\). This completes induction.

**Fair convergence.** As the finite increasing sequence can add at most
\(|U\setminus S_0|\) distinct atoms, it eventually stops growing, say at
\(S_\infty\). If a rule remains enabled with a missing head, fairness
requires it to fire at some later step, contradicting stabilization. Thus
\(S_\infty\) is closed, so by minimality of \(L\), \(L\subseteq S_\infty\).
Soundness gives the reverse inclusion. Consequently \(S_\infty=L\).

**Important limitation:** fairness imposes no uniform number of updates;
an arbitrarily long legal stutter can delay an enabled rule. A finite
training deadline is a separate empirical requirement.

[AsyncClosure.lean](AsyncClosure.lean) checks only the finite Boolean Horn
primitives by Lean's kernel. The general fairness theorem above is an
ordinary mathematical argument, rather than a Lean4-certified theorem.

## Differentiable asynchronous surrogate

The model emits membership probabilities \(p_t(v)\) for each atom after
successive recurrent calls. Training constrains their meaning using the
exact original host-produced least Horn closure \(L\), initial facts \(S_0\),
and the supplied graph. It does not require each \(p_t\) to match the exact
synchronous Boolean wave \(\Phi_R^t(S_0)\).

At the final sampled horizon, all-atom cross-entropy trains membership in
\(L\). Additional penalties encourage initial facts to stay present and
all unreachable atoms to stay absent throughout recurrence. Monotonicity
penalizes probability decreases. A soft support penalty prevents a
non-fact head from becoming more probable than the strongest conjunction
of its parents at the preceding step. This support penalty is a surrogate;
its satisfaction is not a proof of Boolean correctness.

Two fitting clock horizons are interleaved on the *same training examples*:
4/6 recurrent calls for depth2 and 6/8 for depth3. Every arm gets exactly
these calls in the same deterministic update order. This keeps recurrence
work matched, while the auxiliary losses require extra backward computation.

## Frozen protocol and controls

| Dimension | Value |
|---|---|
| Seeds | 17, 29 |
| Train / dev | 672 / 224 original AF-05 canonical inputs |
| Prospective test | 128 depth3, 16 composition, 256 depth4, 256 depth5 |
| Prospective layouts | 20–25, all unseen in AF01–12 |
| Carried state | 58 FP32 values, 29 graph tokens, working width16 |
| Fitting | 160 AdamW updates per arm, 32 cases/update |
| Arms | `terminal`, `relay_final`, `async_sound` |
| Heldout horizons | 2d and 2d+2 recurrent calls, evaluated separately |
| Negative controls | Irrelevant-Q patches, false-goal attempts, touched/untouched Q |

The three arms have identical initial weights and donor input streams per
seed. The registered hypothesis compares `async_sound` against `relay_final`.
Both controls are trained with jittered clocks. All heldout evidence is
withheld until the frozen checkpoints have been written. No old AF-12
heldout cases enter fitting or eligibility.

## Reproduction

From a repository checkout with Python 3.12, PyTorch 2.5.1 CPU,
NumPy and the exact AF-06 training artifacts:

```bash
python -m unittest discover -s experiments/activation_funnel_af13/tests -p 'test_*.py' -v
python -O -m unittest discover -s experiments/activation_funnel_af13/tests -p 'test_*.py' -v
python -m experiments.activation_funnel_af13.experiment train --seed 17 --source-zip af06-s17.zip --out artifacts/af13/s17/training
python -m experiments.activation_funnel_af13.experiment evaluate --seed 17 --train artifacts/af13/s17/training --out artifacts/af13/s17/evaluation
python -m experiments.activation_funnel_af13.verify --seed 17 --training artifacts/af13/s17/training --evaluation artifacts/af13/s17/evaluation --out artifacts/af13/s17/evaluation/independent_audit.json
```

Local runs with a different Torch version are separately labeled engineering
replications. Independent replays verify the complete clean and donor scoring
roster and extra-horizon outputs. The fixed primary scientific gate may fail
while all engineering tests pass.
