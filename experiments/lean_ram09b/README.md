# RAM-09B | Family-balanced proof-path ranking (retrospective)

**November 15, 2026.** [Protocol](PROTOCOL.md) was committed at **`4d819d26`** before RAM-09B CPU fitting. This study is stacked on draft RAM-09 PR #69 and inherits the **original older MathCan** training artifact from draft RAM-08 PR #61. All 120 new PEF theorem-policy searches remain **unscored** due to hosted runner assignment failures and missing Lean 4.35 in the current sandbox.

RAM-09B is explicitly exploratory. Its 15-root inner holdback was already examined during RAM-09 stage 0, and **all 15 roots were already in RAM-08's original training corpus** before the new residual heads were fit. Thus these roots test only the new residual head's reuse behavior, never end-to-end unseen theorem generalization.

## Motivation and registered method

RAM-09's previous pairwise pilot had 694 pairs, **487/694 negative `apply_local` actions** and only three same-family pairs. RAM-09B trains the previously frozen selected models with a **new frozen-trunk linear residual head**, using the shortest independently observed proof-closure route when one exists.

At every optimizer update, the algorithm samples a positive tactic family uniformly, then a proven reachable state within that family uniformly. It selects a genuinely compiler-rejected tactic family uniformly from that state's observed rejected families and chooses that family's strongest original-scored rejection as the hard negative. Positive compiler effects with unobserved future closure are **censored**. A negative action is never synthesized.

Each family-balanced CPU model uses **200 AdamW updates**, 32 paired examples/update, LR .003, seed 20261115, regularizer .02, correction cap .35. Model selection never uses the inner holdback or old DEV. The original trunks and selected checkpoint bytes remain frozen.

The output is a bounded scoring correction

\[
r_B(s,a)=r_0(s,a)+\alpha\tanh(w^\top z_0(s,a)+b),\qquad\alpha=0.35.
\]

## The fixed study failed its target

Full score receipt: [balanced_first_cpu_20261115.json](receipts/balanced_first_cpu_20261115.json).

| Original frozen model | Old 39 reachable inner-holdback states | RAM-09B balanced residual | Change |
|---|---:|---:|---:|
| Looped32 seed29 | 26 | 25 | −1 |
| Looped64 seed43 | 18 | 18 | 0 |
| Flat64 seed29 | 26 | 25 | −1 |

**Registered descriptive target: failed.** No model improves under the original witness-first-action metric. Observed first-action utility without a known full path did increase for some policies (Looped32 71/191 to 89/191), showing that immediate compiler progress is an inadequate substitute for certified closure-route selection.

Negative family sampling became broader: for Looped32, `apply_local` accounted for **497 of 6,400** sampled negative pairs, compared with the 487/694 imbalance in the earlier stage-0 program. The initial hypothesis that this class imbalance explained most of the closure-path regression is therefore unsupported by the new negative result. Other explanations, including representational capacity, partial path censoring, and the score perturbation cap remain testable.

## Post-score bound: what can a residual correction possibly fix?

After the preregistered fit's negative result, an independent **explanatory** bound analysis used the original frozen scores. It makes no training selection decisions.

For a state \(s\), let \(W_s\) be the set of actions with a finite *observed* route to closure. Write

\[
w_s=\max_{a\in W_s}r_0(s,a),\quad
n_s=\max_{a\notin W_s}r_0(s,a).
\]

For every score correction \(|\delta(s,a)|\le \alpha\):

* If \(n_s-w_s\ge 2\alpha\), no admissible strict-bounded correction can make an observed witness outrank every nonwitness. It is a **certified unflippable failure** of the bounded ranking class for that state.
* If \(w_s-n_s>2\alpha\), every admissible correction keeps a witness ahead of all nonwitnesses.
* If \(n_s-w_s<2\alpha\), selecting a witness is **possible** with independent per-action corrections, since the witness can be raised by nearly \(\alpha\) and the strongest nonwitness lowered by nearly \(\alpha\). A shared residual head may fail to achieve this independent-action upper bound.

The proof follows directly from pairwise score subtraction:
\(r'_0(s,n)-r'_0(s,w)\ge n_s-w_s-2\alpha\).

Full audit: [bounded_rank_feasibility_20261115.json](receipts/bounded_rank_feasibility_20261115.json).

| Model | Observed witness originally | Absolute maximum under ±0.35 | Unflippable nonwitness winners |
|---|---:|---:|---:|
| Looped32 | 26/39 | 38/39 | 1 |
| Looped64 | 18/39 | 30/39 | 9 |
| Flat64 | 26/39 | 35/39 | 4 |

This is an upper bound under a *more permissive independent-per-action* correction class. It does not guarantee any learned, shared-parameter correction can approach the ceiling. It is a scientific explanation of why Looped64 may require a larger adaptation or better original representation.

## Reproduce from original immutable RAM-08 archive

The exact RAM-08 artifact is `ram08-frozen-audited-train-43689874169.zip`, ZIP SHA256 `a2ac363547858871c4e066c96bac67253dfeb7ff4e2a0050bb5886c8ca9edcfc`. It contains 20,930 original independently replayed Lean compiler observations with data SHA256 `f92d5654fdb8fc1d4bd43e7fb4c0b1f03b7a27c1d7335dd86e6ec947b582542f`.

From the full LoopedTransformers source checkout, using Python 3.11+ and PyTorch CPU:

```bash
python -m unittest discover -s experiments/lean_ram09b -p 'test_*.py' -v
python experiments/lean_ram09b/ram09b_balanced_cpu.py ram08-frozen-audited-train-43689874169.zip --out out/ram09b_first
python experiments/lean_ram09b/ram09b_bounds.py ram08-frozen-audited-train-43689874169.zip --out out/ram09b_rank_bounds.json
```

A newly empty output destination is mandatory for the balanced run. All original RAM-08 checkpoint bytes are verified before scoring; the only learned RAM-09B parameters are the three small residual heads. Tests cover graph censoring, sampling by family, score-margin invariance, and strict feasibility bounds. Original sandbox verification: **12/12 source-level Python tests passed** across RAM-09 and RAM-09B; the six RAM-09B-specific tests also passed. GitHub-hosted CI validation remains pending due to the runner blocker.

## Next experiment boundary

The evidence favors collecting **new genuine MathCan tactic successors and fully kernel-certified proof searches** before additional retrospective hyperparameter tuning. A fresh future program should measure `exact` and rewriting under shared source-argument grammar and theorem-family disjointness, compare strong symbolic shape-frequency search, and retain a flat-model control. The old 15-root holdback and five RAM-08 DEV roots should not be presented as unseen generalization.

**Status:** draft, separate branch; preserve all negative data. No PR merge, production deployment or paid compute.

### Windows CPU-only runner (existing RTX 3050 machine)

With Python and PyTorch already installed, run from the full RAM-09B branch checkout:

```powershell
powershell -ExecutionPolicy Bypass -File experiments/lean_ram09b/run_windows.ps1 -Archive "D:/Research Engine/RAM-08/ram08-frozen-audited-train-43689874169.zip" -Out "D:/Research Engine/RAM-09B/first-balanced-run" -Python "D:/Research Engine/ram03-env/Scripts/python.exe"
```

Use the **original byte-identical Actions ZIP**, rather than a recompressed copy; its SHA-256 is verified before fit. The output directory must be absent. The script performs the six RAM-09B-specific unit tests, all three fixed CPU-head fits, the separate rank-bound analysis and an immutable run manifest. This does not run Lean or score any new PEF theorem.
