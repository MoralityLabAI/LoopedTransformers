# COE-07R: Native-output reversal-gauge audit

**2026-11-15.** Post-hoc computation on the exact, previously committed LoopScope-02 output-correctness traces. This is an additional diagnostic under COE-07's sealed benchmark and checkpoint pins. It is **not** the preregistered learned-activation-probe outcome.

## Source and scope

- Repository: `MoralityLabAI/LoopedTransformers`.
- Checkpoint: `gaoxin492/TinyRecursiveModels-Sudoku-Extreme-att`, `step_21700`, SHA-256 `932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3`.
- Original native data: `mechinterp/loopscope02/results/trm_results.json`, Git blob SHA `3c35daeb71da32f5c9ebbf03da539b8ee718b0ee`; original 256-board benchmark Git blob SHA `01ec63b4b2a4ae5399f783c6036a7fc018825a87`, benchmark SHA `59459ae0bb60b75db5be017e5f41adead391f2abb0b8fb043cd331c0ee043811`.
- For each target cell and each ACT step, the observed bit is `1` precisely if the model's predicted digit equals the **known correct answer**. These bits are an oracle that depends on target truth. The readout is unsuitable for deployment without access to those answers.
- Every group has 64 Sudoku boards. Pair counts include pairs with **different** correct digits and exclude same-digit pairs, leading to unequal denominators across strata. The benchmark was fixed before these post-hoc analyses.

## Exact group-action decomposition

For an ordered T-bit history `W`, let `N(W)` be its cyclic necklace, `B(W)` its dihedral bracelet (allowing rotation and reversal), and `C(W)` its popcount. On any fixed evaluator edge set, the retained-pair count obeys

```
D(W) >= D(N) >= D(B) >= D(C).
D(W)-D(C) = [D(W)-D(N)] + [D(N)-D(B)] + [D(B)-D(C)].
```

Thus the information lost when the start clock is hidden, the additional loss when forward/backward time is ambiguous, and the order-shape information retained despite those symmetries can be priced separately. The companion MathCan note proves the formula for arbitrary repeated task labels.

## Replayed pretrained observations: stalled stratum S4

There are **1,795** distinct-digit evaluator pairs in the 64-board S4 group. The original checkpoint and dataset digest were independently checked using the linked GitHub connector before reporting the following values.

| ACT visits | Ordered | Necklace | Bracelet | Count | Unknown clock loss | Unknown orientation loss | Dihedral order gain |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1,278 | 1,244 | 1,244 | 1,244 | 34 | 0 | 0 |
| 8 | 1,414 | 1,389 | 1,389 | 1,369 | 25 | 0 | **20** |
| 16 | 1,447 | 1,428 | 1,428 | 1,422 | 19 | 0 | **6** |

For the **same** 64 labeled boards, assigning each observed descriptor to its most frequent target digit gives the following in-sample, truth-dependent *decoder oracle upper-bound scores*; these are intentionally not held-out learning metrics:

| ACT visits | Ordered | Necklace | Bracelet | Count |
|---:|---:|---:|---:|---:|
| 4 | 23/64 | 18/64 | 18/64 | 18/64 |
| 8 | 30/64 | 24/64 | 24/64 | 19/64 |
| 16 | 31/64 | 26/64 | 26/64 | 24/64 |

The other three groups of 64 each have zero bracelet-beyond-count gain for these windows; S1 and S2 are uniform correctness traces in the replay. S3 has 446 retained different-digit pairs under each of the four descriptors, independent of which window was chosen.

**Empirical boundary:** the eight-step, 20-pair gain survives unknown start time **and** unknown time direction in this particular oracle descriptor. The underlying probe is defined by the answer, so its performance cannot establish an unsupervised, externally observable or causally active dark-fiber mechanism. The native latent run of `mechinterp/phase_gauge07/run_native.py` is required for that question.

## Independent finite positive control

At 8 visits and Hamming weight 4 there are exactly 10 binary necklaces and 8 binary bracelets. Under time reversal, 6 of the 10 necklaces are fixed, hence Burnside's formula gives `(10+6)/2=8`. The words `0b00001011` and its reversal have equal bracelets and counts but distinct cyclic necklaces. The reference verifies reversals, rotations, nested pair losses and the weighted-label decoder on exhaustive small inputs.

## Reproduction and CI

```
python -m unittest discover -s mechinterp/phase_gauge07 -p 'test_*.py' -v
python -m mechinterp.phase_gauge07.dihedral_replay --out /tmp/coe07r_gauge.json
```

The first command works with NumPy and no model checkpoint. Its source-file-dependent test runs inside a full repository checkout. The second command also checks the **Git blob SHA** of the two original native receipts. Both are pure CPU and use only standard-library arithmetic for the new code.

**Hosted inference status:** On the COE-07 PR branch, GitHub Actions run `43699655642` failed in its preliminary `exact-unit` job before exposed step logs and skipped `native-pretrained`. Its second attempt failed identically. A successful native activation-probe result is therefore pending; no synthetic or truth-conditioned substitute is promoted into that role.
