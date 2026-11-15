# AF-10: exact proof-state causal cut | November 15, 2026

**Type of result:** source-bound mechanistic diagnostic and an exact structural theorem. The two frozen AF-09 seeds are reused without training. The AF-05 held-out cases have informed prior AF campaigns and the design of this mechanism. **Any accuracy changes on these cases are retrospective observations; there is no preregistered comparative scientific superiority claim.**

- Original AF-09 frozen model source: [`643fd90`](https://github.com/MoralityLabAI/LoopedTransformers/commit/643fd9085c4801960a228a16ddbe49a8a58dc750).
- AF-10 immutable diagnostic source: [`0b50e66`](https://github.com/MoralityLabAI/LoopedTransformers/commit/0b50e66b85d66f624a2d07ecc1409e9bef17bacd).
- [AF-10 causal-cut theorem](AF10_CAUSAL_CUT_THEOREM.md) and [Lean finite-coordinate proof](../../experiments/activation_funnel_af10/CausalCut.lean).
- [AF-10 draft PR #59](https://github.com/MoralityLabAI/LoopedTransformers/pull/59), stacked on AF-09 #54.

## Mechanism

The original native TypedLoop mixes both independent Q atom states through a shared downstream rule, so a selected Q donor can alter the neural representation of the untouched Q after recurrence. The original cell's Q0-to-Q1 code Jacobian was nonzero on a sampled, checksum-verified AF-09 example (seed17 L2 ≈0.00770; seed29 L2 ≈0.00191). This establishes an available information path, rather than a statistical estimate of its average strength.

AF-10 changes the downstream interface: after each original native-cell call, it restores the Q0/Q1 carried codes to their exact upstream-boundary values **before** the next decode. Readouts use the same original affine decoder and status head. The Lean-checked coordinate-cut theorem shows `P z_t = P z_0` for arbitrary recurrent updates, and the concrete Python tests demonstrate that the role-selected operation implements that projection. Exact protected-Q readout invariance and a zero cross-code Jacobian follow for the protected coordinates. The goal and rule state still update through the native cell.

A correct original recipient-protected Q and donor-selected Q at the **prefix** imply a correct composed Q pair after the addressed cut for every downstream step. Incorrect prefix Q truth persists under the cut, which separates semantic inference errors from transport errors. A goal prediction may still fail after a correctly transported Q pair.

## Frozen-model depth diagnostics

Values below are **protected-subgoal correctness after the addressed patch**, averaged across the 256 exact changed-status interventions in each depth family. Every row compares the original frozen AF-09 model with the same weights under the causal cut.

| Seed | Frozen AF-09 arm | Held depth | Original recurrent | Causal cut |
|---:|---|---:|---:|---:|
| 17 | status | 4 | .33594 | .62500 |
| 17 | status | 5 | .28516 | .87500 |
| 17 | locality | 4 | .32422 | .62500 |
| 17 | locality | 5 | .50000 | .68750 |
| 17 | signed margin | 4 | .38281 | .75000 |
| 17 | signed margin | 5 | .43750 | .50000 |
| 29 | status | 4 | .57813 | .50000 |
| 29 | status | 5 | .25000 | .75000 |
| 29 | locality | 4 | .50000 | .50000 |
| 29 | locality | 5 | .32813 | .75000 |
| 29 | signed margin | 4 | .71094 | .50000 |
| 29 | signed margin | 5 | .62500 | .75000 |

The cut's protected-subgoal correctness equals the *prefix protected-subgoal correctness* exactly in every family. Where the original recurrence corrected an initially wrong code, the cut can reduce final accuracy. The table includes those negative controls.

### The goal is a separate mechanism

The identical intervention roster produces these **patched-goal accuracy** measurements:

| Seed / model / held depth | Original recurrent | Causal cut |
|---|---:|---:|
| 17 / status / 5 | .40234 | .84375 |
| 17 / signed margin / 5 | .53906 | .43750 |
| 29 / locality / 4 | .42969 | .71875 |
| 29 / locality / 5 | .50000 | .59375 |
| 29 / signed margin / 4 | .57813 | .50000 |

The goal head was trained on ordinary recurrence and can depend on the feedback being removed. Exact code locality alone cannot promise correct Horn-goal computation.

A stronger decomposition uses only pairs whose two *required* prefix Q predictions agree with the independent Horn recipient/donor oracle. Among seed17's status model on depth5, this condition holds for 87.5% of the 256 donor pairs; the latched goal succeeds on 86.61% of that subset. For seed29's locality model at depth5, it holds for 18.75% of pairs, and the latched goal succeeds on 50.0% of the qualified subset. In the first case the upstream state is relatively reliable and there is still downstream goal error; in the second both mechanisms contribute.

## Integrity

Original AF-09 model checkpoint artifacts are verified against their original GitHub artifact ZIP SHA-256 digests and complete Torch/NumPy export rosters. The evaluator checks original Horn donor/recipient targets and exact role-selected code replacements. A separate checker reconstructs the protected and transplanted Q readouts from the saved two-float code, decoder and status-head matrices in NumPy float64.

The local audit covered **8,640 donor-intervention rows** across six original model checkpoints and five data families, with **180** independent sampled Torch forward replays. The maximum absolute NumPy affine readout disagreement was below `4e-7`. Every pair satisfied exact protected-code equality and the compositional prefix-state theorem. One test intentionally rewrites an output logit and recomputes its file hashes; the independent verifier still rejects the forged readout.

The local evidence run used Torch 2.10.0+cpu. GitHub Actions reruns the original-source diagnostic under Torch 2.5.1+cpu and Lean 4.19.0 to establish the hosted reproduction; the main run and separate aggregate repair are kept in the CI audit trail. The Lean theorem concerns a generic cut operator. Earlier Lean checks of the synthetic Horn oracle remain distinct from neural proof-state accuracy.

## Next controlled science question: causal boundary plus Boolean influence

Given `g=q0 AND q1`, toggling Qj changes the true goal precisely when `q(1-j)=1`. Given `g=q0 OR q1`, toggling Qj changes the goal precisely when `q(1-j)=0`. That exact Boolean influence is an interpretable training label for the downstream goal gate.

A prospective experiment can train the upstream Q inference and goal-gate influence on **new graph families**, keeping the latch active during fitting and holding new deeper dependency layouts out before architecture selection. It should report prefix Q competence and goal accuracy conditional on correct Q inputs separately, under matched compute and original theorem checking. This will measure whether structural causal isolation aids learned reasoning, rather than only post-hoc invariance of frozen models.