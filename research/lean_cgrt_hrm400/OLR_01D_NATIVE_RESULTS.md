# OLR-01D: native multi-goal probe scheduling and rollback

**Date:** 2026-11-23  
**Status:** Native verifier and architecture-interface pilot completed; **no trained feedback-controller gain claimed**.  
**Prescoring registration:** [OLR_01D_REGISTRATION.md](OLR_01D_REGISTRATION.md), commit `be4c6b5`.  
**Native CI:** [GitHub Actions #45180581378](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45180581378), green; [CPU regressions #45180581315](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45180581315), green.  
**Full native evidence:** [olr-01d-native-delayed-guards, artifact 14818749899](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45180581378/artifacts/14818749899). ZIP SHA-256 `e7b1677f6ef8b13bf209706176899aee875d655a519916f232c2448ec1eb4a0b`.

## Experiment design

The proof root is

```lean
∀ (P Q R : Prop) (h0 : P) (h1 : Q)
  (r0 : (a + b = c0) → P → Q → R)
  (r1 : (a + b = c1) → P → Q → R), R
```

A native `apply rN` opens **three pending Lean goals**: an arithmetic equality, `P`, and `Q`. The paired world swaps `c0` and `c1`, so the valid opening reverses while the allowed root menu `[apply r0, apply r1]`, numerical source of evidence and candidate budget remain fixed. The stored 01C neural checkpoints see only the public guard numerals. They choose one opening before the 01D native outcomes are loaded.

Two deterministic host search controllers then solve subgoals:

- **Probe early:** ask Lean `rfl` on the guard immediately. On rejection, restore the original immutable Lean state and try the other root rule if at least four tactic checks remain. A rejected `rfl` is a failed tactic attempt, never a logical refutation. On guard success, discharge the still-open `P` and `Q` goals with `apply h0` and `apply h1`.
- **Probe late:** first move the `P` and `Q` goals to the front using real Lean `focus` host operations, discharge them and then ask `rfl` on the guard. If the guard fails after four checks, a six-check budget cannot accommodate the four-check alternative proof.

Tactic checks are priced at one per `apply` or `rfl`; Lean `focus` and the host rollback are measured separately. No feedback weight update occurs. The controller's behavior after `rfl` is a fixed verified-search algorithm.

### Frozen evaluation custody

The cohort has **eight new pairs and 16 worlds**, seeded 20261127 from the larger arithmetic operand range. All source, dev and OOD arithmetic triplets of the OLR-01C bank are excluded from allocation. The 01C source model and optimizer state were not retrained for 01D.

- **32 forced native opening tests** verified exactly one reachable choice per world before scoring.
- **16 independent standalone native-reference proofs** compiled with an empty axiom audit.
- The `public.jsonl` registry and private witness registry were separately hashed. The fit source manifest and the loaded 01C checkpoint were hash-checked.
- All **272 root model/baseline decisions** were computed from the public packet before scoring. Their frozen `frozen_choices.jsonl` file was written before the policy trajectories were executed.
- The 4/6-call and early/late experiments consist of **128 distinct native search trajectories** over the two forced openings, independently replayed from fresh Lean state. Every scorer reuses the authenticated result of the identical root/action/schedule/budget combination, producing 1,088 policy-scored observations.
- **32 distinct successful theorem-and-branch/schedule proof paths** compiled in separate Lean 4.19 declarations with no axioms. The native focus events are replayed exactly. Standalone scripts use a **canonical guard-first goal order** for the same certified winning branch, because the host's `focus(1)` is a swap of the selected goal to the front and differs from the Lean `rotate_left 1` tactic on three goals.

## Results at a six-tactic budget

All entries count native Lean kernel-closed theorems out of 16 worlds. Call cost includes successful and rejected tactics; focus/rollback are separate.

| Frozen initial policy | Probe early certified | Probe late certified | Probe early tactic calls | Probe late tactic calls |
| --- | ---: | ---: | ---: | ---: |
| Exact arithmetic host oracle | 16 | 16 | 64 | 64 |
| Fixed `r0` | 16 | 8 | 80 | 64 |
| Cross recurrence, seed 17 | 16 | 8 | 80 | 64 |
| Cross recurrence, seed 29 | 16 | 9 | 78 | 64 |
| Cross recurrence, seed 43 | 16 | 16 | 64 | 64 |
| One-pass cross, seed 43 | 16 | 8 | 80 | 64 |

With the same 16 paired worlds, the early probe rescues **eight proofs** for the fixed-r0 policy over probing late. It pays **16 additional tactic calls** because each wrong opening needs an initial `apply`, a rejected `rfl`, and a full four-step restart. At the four-call cap, every arm's proof completion is determined by the first root choice: there is no budget for recovery.

The exact host arithmetic reference closes every world at four checks without speculative tool feedback. It is the dominant algorithmic baseline on these visible equalities. The early-probe controller reaches equal proof coverage at six checks but spends more calls when the initial model guesses wrong.

## Four-call initial ranking transfer and seed sensitivity

| Frozen model | Seed 17 proofs / 16 | Seed 29 / 16 | Seed 43 / 16 | Both-world complete pairs across seeds / 24 |
| --- | ---: | ---: | ---: | ---: |
| 3-pass cross | 8 | 9 | **16** | **16** |
| 1-pass cross | 8 | 8 | 8 | 9 |
| Direct reader | 7 | 8 | 8 | 9 |
| Isolated recurrent reader | 7 | 8 | 8 | 9 |
| Frozen recurrent body | 9 | 8 | 8 | 12 |
| Exact arithmetic host oracle | 16 | 16 | 16 | 24 |
| Fixed r0 | 8 | 8 | 8 | 0 |

The seed-43 three-pass checkpoint selected both valid root tactics in all eight paired worlds, providing a useful out-of-training-structure result on this *toy arithmetic family*. Seeds 17 and 29 were much weaker. These are repeated seed evaluations on the same eight independent pairs. The effect is not a direction-specific learned-obligation or message-mechanism result, especially given the earlier 01C margin-matched sham and the exact host solver.

## Verification correction retained in the audit

The first native workflow for 01D [#45180060529](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45180060529) failed its standalone compilation check: 16 of 32 scripts carried `sorryAx`. The live Lean proof search and event replay had reached kernel-closed states, but the exporter incorrectly equated the host's `focus(1)` permutation with `rotate_left 1`. A three-goal focus moves goal 1 to the front; Lean's rotate tactic shifts the entire goal sequence.

Commit `031a24d` corrected the audit contract. It records host focus separately, verifies each focus through an independent native event replay, and then compiles a canonical equivalent proof of the same theorem using the same successful opening rule. The corrected CI built all 32 canonical declarations with an empty axiom set and passed the full PPT regression suite. The first failed run remains a visible negative engineering receipt.

## Identity and reproducibility receipts

| Artifact | SHA-256 |
| --- | --- |
| 01D registered manifest | `63c400cf5eb4cf4132252514ded940d2adcc511d6a984e33021cacf1943c96f1` |
| Frozen source model checkpoint | `97526c4697fc14239dc4ccaf56ea26485e5e07c9821dd98fe9ada5c18199c1c6` |
| 01D frozen model-root choices | `d9e0914d2e932a6511e09211dd5d6d77ee8beb8b1ca0c2517660f4a231eadc44` |
| Native search trajectories | `eae2f12eebc9d319f64f49965064107fc52f5546611be1b205c5df7bfcc31d8a` |
| Policy score rows | `89e7e4fe1a44a457410bc93122631002662bfd2d188a22acfcad1362189bb4dd` |
| Compiled standalone Lean source | `c1ba196bc666ecf7a88e5ad35a97c6dd3a1ca559a4893f143d11b49a1763c223` |

Toolchain: pinned Lean 4.19.0; Lean binary SHA-256 `92c3d35b5bfaa5e0fea413a775d504cf46cd95e1345df61c2274f76779e7e023`. The full source bank, private witness bank, replay events and standalone source are in the [uploaded CI artifact](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45180581378/artifacts/14818749899).

## Reproduction

From `research/lean_cgrt_hrm400` after preparing the OLR-01C verified source bank and frozen models:

~~~sh
python -m ppt_gym.olr_delayed_campaign freeze --source-bank runs/olr-01c-bank --seed 20261127 --pairs 8 --out runs/olr-01d-bank
python -m ppt_gym.olr_delayed_campaign score --registry runs/olr-01d-bank --source-bank runs/olr-01c-bank --source-fit runs/olr-01c-fit --out runs/olr-01d-score --max-seconds 500
python -m unittest discover -s ppt_gym/tests -p "test_olr_delayed.py" -v
~~~

The source bank and model inputs originate in [OLR_01C_GUARD_LEARNING.md](OLR_01C_GUARD_LEARNING.md). GitHub Actions runs the complete pinned sequence and uploads receipts.

## Next scientific gate

This experiment establishes live pending-goal and budgeted rollback correctness. The next model experiment must *learn* when to spend a verifier check, when to preserve an obligation, and which branch to restore, on new computational or dependent-type cases with feedback that changes the optimal choice. It must compete with the exact symbolic evaluator and a direct supervised reader at matched inference and native-check costs. A fresh prospective cohort, matched-message shams, and EvalMath FSI-01 power criteria are still prerequisites for an OLR mechanism claim.
