# AF-13 asynchronous Horn-closure training | local scored research report

**Frozen local scientific verdict: FAIL.** The registered thresholds applied to the complete fixed local model matrix find an eligible failure for seed 17. Seed 29 fails preregistered eligibility in both comparison arms, so its deeper scores are descriptive. **Engineering: PASS locally** for both independent seed audits and the cross-seed scorer. **Hosted scientific verdict: unverified**, because GitHub Actions preflight failed before executing a single step.

- [Frozen scientific source commit](https://github.com/MoralityLabAI/LoopedTransformers/commit/57ac6503a2d323f15a51114eb6f9a97d2c6a3533); [draft AF-13 PR #66](https://github.com/MoralityLabAI/LoopedTransformers/pull/66) stacked on AF-12 PR #63, draft/unmerged.
- [Preregistered AF13 PROTOCOL.json](https://github.com/MoralityLabAI/LoopedTransformers/blob/feat/activation-funnel-af13-asynchronous-closure/experiments/activation_funnel_af13/PROTOCOL.json), SHA-256 `aa3e916a2457c6c31b363730bf5d65dfda10f6af3b9b0015c6e4811f6e2e426d`.
- New physical layouts: 20–25, heldout feature-hash manifest SHA-256 `c5089714cc466cfa6d87e0a8627f857a5cd4b076bf6367762aa3e20db4561f77`. Train/development remain the original AF05 cases. Cases: train 672, dev 224, heldout 128 new depth3, 16 new composition, 256 new depth4, 256 new depth5.
- Original qualified AF06 source checkpoint ZIPs: seed17 SHA-256 `c8cf6c6c8264cf313e755c4d3008c26ae72cdd6ef20583fadc112bf02169ffdd`, seed29 SHA-256 `660b71acd5dc95bbfd200e55f206f609dcf6cc141ccdbb131d74bb7e5f0d4ebf`.
- **Local alternative runtime:** Python 3.13.5, Torch 2.10.0+cpu; the frozen hosted target is Python 3.12 with Torch 2.5.1+cpu. These results are a complete local test of the registered scientific gate, not a successful hosted replication.

## Question and intervention

AF12's rigid wave-by-wave closure supervision harmed long-depth Q inference in its eligible local seed. AF13 uses the original unchanged native TypedLoop and a separate upstream trace of each atom's class probabilities. It places the Horn oracle at the **final sampled horizon** and trains constraints on soundness and persistence throughout recurrence, without assigning each intermediate proof to a particular recurrence step.

All three arms share the same 160 AdamW updates per seed, original AF06 source weights, 16 depth2 and 16 depth3 cases per update, donor draw streams, and source-cell forward calls. For each training depth, the independent extra trace alternates between original `2*depth` and `2*depth+2` calls every update; all arms receive the same horizon and update order. The `terminal` arm receives the unchanged clean/donor causal-cut loss; `relay_final` adds terminal least-closure CE; `async_sound` adds fact persistence, unreachable-atom exclusion, probability monotonicity and one-step soft rule-support penalties. The additional loss/backpropagation work differs and is **not FLOP-matched**.

## Local training and eligibility

| Seed | Arm | Dev goal | Dev exact Q0/Q1 | Updates | Wall-time |
|---:|---|---:|---:|---:|---:|
| 17 | `terminal` | 1.000000 | 1.000000 | 160 | 18.4s |
| 17 | `relay_final` | 1.000000 | 1.000000 | 160 | 17.9s |
| 17 | `async_sound` | 1.000000 | 1.000000 | 160 | 26.4s |
| 29 | `terminal` | 0.834821 | 0.897321 | 160 | 19.1s |
| 29 | `relay_final` | 0.897321 | 1.000000 | 160 | 18.0s |
| 29 | `async_sound` | 0.794643 | 1.000000 | 160 | 26.5s |

Qualification requires `relay_final` and `async_sound` to each achieve >=.90 goal and exact Q accuracy on train **and** dev. Seed 17 qualifies; seed 29's goal accuracy falls below the gate in both compared arms.

## Prospective heldout depth accuracy

| Seed | Depth | Relay final Q | Async Q | Difference | Relay all-atom | Async all-atom | Relay Q at +2 calls | Async Q at +2 calls |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 4 | 1.00000 | 1.00000 | +0.00000 | 1.00000 | 0.96667 | 1.00000 | 0.76562 |
| 17 | 5 | 0.76562 | 0.56250 | -0.20312 | 0.98333 | 0.86667 | 1.00000 | 0.76562 |
| 29 | 4 | 0.14062 | 0.25000 | +0.10938 | 0.25000 | 0.26667 | 0.25000 | 0.25000 |
| 29 | 5 | 0.25000 | 0.25000 | +0.00000 | 0.53333 | 0.43333 | 0.06250 | 0.06250 |

For eligible seed17, depth4 joint Q remains 1.000 in both arms, so the registered >=0.05 gain is absent. Depth5 joint Q falls from 0.765625 to 0.5625 (−0.203125), and all-atom final accuracy falls from 0.983333 to 0.866667. Depth4 clean goal accuracy falls 1.000 to 0.9375 and false-positive patched goals increase from 0 to 16; depth5 goal accuracy declines from 0.875 to 0.703125, while informative patched goal accuracy declines 0.9375 to 0.59375. Irrelevant-patch goal retention at depth5 is 0.71875, below the registered 0.90 minimum.

Seed29's depth4 Q improves from 0.140625 to 0.25 (+0.109375), but both arms are development-ineligible and the all-atom gain is only +0.016667. Its depth5 exact Q remains 0.25 in both arms and the asynchronous all-atom final accuracy decreases from 0.533333 to 0.433333. These remain labeled exploratory.

### Recurrence-clock diagnostic

At unseen depth5, the eligible seed17 final-relay baseline achieves 0.765625 exact Q after **10** upstream calls and 1.000 after **12**. The asynchronous model achieves 0.5625 after 10 and 0.765625 after 12. Depth4 is different: seed17 asynchronous model reaches 1.000 Q after eight calls, but only 0.765625 after ten, showing that extra recurrence can erode an already correct representation.

This separates two observed effects: delayed inference may benefit from more native-cell calls, and unprotected longer recurrence may destabilize the inferred Q readout. A soft monotonicity penalty on atom probabilities has not enforced Q-level stutter invariance.

All measured arms have `fully_admissible_fraction = 0.0` on the scored depth4/depth5 trajectories under the stringent condition that **every** stage has no unreachable-atom false positives and never loses an initial fact. The asynchronous objective reduces some per-stage false-positive counts on seed17 but still leaves thousands of unsupported predictions, revealing a significant gap between soft penalties and discrete soundness.

## Independent verification

- 6 final `.pt` checkpoints and matching complete named `.npz` exports, plus 2 original source checkpoint receipts.
- 3,936 heldout clean rows, each with all raw atom logits over original upstream recurrence steps and an independent +2-call Q/goal readout.
- 7,872 donor counterfactual records, with exact composed Horn oracle labels, Q-code latch checks, original source identity, goal logits and negative-control metrics.
- Both independent validators reconstruct the *entire* roster of source oracle truths directly from original rule bytes, check stage-wise fact and unreachable-atom error counts, recompute all scored metrics, and replay 72 sampled clean/donor forward calls per seed from saved model weights: **144 replays passed**.
- Cross-seed aggregator applies the frozen registered gates and reports `registered_verdict=fail`, `engineering_pass=true`; full `results.json` SHA-256 `ad80f70faa8eceeb2a6d66a081bb66a5e6eecd85de0c499d258c9dab9bed8a3b`.
- Python unit suite: 10 tests passed under normal Python and 10 under `python -O`. The smoke includes a rehashed forged-logit rejection.
- [GitHub Actions run #43695201112](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43695201112) failed at preflight with **zero reported executed steps**. Both subsequent hosted Lean 4.19.0 certification and Torch 2.5.1 scoring were skipped. The general asynchronous fairness theorem is a mathematical proof argument in the README; the Lean source was committed but its kernel compilation is **not verified by this run**.

The physical layouts are prospective new permutations of the same finite Boolean Horn templates. They test recurrence and graph-layout transfer, not new logical connectives or independent Lean proof search. The original AF01–12 branches and negative/qualified results were untouched.
