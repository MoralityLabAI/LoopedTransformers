# AF-04 scored research result | 2026-11-14

**Registered primary verdict: FAIL.**  
**Engineering verdict: PASS.** All 14 GitHub Actions jobs passed, including source/corpus preflight, Lean validation, all six seed-width training jobs, independent per-case replay, and cross-seed aggregation.

- Source: [frozen AF-04 commit 83dc51e](https://github.com/MoralityLabAI/LoopedTransformers/commit/83dc51ef204a16751bda5a9d93c2157c6103ce07)
- Protocol SHA-256: `c729ca4a99762515d4a72f456ec2ee5ed4db28496d7c52619c5935418dca1cf5`
- [Primary CI run 43593567030](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43593567030)
- [Draft PR #43](https://github.com/MoralityLabAI/LoopedTransformers/pull/43), stacked on AF-03 #40

## Scientific question and architecture

AF-04 asks whether addressed counterfactual intervention gradients provide more generalizable **inferred** intermediate proof state than equivalent-compute intermediate-status supervision. The experiment uses the original hash-pinned `TypedLoop` cell (5,024 recurrent-cell parameters) in a roughly 5.4k-parameter model. It reads initial facts, public rule incidence and role tags, never the truth values of derived nodes. After five upstream recurrent calls, the upstream premise edges are removed; three further calls process only the downstream Boolean Horn goal dependency. Swapping an inferred Q0/Q1 code with a matched donor has an exact independent least-fixed-point Horn oracle.

The 11 atom and 10 rule tokens have 336 working-state floats, compressed after each recurrent call to either 21 floats (one per token) or 42 floats (two per token). Every seed-width combination starts all task, status, and causal training arms from identical weights and inputs, with matched numbers of native-cell calls. Each arm trains for 240 fixed steps; fitting uses the 336 train cases, with 112 development cases for qualification. The three held-out families contain 16 novel-graph-composition cases, 112 new physical-layout cases and 128 longer-depth cases. Each case supplies two exact changed-status donor interventions.

## Complete frozen seed matrix

The primary comparison requires both status and causal arms to reach 80% training **and** development goal accuracy and exact Q0/Q1 accuracy. It then requires a causal improvement of at least 0.05 in informative-goal-flip patch accuracy on **both** held-out composition and depth, alongside clean goal and locality requirements.

| Seed | Carried floats | Development status goal/Q exact | Development causal goal/Q exact | New-composition informative goal patch (status / causal) | Depth-3 informative goal patch (status / causal) | Registered seed verdict |
|---:|---:|---|---|---|---|---|
| 17 | 21 | .491 / .277 | .491 / .277 | .250 / .250 | .500 / .500 | Inconclusive (ineligible) |
| 29 | 21 | .491 / .277 | .491 / .348 | .250 / .250 | .500 / .500 | Inconclusive (ineligible) |
| 43 | 21 | .795 / .277 | .509 / .420 | .750 / .750 | .500 / .500 | Inconclusive (ineligible) |
| 17 | 42 | **1.000 / 1.000** | **1.000 / 1.000** | **1.000 / 1.000** | **.938 / .578** | **Fail** |
| 29 | 42 | **1.000 / 1.000** | **1.000 / 1.000** | **1.000 / 1.000** | **.594 / .594** | **Fail** |
| 43 | 42 | .795 / .277 | .795 / .277 | .750 / .750 | .500 / .500 | Inconclusive (ineligible) |

The **registered width-2 aggregate fails**, because two eligible seeds fail the required improvement gate. At width 2 seed 17, the causal arm loses **.359375** informative goal-patch accuracy on longer proof chains relative to the status-supervised comparator. Its other-patch clean-goal retention on that family is **.625**, violating the explicit locality threshold. Seed 29 achieves perfect novel-composition patch accuracy in both arms, with no depth-3 patch improvement at .59375. The development success of two seeds demonstrates that the short-depth Horn state can be inferred; it is not evidence for a causal training advantage.

All three width-1 seed pairs fail the registered qualification threshold and are retained as inconclusive stress tests. The protocol prevents reinterpreting ineligible or saturated results as a mechanism advantage.

## Verification and scope

The full matrix contains **18 frozen trained models**, **4,608 canonical held-out clean prediction rows**, and **9,216 donor-recipient counterfactual rows**. Every evaluation job ran independent label reconstruction, score recalculation, saved-code checkpoint replay, source hash validation, and whole-code/direct-donor logit matching, all passing. Preflight compiled **2,112 generated Lean 4.19.0 Boolean-Horn theorem certificates** across all 704 unique data cases and their three targets.

The Lean certificates validate the synthetic oracle's closed Boolean formulas. They certify neither the neural model's predictions nor original Lean theorem proofs. AF-04 tests latent reasoning-state transport on a small structured Horn DAG under exact graph-dependent supervision. It does not establish general semantic mechanistic interpretability or independent proof discovery in language models.

## Mechanism follow-up

The observed depth-3 collapse on causal seed 17 suggests testing whether patch training overweights short-range state substitutions and damages longer compositional proof-state invariants. A subsequent **new registration** could introduce a depth-balanced intervention curriculum, explicit commuting closure across stages, and a control using deeper graph dependencies with matched forward calls. The existing AF-04 source, seed matrix and verdict remain immutable.

## Permanent verified evidence release

The full original 14-artifact GitHub Actions evidence has been preserved in the [private-repository AF-04 evidence prerelease](https://github.com/MoralityLabAI/LoopedTransformers/releases/tag/af04-ci-evidence-83dc51e-20261114), targeting the frozen scientific source commit `83dc51ef204a16751bda5a9d93c2157c6103ce07`. The deterministic TAR is **1,382,400 bytes** with SHA-256 `526d296f3c2a937f5f708e611cffb9ec518f7c1d2d94ef64f78668f93a2393d1`. GitHub's uploaded-asset digest independently matches the packaging job's checksum. Its signed manifest binds each original CI ZIP and member content. The [archive publication CI](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43594410927) passed. The initial artifact-download redirect attempt failed engineering authentication and was corrected in the publication workflow without changing the frozen model/training/evaluation code.
