# AF-05: frozen depth-stable causal abstraction result (2026-11-14)

**Registered scientific verdict: INCONCLUSIVE.**
**Engineering verdict: PASS**, 14/14 CI jobs, six independent per-case verifiers, 18 trained models and the cross-seed reconciler.

- Frozen scientific source: [`049e7b5`](https://github.com/MoralityLabAI/LoopedTransformers/commit/049e7b50d0c8f74854a1c52b506da9c651dde226), with CI trigger correction [`dbc3cca`](https://github.com/MoralityLabAI/LoopedTransformers/commit/dbc3ccac12a0263626be7466b62f0486b0a2b01b).
- Parent AF-04: [draft PR #43](https://github.com/MoralityLabAI/LoopedTransformers/pull/43); its negative depth-3 result is preserved unchanged.
- [Full primary CI run #43596117629](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43596117629), all 14 jobs successful.
- Registered [protocol](../../experiments/activation_funnel_af05/PROTOCOL.json), SHA-256 `c954b7087a185be26a16f30f94827dc2b9c3114061060579c6d94d8c00b67bce`. Dataset hashes are fixed in the [corpus manifest](../../experiments/activation_funnel_af05/corpus_manifest.json).

## Experimental question

AF-04's eligible causal seed17 lost 35.94 percentage points of informative goal-flip patch accuracy on depth-3 Horn graphs. AF-05 tests whether alternating upstream recurrence lengths and balanced depth-2/3 training, with a shared pre/post-stage Q consistency penalty, preserves causal code interchange into **unseen depth-4/5 proof graphs**.

The model reads only initial facts, graph incidence, rule stage/arity and node roles. It derives hidden Q0/Q1 proof state with the original source-pinned TypedLoop(16), then processes a downstream AND/OR goal using separate incidence. A matched donor changes one Q status. Every addressed intervention target is recomputed by exact Horn closure, independently of the student.

The model uses 15 atoms, 14 rules, 464 decoded working floats and either 29 or 58 carried floats. All task, status and causal arms share initial weights per seed, matched forward-call counts and 240 fixed steps. Each training minibatch samples 16 depth-2 and 16 depth-3 cases; upstream recurrence alternates four and six calls. Held-out depth-4 and depth-5 use eight and ten, respectively.

## Complete registered seed table

| Seed | Carried coords | Status dev goal/Q exact | Causal dev goal/Q exact | Depth4 informative patch: status/causal | Depth5 informative patch: status/causal | Verdict |
|---:|---:|---|---|---|---|---|
| 17 | 29 | .491/.277 | .509/.277 | .500/.500 | .500/.500 | Inconclusive (ineligible) |
| 29 | 29 | .491/.277 | .808/.513 | .500/.500 | .500/.500 | Inconclusive (ineligible) |
| 43 | 29 | .509/.277 | .491/.277 | .500/.500 | .500/.500 | Inconclusive (ineligible) |
| 17 | 58 | .830/.781 | .491/.277 | .406/.500 | .359/.500 | Inconclusive (ineligible) |
| 29 | 58 | .741/.562 | .504/.589 | .500/.500 | .500/.500 | Inconclusive (ineligible) |
| 43 | 58 | .813/.562 | .509/.277 | .641/.500 | .500/.500 | Inconclusive (ineligible) |

Qualification requires *both* the status and causal arms to reach >=.80 original goal and >=.80 joint-Q exact accuracy on **each** training and development split. All six pairs fail this requirement. The reported patch deltas are descriptive, and their apparent improvements cannot establish a causal-training advantage. This is an **inconclusive** primary judgment, not a scientific pass and not an eligible scientific failure.

### Held-out clean goal transfer at width2

| Seed | Depth4 status/causal | Depth5 status/causal |
|---:|---|---|
| 17 | .531/.500 | .500/.500 |
| 29 | .500/.500 | .500/.500 |
| 43 | .766/.500 | .781/.500 |

The depth-balanced patch objective did not create sufficiently accurate intermediate proof-state representation for any of the three width2 causal seeds at the registered 240-step budget. Seed17's status comparator narrowly misses Q eligibility at .781 and its own deeper task accuracy is also limited. The more competitive status comparator seed43 maintains useful clean-goal transfer, while its causal peer remains at chance-level goal accuracy on the held-out families.

This result motivates measuring the onset of representational collapse during optimization. A separately registered AF-06 could use status-to-causal warm starts, explicit depth-balanced *training* patch fidelity, and actual held-out depth and compute-matched control gates. Changing the AF-05 learning schedule or eligibility after viewing this matrix would invalidate its registration.

## Evidence integrity and mathematical scope

- **1,392** exact distinct canonical input graphs: 672 training, 224 development, 16 held composition, 224 held layout, 128 held depth4, 128 held depth5.
- **4,176** generated closed Boolean-Horn proof statements were compiled by pinned Lean 4.19.0, checking the oracle's graph-expanded truth assignments.
- **18** frozen trained models, **8,928** clean held-out prediction rows, and **17,856** donor-recipient counterfactual rows with real Q code and raw logits.
- All six seed-width evaluation jobs passed independent checkpoint and per-case label/mask/metric replay. Whole-code donor replacement numerically matches the direct donor's complete forward result.
- The separate cross-seed aggregator verified complete source/seed/width roster, source hash identity, verified audit receipt for each seed/width, and all registered gate judgments.
- The Lean certificates establish the synthetic Horn truth oracle. The experiment makes no claim of general Lean proof search or novel human-theorem proof generation.

Full original CI artifacts are available from [GitHub Actions run #43596117629](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43596117629). Scientific outputs remain immutable, including the ineligible runs and exploratory patch scores.

## Permanent evidence release

The original **14** CI ZIP artifacts are preserved byte-for-byte in the private-repository [AF-05 pinned evidence prerelease](https://github.com/MoralityLabAI/LoopedTransformers/releases/tag/af05-ci-evidence-dbc3cca-20261114), whose tag points to frozen scientific source commit `dbc3ccac12a0263626be7466b62f0486b0a2b01b`. The deterministic TAR is **2,068,480 bytes**, with SHA-256 `dfe2a58c1ba42d6ada71865f2409546c8c5127e36b037ed4d8de48c9775e8289`. GitHub's asset digest agrees with the archival packaging log, which verified all 14 original ZIP digests and their member-level checksums. [Evidence publication CI #43596828987](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43596828987) passed. This prerelease preserves the complete inconclusive verdict and all failed eligibility strata.
