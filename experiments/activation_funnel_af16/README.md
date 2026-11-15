# AF-16: transfer of a host-sound rule-priority policy to new Horn compositions

AF-15 showed seed-sensitive frontier-ranking gains under a host-owned legal-rule gate. AF-16 keeps the exact same 5,234-parameter TypedHornPolicy and introduces genuinely unseen **cross-branch Horn dependencies**. It trains four independently initialized seeds, maintains AF-15's shortest-proof labels, and includes a stronger imitation control that receives one-third more optimizer updates.

## New graph families

Within 15 atom slots and 14 rule slots, the training additions provide swapped support and alternative short-detour rules. The primary holdouts use an explicit dependency from Q0 to Q1, a join requiring both branch relays, and a direct goal shortcut that bypasses some subgoals. These arrangements require rule selection beyond a physical node permutation. Rule incidence, current proof facts and public role tags are the only inputs to the model.

The train/dev splits comprise 928/352 cases including AF-15's unchanged 672/224 original cases. Novel-layout diagnostics contain 64 inputs; new logical-composition depths 4 and 5 contain 192 inputs each, with disjoint physical layouts. The cases and feature hashes are preregistered in [fresh_manifest.json](fresh_manifest.json).

## Safety boundary

Every model output is passed through the same host eligibility mask. A rule may fire only when its publicly declared premises have been proved and its head is absent. The host writes the resulting proof state and each exact edge. Neural model logits have no authority to assert proof correctness. On unprovable problems the model may safely STOP. Lean's [RulePriority.lean](RulePriority.lean) establishes abstract Boolean soundness and inflationarity, independently of any trained policy result.

The scientific metric is reachable-goal success under **exact shortest proof steps plus one firing**. The positive target comprises all optimal available rules; a physically invalid action never earns proof credit. The source and recurrence architecture are unchanged from AF-15; this is an extended structured Horn proof-search gym, not an unrestricted Lean tactic reasoner.

## Registered fitting and audit

Seed roster: 17, 29, 43, 61. Imitation and frontier receive 240 fixed steps with identical initial parameters and sampled minibatches; `imitation_extra` gets 320 steps and identical first 240 batches as a compute-advantaged comparison. The frontier loss adds a legal pairwise margin to exact shortest-path imitation. Both policy arms are matched on forward calls. The frontiers have extra backward arithmetic, and measured wall times are included in receipts.

The independent scorer reconstructs graph edges directly from public incidence matrices, calculates a separate breadth-first shortest-path oracle, verifies every host proof edge and action logit, and samples checkpoint forward passes. It checks complete output rosters and both Torch/NumPy weight exports. A tamper test changes an action trace and recomputes its file hash to test semantic replay.

## Reproduction

From the repository root with Torch and NumPy installed:

```bash
python -m unittest discover -s experiments/activation_funnel_af16/tests -v
python -O -m unittest discover -s experiments/activation_funnel_af16/tests -v
python -m experiments.activation_funnel_af16.experiment manifest --out /tmp/af16-manifest.json
python -m experiments.activation_funnel_af16.experiment train --seed 17 --out artifacts/af16/s17/training
python -m experiments.activation_funnel_af16.experiment evaluate --seed 17 --training artifacts/af16/s17/training --out artifacts/af16/s17/evaluation
python -m experiments.activation_funnel_af16.verify --seed 17 --training artifacts/af16/s17/training --evaluation artifacts/af16/s17/evaluation --out artifacts/af16/s17/evaluation/independent_audit.json
```

Run all four seeds and then `python -m experiments.activation_funnel_af16.aggregate --root artifacts/af16 --out artifacts/af16/results.json`. The default CI protocol also attempts checksum-pinned Lean 4.19.0, but a hosted Actions failure before runner execution is reported separately from this local audit.

The experiment is stacked on AF-15 and leaves AF-01 through AF-15 and their negative findings unchanged. Holdout scores are evaluated only after the four seed checkpoints have been frozen.
