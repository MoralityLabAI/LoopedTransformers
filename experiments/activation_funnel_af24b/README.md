# AF-24B: frozen AF-23 TypedLoop transfer to unseen Horn grammars

This experiment measures whether the original AF-23 eight-seed 6,036-parameter recurrent checkpoints carry useful **inference-time proof-action heuristics** into new public Horn structures. It modifies neither a checkpoint nor any earlier experiment's registered verdict.

The [original AF-23 CI](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43812708916) passed its weak fixed/random controls. [AF-24](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43818115192) found a deterministic public-graph closure controller that solved that original grammar perfectly. AF-24B tests separate nonisomorphic graph families and compares against the same stronger symbolic controls.

## Preregistration and structural test

[PROTOCOL.json](PROTOCOL.json) was committed first at `7e845ad`, before the generator or new scores were constructed. New cases are split into two source-registered cohorts of 192 each, with 80 provable and 112 unprovable goals. Provable cases require exactly five or six legal firings.

Each cohort consists of three grammar families:

- **shared_overcount:** two subgoals share a prerequisite chain. A physically preferred, longer direct-goal path causes the earlier additive-cost closure heuristic to overcount shared work and select the wrong derivation.
- **or_goal_competition:** two individually sufficient subgoal paths have different proof costs and jointly replace the original AF-23 goal conjunction.
- **crossed_conjunction:** the optimal goal proof crosses prerequisite branches, while a longer direct-goal derivation competes.

All families have two goal-concluding rules, unlike AF-23's original grammar. Physical labels and rule slots are independently shuffled on held-out layouts 300–305 and 310–315. Positive cases begin with two or three root facts, making the first-fact distribution itself out of the original AF-23 training range; this covariate shift must be acknowledged when interpreting transfer.

The **source-only structural gate** requires independent bitset and host BFS to agree, exact expected proof depth, at least two legal root rules with at least one suboptimal goal-ancestor action, and a pooled additive-closure completion rate at or below 75%. This was checked before model scoring in [preflight CI #43822626726](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43822626726), which passed.

| Cohort | New provable / all | Shared-overcount closure solved | Total closure solved |
|---|---:|---:|---:|
| Held depth 5 | 80 / 192 | 0/24 | 56/80 |
| Held depth 6 | 80 / 192 | 0/24 | 56/80 |

The preflight manifest is a **structural** result about the generator and fixed symbolic controller. It does not provide any evidence of neural transfer.

## Frozen neural and symbolic arms

One original AF-23 final checkpoint per seed, eight fixed seeds. Three neural inference methods reuse its exact parameters:

- Direct legal-action logits
- VI-1 one-step successor neural value scoring
- VI-2 two-step min-plus backup scoring, reusing the unchanged AF-23 scorer

Symbolic controls are reused unchanged from AF-24: first-legal, goal-ancestor greedy, additive-cost closure, bounded public BFS with 16 or 64 node expansions per decision, and the exact shortest-proof oracle for an upper bound. The normal host controls action legality for every policy.

All methods receive the same public physical graph, known facts and shortest+1 rule-firing budget (two steps for unprovable goals). Actual neural forward-node calls, symbolic rule examinations and BFS expansions are reported separately.

### Registered scientific gate

The pre-registered target is **VI-2**, with no hindsight checkpoint or head selection. For a seed to pass at both depths, VI-2 must exceed the **best of the four named non-oracular graph-aware controls** by **at least eight of 80 proofs (10 percentage points)**, reach 50% absolute success, correctly STOP immediately on at least 90% of unprovable cases, and show zero false accepted physical Horn proofs.

**At least six of eight fixed seeds** must pass the full two-depth gate for `NEURAL_TRANSFER_PASS`. Any complete tested cohort failing that rule is `NEURAL_TRANSFER_FAIL`. Missing or corrupted source artifacts lead to `INCONCLUSIVE`.

The original AF-23 registered PASS and AF-24 symbolic match persist independently, whatever the new outcome.

## Reproducibility

- [gym.py](gym.py) constructs the source-registered graph families and independent BFS, with **no Torch import until observations**.
- [transfer_planner.py](transfer_planner.py) uses the original AF-23 min-plus scorer over new legal host successor states.
- [evaluate.py](evaluate.py) checks the eight original GitHub artifact ZIP SHA-256 digests, all Torch/NumPy weights and signed training receipts before score generation, then logs full proof episodes for all policies.
- [audit.py](audit.py) reconstructs the physical Horn graph from public **model-visible tensors**, independently checks every legal firing and terminal decision, recomputes the numeric backups, and samples genuine frozen-checkpoint Torch forwards.
- [aggregate.py](aggregate.py) evaluates the preregistered all-seed gate with per-family scores and independent digest provenance.

### Commands

Source-only (Python standard library):

```bash
python -m unittest discover -s experiments/activation_funnel_af24b/tests -v
python -m experiments.activation_funnel_af24b.gym > af24b_manifest.json
```

After downloading an original AF-23 seed ZIP as `af23-branch-seed-17.zip`, with the originally pinned `torch==2.5.1+cpu` and NumPy installed:

```bash
python -m experiments.activation_funnel_af24b.evaluate --seed 17 \
  --original-zip-dir original --out /tmp/af24b-17
python -m experiments.activation_funnel_af24b.audit --seed 17 \
  --original-zip-dir original --output-dir /tmp/af24b-17
```

The source-locked GitHub Actions workflow orchestrates eight separate seed runs, independent audits, and aggregation. Every failure receipt is retained. Keep this branch **draft and unmerged** for independent review.
