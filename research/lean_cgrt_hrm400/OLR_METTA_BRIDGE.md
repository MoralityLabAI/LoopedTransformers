# OLR-METTA: native MeTTa curation, STOP repair, and action-set supervision

**Date:** 2026-11-23  
**Status:** Implemented and tested on native Hyperon and native Lean; engineering and development evidence only.  
**Parent:** [OLR-01D](OLR_01D_NATIVE_RESULTS.md)  
**CI:** [Lean/Hyperon #45186782600](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45186782600), [CPU regression #45186782625](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45186782625): green.

## Recent DataGoblin evidence and source boundary

[DataGoblin PR #46](https://github.com/MoralityLabAI/DataGoblin/pull/46) established a gain from stringent native MeTTa curation: 190/600 vs 147/600 verified completions on the original fresh primary roots (+7.17 percentage points). [PR #47](https://github.com/MoralityLabAI/DataGoblin/pull/47) later reported 178/600 vs 117/600 (+10.17 points) on another cohort. The native policy filters closed, eligible, within-budget, unique-lineage, current-revision, dependency-satisfied and conflict-free training rollouts. See DataGoblin/metta_research_bench/looped/good.metta.

[DG-METTA-12 draft PR #57](https://github.com/MoralityLabAI/DataGoblin/pull/57), exact head 88c910993b3a86438c6de85d7efa039dcdb68cac, proposes full-set supervision for all tied optimal next actions, a stable hashed-singleton control and certified two-step STOP exchange. The claimed local courier Storyworld dependency was unavailable remotely, so native courier transfer remains blocked. OLR integrates the verified *methods* on its own Lean gym, without importing or claiming courier results.

## Components

- [metta_obligations.py](ppt_gym/metta_obligations.py): runs pinned Hyperon 0.2.10 queries over source-only, independently verified rollout metadata; filters public typed repair candidates and enforces source lineage and remaining verifier budgets.
- [metta_olr_cli.py](ppt_gym/metta_olr_cli.py): native MeTTa proposes a two-tactic repair after a premature STOP, while immutable saved Lean states protect against partial commit. Lean executes each action and independently certifies the original complete theorem.
- [metta_action_sets.py](ppt_gym/metta_action_sets.py): full legal-menu log-sum-exp action-set loss, canonical first-action labels, and stable SHA-256 hashed singleton controls. Only native-kernel and standalone-certified minimum-cost source actions can earn positive proof labels.
- [metta_actionset_native.py](ppt_gym/metta_actionset_native.py): fresh three-option Lean roots with one or two equally short successful actions. Native MeTTa curates source lineages; the same small tied recurrent model is trained from matched initialization on three label objectives.
- [test_metta_olr_bridge.py](ppt_gym/tests/test_metta_olr_bridge.py) and [test_metta_actionset_native.py](ppt_gym/tests/test_metta_actionset_native.py): atomic rollback, malformed MeTTa reply, budgets, curriculum split and differentiable tie controls.

A MeTTa match result is an *advisory policy offer*, never a Lean proof. The Lean 4.19 kernel and standalone source compiler supply proof authority. A failed macro leaves the original immutable state available. Each attempted action consumes one native tactic check. No held-out proof label or oracle continuation is passed to the recurrent model before its choice.

The hashed-singleton control follows DG-METTA-12 conceptually, but OLR hashes a public *state SHA-256* rather than DG's full canonical state serialization. Tie selection can therefore differ between the two implementations. All OLR arms have the same deterministic hash control, model architecture, root allocation, training steps and minibatch schedule.

## OLR-METTA-01: native STOP repair

From a real Lean proof root, a correct arithmetic guard is closed by Lean, leaving goals P and Q. The deliberately proposed STOP leaves these goals unfinished. Native Hyperon inspects public typed repair candidates and offers the two-step script applying h0 and h1; a reversed script, incomplete script or conflicted source fails preflight.

The accepted plan was executed by native Lean and the resulting full proof independently compiled with no axioms. It consumed exactly two tactic calls. With one available call, the MeTTa policy offered nothing, spent zero tactic calls and left the source state unchanged.

**Evidence:** one certified successful source proof and an actual Hyperon 0.2.10 rule query; source curation admitted the independently verified TRAIN trace, excluding unverified, stale and duplicate-lineage examples. This is a manually injected STOP, not learned STOP scheduling or a neural proof gain.

[Verified OLR-METTA-01 native receipt #14823257263](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45186000256/artifacts/14823257263).

## OLR-METTA-02: multiple equally optimal Lean actions

The native three-option Lean family has public candidate rules r0, r1 and r2. Each target has either a single four-tactic certifiable action or two equally short certifiable actions. Native Lean separately evaluates every candidate and independently compiles every successful complete theorem. Unsuccessful actions remain in the public normalization denominator and receive no positive target.

The experiment froze 12 training roots (three tied) and eight distinct development roots (two tied). Native MeTTa admitted all 12 verified TRAIN roots; ties remained *one training state presentation* each. There were 25 independently compiled successful scripts across source and development. Three paired model seeds (17, 29, 43), 40 source optimizer updates per arm, and the same small tied three-slot recurrent scorer were used for all label variants.

| Objective | Native-certified development choices / 24 | Certified tied-root choices / 6 | Native tactic calls |
| --- | ---: | ---: | ---: |
| A: canonical singleton | 21 | 6 | 90 |
| B: hashed singleton from optimum set | 24 | 6 | 96 |
| C: all-optimal action-set supervision | 24 | 6 | 96 |

The experiment contains eight independent development roots shared across three model seeds. All three objectives completed every repeated tied-root trial. B and C tied on overall verified completion, so this pilot **does not establish** a gain from full action-set supervision over hashed singleton. The difference over A is a small development observation and no sealed result. The exact environment guard remains solvable by direct arithmetic.

[Verified OLR-METTA-02 native data and checkpoints #14824166743](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45186782600/artifacts/14824166743).

## Reproduce

From research/lean_cgrt_hrm400 with the pinned Lean 4.19 server and CPU PyTorch installed:

~~~sh
python -m pip install "hyperon==0.2.10"
python -m unittest discover -s ppt_gym/tests -v
python -m ppt_gym.metta_olr_cli --out runs/olr-metta-01
python -m ppt_gym.metta_actionset_native --out runs/olr-metta-02 --seed 20261128 --train 12 --development 8 --steps 40
~~~

These output directories must be new. The hosted native CI workflow runs the same gates and uploads full proof receipts, native source states, model weights, training metadata and selected development actions. This code does not require a GPU.

## Next OLR-METTA experiment

The missing experiment is a *learned* obligation controller that decides when to pay for guard evidence or spend its two-call STOP repair allowance, rather than inheriting a deterministic branch schedule. Build fresh computational/dependent proof families where a direct arithmetic solver cannot simply close everything. Compare full optimal-set labels against hashed singleton, then compare the exact same checkpoint with and without MeTTa-backed atomic repair. Match actual Lean calls, model FLOPs, candidate menus and causal shams; obtain an EvalMath FSI-01 sample/power gate before any mechanism claim.
