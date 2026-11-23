# LSA-02B CPU checkpoint pilot

**Date:** 2026-11-23. **Status:** registered development-only native teacher/CPU training protocol. Existing [LSA-02B native proof evaluation](lsa02b_native_lean_bridge.md) remains the read-only evaluation authority. This campaign trains a genuinely Lean-verified *shortest-continuation* tactic ranker on GitHub Actions CPU workers before proposing a larger gym.

## Why the previous native result was zero

The preceding [two-root CI smoke](../reports/lsa02b_native_paired_smoke_20261123.md) produced 0/2 completed proofs for both block orders, while Lean independently compiled reachable reference proofs for both roots. That checkpoint had received two synthetic-fixture optimizer steps, so the result established wiring and accessible proof opportunities.

The [PPT-G04 archive](https://github.com/MoralityLabAI/LoopedTransformers/pull/281) reports native verification and limited positive closure for another architecture, but it contains no directly loadable, trained `ppt_gym.checkpoint.v1` Plan B model. The stronger [PPT residual-transfer work](https://github.com/MoralityLabAI/LoopedTransformers/blob/3df3d5efaff77686bcc46190fe01c05dd053da2b/research/lean_cgrt_hrm400/PPT_RESIDUAL_TRANSFER.md) fits a separate two-slot model. Their scores are retained as context, never imported as weights.

## Native source construction

`ppt_gym.native_train collect` consumes the existing frozen G03 suite generator, selecting **24 independent training roots** from source families and depths 2–4. It uses the actual Lean 4.19 `lgym` server and `lcgrt.extract.explore` to build the Lean-reported action graph and shortest proof. For each action along that reference path, it creates a model-visible state packet and a label for each public-menu tactic:

\[
y(s,a)=\mathbf1\left[1+\sum_{g\in{\rm children}(s,a)}d_{\rm Lean}(g)
=d_{\rm Lean}(s)\right].
\]

Here \(d_{\rm Lean}\) is the shortest proof size computed from **native Lean** transitions. Zero labels mean *a tactic is not on the shortest continuation*, including unreachable and longer alternatives. A zero label **does not claim that Lean rejects the tactic or that it cannot ultimately close the goal**. Labels are bound to a state, visible candidate menu, theorem root and verified proof path.

After collecting, the runner compiles **all 24 shortest scripts as standalone Lean declarations**, checks no extra axioms and hashes the exact training JSONL, census, compilation receipt, suite, source and executable fingerprints. Failures abort the source build, preserving all unconsumed evaluation splits.

No training example comes from calibration, `ordinary_test`, or `ood_family`; the generator loads their shards only to verify hashes and disjoint ancestry. Import rights and external dataset ancestry remain unverified, so any result is a scoped synthetic-Horn development experiment.

## Matched CPU training

The first cloud rung is two seeds, `101` and `103`, with two schedules per seed:

| Axis | Registered value |
| --- | --- |
| Backbone | PPT-B, same byte encoder, eight typed slots, two middle blocks, message gates and coda |
| Hidden width | 32; 4 attention heads; UTF-8 budget 384 |
| Architecture orders | AB and BA, same two physical parameter modules applied once per pass |
| Initial weights | Exactly matched by seed, logged SHA-256 |
| Data order | Uniform theorem-root sampling then within-root frame sampling using the same seed stream |
| Training | 240 CPU FP32 AdamW steps per cell, LR 0.001, gradient clip 1 |
| Supervision | All-pass shortest-continuation tactic ranking and outcome-label BCE, two recurrent passes |
| Resource ceiling | GitHub-hosted CPU only, at most two concurrent fit/probe jobs, two PyTorch CPU threads, bounded per-job runtime |
| Checkpoints | One end-of-run checkpoint per cell, exact source-manifest and training-metadata hashes |
| Native evaluation | Same **eight disjoint calibration roots** per cell, eight tactic checks per root; independent reference proof ceiling |
| Held-out tests | Locked for this campaign |

The two order schedules share model dimensions, parameter count, forward visits, recurrence depth, data and optimizer exposure. Their learned parameters may differ after training because the order changes the computation. Train-time policy loss is **reference-path imitation**; final evaluation counts completed, kernel-verified and standalone-compiled proofs.

## Evaluation and hard gates

The native evaluator uses the [LSA-02B proof bridge](../research/lean_cgrt_hrm400/ppt_gym/native_order.py). For each trained checkpoint it applies the frozen AB and BA dispatch interventions to the same source roots, with a public-order `auto8` baseline. It records tactic checks, full proof paths, independent standalone certificates and model-free replay. It also recomputes the shortest reference proof only after all model episodes finish. Every checkpoint must verify its native teacher manifest, the complete training census and lineage disjointness before scoring.

Calibration endpoints are:

- Native **proof completion per independent theorem**, including complete failures.
- Trained-order policy closure compared with public-order and the attainability ceiling.
- Frozen-order counterfactual effect, paired within each checkpoint/root.
- Identical seeded AB/BA start tensors and matching unique parameter counts.
- Actual effective optimizer steps, source hashes, training/fit loss and resource consumption.

This pilot is an **engineering and development comparison** with eight calibration roots. It does not satisfy the earlier competence threshold requiring at least **32 independent eligible calibration roots**, a minimum 0.40 trained-order close rate and at least 0.10 improvement over a registered random-menu baseline. No significance test or architecture promotion is authorized here. We keep the prospective `+0.025` meaningful difference for eventual performance testing and apply the EvalMath FSI-01 matched-sham preflight before sealed outcomes.

If either order is incapable of closing development roots, record `BLOCKED_INCOMPETENT_CHECKPOINT`. More training steps or a richer public-state representation would be a newly registered development intervention. This campaign will **not** open the locked test set, edit native oracle receipts, or selectively omit poor-performing roots.

## GitHub Actions execution

[`lsa02b-cpu-checkpoint-pilot.yml`](../.github/workflows/lsa02b-cpu-checkpoint-pilot.yml) has four stages: offline source-law tests, a real-Lean teacher builder, seed-matched CPU fits, and independently replayed native Lean calibration. GitHub retains separate short-lived artifact bundles for the frozen training source, each seed/order checkpoint and each proof transcript.

The workflow uses the existing native Lean 4.19 toolchain build and CPU-only `torch==2.5.1`; all fit jobs run with a two-thread ceiling. It requires no GPU, external inference API, or private local machine. Reproduction uses:

```bash
cd research/lean_cgrt_hrm400
python -m ppt_gym.live_cli freeze --out runs/lsa02b-cpu-suite --seed 20261123 --per-split 24
python -m ppt_gym.native_train collect --suite runs/lsa02b-cpu-suite --out runs/lsa02b-cpu-teacher --roots 24

python -m ppt_gym.native_train fit \
  --teacher runs/lsa02b-cpu-teacher --out runs/lsa02b-ab-101 \
  --middle-order ab --seed 101 --steps 240 --depth 2 --width 32 --max-bytes 384

python -m ppt_gym.native_order run \
  --suite runs/lsa02b-cpu-suite \
  --checkpoint runs/lsa02b-ab-101/checkpoint.pt \
  --native-teacher runs/lsa02b-cpu-teacher/manifest.json \
  --roots 8 --checks 8 --depth 2 --out runs/lsa02b-ab-101-native
```

The exact GitHub Actions job set is the authority for this pilot's registered runtime limits. A failed CI gate is a recorded construction or competence failure and is corrected openly in a new commit. Only train/calibration data may inform changes.
