# OLR-01B: Native paired Horn proof preflight

**Date:** 2026-11-23  
**Status:** Completed verifier-integration preflight; this is **not a learned neural architecture result**.  
**Parent:** [OLR architecture](OLR_01_OBLIGATION_LEDGER_RECURRENCE.md) and [investigation](OLR_01_INVESTIGATION.md).

**Verified CI:** [native Lean workflow #45154630966](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45154630966), [CPU regression #45154631034](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45154631034). Both completed successfully.

## Native experiment

[ppt_gym/olr_native.py](ppt_gym/olr_native.py) generates four fresh pairs of propositional Horn worlds under seed 20261123. Each pair has identical primitive facts, target, two binary candidates, and mean incoming evidence cue (0.5). The grouping of premises within r0/r1 swaps across worlds, so the uniquely certifiable first action also swaps. Nuisance-chain length changes across pairs.

A native Lean 4.19 REPL supplies every public goal, hypothesis, menu and tactic outcome. The scorer parses only the public Lean JSON with [ppt_gym/live_lean.py](ppt_gym/live_lean.py). The reference winner is reserved for outcome auditing. The correct branch needs one rule application and two fact discharges, with a three-tactic budget. Both branches initially make native progress. The wrong branch leaves unprovable prerequisites.

The exact AND-OR ledger, a pooled-evidence fixed-tie policy, an edge-swapped structural sham, and forced-r0/forced-r1 references receive the same root menu and 3-tactic limit. Every attempt is replayed from a fresh native session and the successful ledger scripts are independently compiled and axiom-audited.

## Verified scorecard

| Arm | Kernel closures / 8 | Native tactic calls |
| --- | ---: | ---: |
| Hard AND-OR ledger | **8** | **24** |
| Pooled premise mean | 4 | 16 |
| Swapped premise-to-rule grouping | 0 | 8 |
| Forced r0 | 4 | 16 |
| Forced r1 | 4 | 16 |

**Independent standalone Lean certificates:** 8/8.  
**Native episode replays:** 40/40.  
**Original root menu:** apply r0, apply r1 in every world.  
**Pooled incoming premise cue:** 0.5 in both worlds.

The GitHub Action job log prints the full scorecard and asserts all of these counts. The generated episode manifest and individual source receipts were retained in the CI workspace during the job; their SHA-256 values are reproducible with the checked-in runner. The job is an engineering preflight, not an independently held-out scientific confirmation.

## What this proves and what remains uncertain

Explicit hyperedge incidence contains information lost by an unstructured mean over incoming premise bits. The deterministic ledger recovers it and Lean verifies the selected proof scripts. This experiment has **zero learned parameters** and offers no evidence of neural-model gains over exact Horn planning. The proof-tree cost counts repeated dependencies individually; it can exceed the cost of DAG-style premise reuse.

The public parser supports atomic propositional Horn expressions and chains of implications. It preserves pending Lean goals at the runner level but scores the focused one. Arithmetic or dependent hypotheses marked raw remain unknown. A structural fixed point is never promoted to a Lean kernel fact.

## Next gate: OLR-01C computational applicability

Freeze a new lineage-disjoint cohort of computationally conditional and dependent-type proof forks with paired opposite winners and independent certificates. Its public symbolic skeleton should admit both actions while their delayed proof guards differ. Any raw type must be scored as unknown until a real Lean response resolves its status.

Train a small, weight-tied evidence/obligation cell with semantic applicability and successor-goal heads; compare with the current two-slot reward actor, a newly trained identical actor, a direct guard reader, head-only OLR, edge-shuffled OLR, a FLOP-matched ordinary tied loop, and exact symbolic controls. Separate forced-route proof completion from learned STOP/probe spending. Price every Lean and inference call, retain three or more independent seeds, and request an EvalMath FSI-01 sample/sham-floor preflight before a sealed confirmation.

A prospective gain requires **additional Lean-certified completions at matched budgets** and a direction-specific causal response stronger than matched shams. Efficiency-only gains should be reported as such.

## Run locally

From research/lean_cgrt_hrm400 with the pinned Lean 4.19.0 REPL built:

~~~sh
python -m unittest discover -s ppt_gym/tests -p "test_obligation_ledger*.py" -v
python -m unittest discover -s ppt_gym/tests -p "test_olr_native.py" -v
python -m ppt_gym.olr_native --seed 20261123 --pairs 4 --out runs/olr-01b-native
~~~

The output directory is create-only and contains the episode ledger, compiled standalone source, certificate receipt, and an artifact hash manifest. The native CI assertion is in [.github/workflows/ppt-g03-live-lean.yml](../../.github/workflows/ppt-g03-live-lean.yml).
