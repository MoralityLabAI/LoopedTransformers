# BPD-LG-01: Counterfactual Lean-action curriculum and FC-12 probe bridge

**Date:** 2026-11-15. Mathematical interface: `MoralityLabAI/EvalMath/paper/bpd_interventional_separation.md` (BPD-01). This is a synthetic curriculum generator and training-only probe planner. It does not train a model, evaluate BPD/VPD weights, or change FC-12's sealed test results.

## Source-first curriculum

`generate.py` emits **384 typed cases**: four candidate tactic families (`exact`, `rw`, `rcases`, `induction`), four structural depths, six source roots per family/depth, a critical-obligation toggle and an irrelevant nuisance toggle. The critical toggle makes a paired positive/negative case; each negative has an explicit finite countermodel. Only 192 positive proof scripts enter the `.lean` compilation files. JSONL retains the source statement, split, exact template witness, countermodel and `kernel_verified=false` so downstream systems cannot silently treat uncompiled scripts as certificates.

The **train/dev/test split is at theorem-root granularity**, with counts 256/64/64. Variants and nuisances stay in the same split. The four source templates are intentionally simple probes of tactic syntax, and a learner can succeed using other tactics; `family` is a provided witness rather than a claim of uniqueness. Depth on `induction` counts nested recursive applications, while the other templates encode nested implication, successor, or conjunction structure. These tasks test controlled structure and proof syntax, with no natural-MathCan generalization claim. Holdout is over roots within the same template grammar, so template-memorization remains a baseline.

Compiler validity is a separate gate. CI installs **Lean 4.19.0**, generates the corpus, compiles every split's positive theorem source, runs the Python reference tests, and checks that negative cases carry the stated finite countermodels. No `sorry`, `admit`, `native_decide`, or added axioms occur in the generated sources. A green compiler exit validates the generated proof scripts only; it cannot justify model success, novelty, or compiler completeness.

## FC-12 training-only uncertainty

`bridge.py` reads `experiments/refutation_transfer/data/tfh01_fc12_verified_incidence.json`. The file contains **known positive action edges** from original Lean probes. An absent edge is **unknown**, because candidate scripts outside the explored grammar may succeed. The bridge runs the existing TFH-01 exact Hall-deficit and critical-edge analyses, then produces a ranked list of prospective source/compiler probes. Proposed edges have no training-success labels until independently replayed against the original MathCan theorem source.

BPD must assemble a matrix of *actually observed* outcomes before calculating the mechanism quotient, set-cover budget, or clock/visibility error floors. The new EvalMath exact API deliberately refuses to turn unprobed cells into negative observations. The next experiment should log `case`, `original theorem/root`, `tactic script`, `component`, `single/all-pass edit`, `state`, `pass`, `visible state`, `route`, and `compiler result`; a per-pass probability trace and a rotation-matched control belong alongside it. Preserve whole-original-theorem splits, independent Lean replay and time-matched symbolic search.

## Run

```bash
python -m unittest experiments.bpd_lean_curriculum.test_bpd_lean_curriculum -v
python -m experiments.bpd_lean_curriculum.generate --out /tmp/bpd-lg01
python -m experiments.bpd_lean_curriculum.bridge --out /tmp/bpd-lg01/fc12_train_probe_plan.json
lean /tmp/bpd-lg01/train_positives.lean
lean /tmp/bpd-lg01/dev_positives.lean
lean /tmp/bpd-lg01/test_positives.lean
```

The pinned compiler commands are automatically run in the dedicated `bpd-lean-curriculum.yml` workflow. On a local system without Lean, Python tests establish the generator/countermodel contract, and the theorem scripts retain their pending-kernel status.
