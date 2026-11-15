# RAM-07 evaluation amendment C | frozen scores and comment-safe third Lean audit

**November 15, 2026. This is an audit-only repair after all 120
original test searches finished, with their model and baseline
outcomes immutable.**

The original RAM-07 preregistration, source-only queue amendment A,
and term-style theorem renderer amendment B are preserved. The first
natural-positive training experiment used old MathCan
`c84e2025334e9a967c1297425a490f98d5b3d189`,
Lean4.35, with **10,644** independently replayed source-root tactic
observations and exactly **one distinct positive TRAIN exact_typed
application**. Nine CPU models were trained and frozen before the new
MathCan holdout was opened. Their immutable original training archive
is `ram07-frozen-audited-train-43682912918`.

## Complete scored experiment, independently preserved

[Complete evaluation-03 CI #43686631584](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43686631584)
passed the new term-style source renderer regression tests, rebuilt
the pinned newer MathCan source
`fca46bfb3430ac85ffca16f6a6a83bb0b0231e41`,
and completed **all 120 original policy/theorem searches** with
the frozen **12 prospective theorem
declarations**, six primary controllers and four secondary controllers.

The scorer ran the original full-source Lean4.35 certificate gate,
checked `#print axioms` and permitted dependencies, and required a
second byte-identical complete Lean compiler replay within the budget
for every accepted proof. Its immutable original
`evaluation_report.json` SHA256 is
`581b62588568e43d282afd74ce8975e631400cb32d6b13836685dfff1fbaba5b`.
All original compiler responses, full source scripts, source hashes,
failure records and proof gates are archived under
`ram07-complete-prospective-eval03-43686631584`,
artifact ID `14150907817`, ZIP SHA256
`5e6616aa6ddec3e70119b26fb10e79023d681f2be5194d4ae0ec5b7c4d73c00d`
(5,954 files). These artifacts are the authority for the
complete first successfully scored search and can be reproduced
without retraining or running theorem search again.

| Search | Native | Global frequency | Shape frequency | Looped32 | Looped64 | Flat64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Primary 30s/48 evaluations/8 steps | 3/12 | 4/12 | 4/12 | 4/12 | 4/12 | 4/12 |
| Secondary 60s/128 evaluations/16 steps | 3/12 | 4/12 | 4/12 | N/A | 4/12 | N/A |

The registered model-only versus all three non-neural baselines
criterion was **false**. The only theorem added beyond fixed native
was the already-existing original theorem
`MathCan/TacticFibers.lean:exactly_two_matching_orbit`,
also certified by global and shape frequency. These are search
results on existing mathematical statements, with no newly
conjectured/proved mathematics claimed.

## Third-audit-only false positive

The extra post-score auditor `audit_results.py` stopped on
`assert "admit" not in text` where `text` was the *entire
original-source rendered theorem*, including documentation.
The original MathCan/TacticFibers module contains the benign
source comment: `It admits two different perfect matchings`.
An independent byte inspection of the frozen original archive
found this substring in **28 accepted compiler certificate
sources**. None of the **38 accepted selected tactic scripts**
contained the forbidden Lean code commands `sorry`, `sorryAx`
or `admit`.

The audit-only correction validates every actual selected tactic
through the unchanged registered `ram07_core.valid_script`
whitelist, scans only those executable candidate script strings
for whole-word unsafe tokens, and preserves the original
byte-identical source-SHA assertions and Lean axiom gate.
It additionally checks the third independent compilation's
allowed axiom set against both archived in-budget compiler
receipts.

Original MathCan comments and reference-code documentation
are copied only as original source context. They never become
a chosen tactic or a generated proof. This audit-only correction
does not rewrite any of the 38 already accepted complete
theorem sources or any model, baseline, case or budget.

## Required independent replay of original scored artifact

The distinct RAM-07 audit-only GitHub Actions workflow must:

1. Download exactly the original complete
   `ram07-complete-prospective-eval03-43686631584` artifact,
   and reject any change to its saved report SHA256.
2. Check the original seed+SHA256 of every selected trained
   checkpoint against the frozen train-only receipt.
3. Count the original 120 scored policy/theorem cases and
   verify their exact 3/4/4/4/4/4 primary and 3/4/4/4 secondary
   coverage; prohibit retroactive score changes.
4. Independently compile all **38** original accepted full
   theorem source files against the exact fca46 MathCan
   original dependency graph using Lean4.35. Check original
   bytes SHA256, source-identity with the in-budget replay,
   permitted axiom inventory and exact selected tactic grammar.
5. Publish an external audit receipt, each source hash,
   source-log evidence and the compact original score report.
   On failure, leave the original accepted scores and failures
   unchanged, preserve the failure and state the exact blocker.

**No new training, case selection, tactic generation, model
checkpoint changes, original full score rerun or cherry-picked
case repair** belongs in this audit-only process.
