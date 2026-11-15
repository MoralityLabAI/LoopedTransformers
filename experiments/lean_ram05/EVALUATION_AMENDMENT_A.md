# RAM-05 evaluation amendment A: post-score audit artifact routing

**November 15, 2026, after the first complete prospectively scored result.**
The exact source-family selector, nine model weights, training labels,
ranker policy, native/frequency baselines, tactic grammar, and both
original budgets remain frozen from the pre-registered protocol at
`215de8648d6db73352a4118f5507b45f9eb4f814`.

## Original score completion

[Original CI run #43675801084](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43675801084)
successfully completed:
- 27-root genuine Lean collection and independent replay of all **5,123**
  new training action effects, with zero outcome disagreements;
- nine CPU training fits with seed29 selected for each model by
  development loss and all checkpoint hashes verified;
- the original twelve-theorem source oracle (12/12);
- all **84** frozen primary/secondary model/native/frequency proof
  searches, with each recorded accepted theorem already receiving
  a full original Lean compilation and an independent same-source
  replay within the registered search budget.

The original scored source file is
`ram05_prospective/evaluation_report.json` in archived GitHub
artifact `ram05-first-prospective-proofs-43675801084`
(artifact ID `14146254137`, ZIP SHA256
`798cdc68aa762aa8ed1b72a024a9add8f4708fb6334ef45113d5cdf23d1e1069`).
Its original complete score report SHA256 is
`89dc5e33255fc0acfb820c344d9986f2e8e95cdc985fde9c35f5af284a94051c`.

## Audit-only failure and repair

The **post-score third independent Lean audit step** failed before
compiling any certificate, because `audit_results.py` inherited
RAM-04-specific artifact path conventions:
`RAM05_report.json` rather than
`evaluation_report.json`; missing the fixed suite component
`prospective_ram05` in each saved per-policy JSONL and certificate
directory. The frozen scorer had completed successfully and archived
all 84 original attempt records and 51 accepted original complete
proof scripts. No model/score-source defect caused this failure.

The audit-only repair changes the report-path lookup and suite-qualified
file routing, plus verifies the suite label when comparing recorded
summary counts. It does **not** change model ranking, training labels,
compiled source proof bytes, original score outcomes, native controls,
registered timeouts, candidate grammar, or test theorem selection.

The separate [RAM-05 independent audit CI](../../.github/workflows/lean-ram05-audit-frozen.yml)
will download the original complete frozen artifact by GitHub run ID,
check the exact saved report SHA256, and then compile every accepted
original complete theorem source in Lean4.35. Any mismatch is a
reported audit failure; the original accepted/rejected score stays
unchanged. The receipt and every compiler log will be preserved in
the new CI audit artifact.

The primary model-vs-native outcome is a **policy-order comparison**:
the new test scored native 6/12, loop32 7/12 and loop64 8/12, while
both flat64 and training-only frequency also scored 8/12. It
therefore shows a model-only gain against the fixed native ordering
but no recurrence-specific benefit. These are recorded frozen
scores; certification requires the still-pending third audit.

Original protocol success criteria and RAM-04 historical outcomes
remain unchanged. This post-score engineering repair is explicitly
disclosed and never used for model selection or a new test search.
