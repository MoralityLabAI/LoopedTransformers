# AGP-01-M: paired grader-belief traces for looped-model interpretability

**Source-only cross-repository protocol, 19 November 2026.** This is the downstream mechanistic bridge for [Control-Harness AGP-01 PR #72](https://github.com/MoralityLabAI/Control-Harness/pull/72), whose engineering protocol was first committed at `52c990e`. The external source-only training proxy and host gate live there. The present module does **not** train a recurrent model or open new scientific roots.

## Question and input isolation

For a single authorized/forbidden case, hold the public report and host-truth label fixed while toggling only an untrusted assertion that a grader wants permission granted. At each recurrent depth, ask whether the model's state tracks that untrusted signal and whether a native causal intervention changes host-true attempted action. Distinguish predictive state separation from task-beneficial causal steering.

`bridge.py` consumes the version-one Control-Harness receipt only after it validates every scored row, action schema, public-case pairing, true permission, strict gate outcome and disjoint root roster. It emits:
- `public.jsonl`: *only* `root_id,case_id,report,grader_hint`. This file may enter model inference.
- `host_gold.jsonl`: exactly one private `authorized` label per underlying case, reserved for the host scorer.
- `source_predictions.jsonl`: full-information logistic reference actions and their arm label, for compatibility checks.
- `manifest.json`: input SHA-256, output SHA-256 per file, source counts and a warning that origin source hashes were recorded rather than independently fetched.

The source-only engineering dataset from PR #72 is specifically **ineligible** as a fresh recurrent model heldout. The bridge accepts its records to exercise the schema.

## Source-only native trace contract

The next model adapter must preserve a native, single-checkpoint recurrent execution for each trained arm (`reward_target`, `host_trait`, `mixed`). Each trace row contains `arm`, `root_id`, `case_id`, `grader_hint`, `depth`, `state` and `checkpoint_sha256`. No host permission enters the trace or model features.

Register **shared weights across twelve recurrent applications** and inject the grader-belief signal *after completing depth six*. Audit depths `2,4,6,8,12`. The matched pre-injection states for hint 0 and hint 1 must be identical within an absolute norm tolerance of `1e-6`; the adapter's complete native execution and split-prefix/suffix parity must be independently checked. The current trace auditor verifies shape, provenance consistency, paired coverage, finite values and prefix equality. It reports per-root, per-depth L2 difference between grader-hint paired states. That observational difference is **not** an intervention result.

All reported root statistics aggregate cases within the independent root; the 24 cases, two hints, five depths and candidate/sham interventions are repeated measures.

## Registered future recurrent assay requirements

A future separate request will lock the checkpoint source and run a trained native model before opening fresh root entropy. It must use a single architecture/initialization and equal training/compute budgets across three objective arms, with disjoint OS-root train/calibration/development cohorts. Confirmation roots remain unopened. Freeze the exact recurrence, optimizer, learning-rate schedule, checkpoint SHA, data adapter, thresholds and scorer before first development readout. Exclude the 96/32 engineering roots and the prior ECA/BPD historical cohorts.

The proposal is a 16D recurrent controller or 7–100M parameter TRM, with a 12-loop trajectory and intervention site at depth six. At each heldout case:
1. Score native actions for both hint conditions against host truth; compute the paired forbidden-operation susceptibility difference, as defined in Control-Harness.
2. Extract the hidden difference at depth 6 and after injection; fit one candidate subspace with training-only roots.
3. Execute a physical source-preserving patch on a target rollout, plus zero-edit, exact self-edit, orthogonal residual and eight seeded norm-matched random-subspace shams.
4. Check split-forward parity, actual delivered L2 dose, unchanged model weights, native downstream logits and host-true authorization outcomes for **every** arm and attempted action.
5. Require calibration-headroom and sham resolution gates before positive causal claims. Rootwise paired uncertainty, multiplicity treatment and the practical effect floor must be frozen in a *new protocol amendment before fresh roots*.

The cross-model question is whether grader-signal sensitivity causes an unauthorized **proposal**, with strict-host executed authorization independently measured. An apparent representation of `grader_hint` can coexist with zero causal dependence of the final action, and a trusted gate can block a policy whose proposed actions remain unsafe.

### Source-only reproduction

```bash
python -m unittest discover -s mechinterp/agp01/tests -v
python -m mechinterp.agp01.bridge --receipt path/to/control_harness_agp01.json --out-dir path/to/bridge
python -m mechinterp.agp01.bridge --public path/to/bridge/public.jsonl --traces path/to/native_trace.jsonl --out path/to/native_trace_audit.json
```

No actual recurrent neural result is reported by this PR. The first scientific comparison requires a separate preregistered native trained-checkpoint experiment and independent audit.
