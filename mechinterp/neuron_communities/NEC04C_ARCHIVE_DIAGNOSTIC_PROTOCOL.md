# NEC-04C, stage 0: immutable NEC-04B parity failure replay

**Status:** retrospective read-only code, preceding any NEC-04C native-model training. **Parent:** [NEC-04C issue #296](https://github.com/MoralityLabAI/LoopedTransformers/issues/296). **Historical experiment:** [NEC-04B run 44992324562](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44992324562), merged [PR #295](https://github.com/MoralityLabAI/LoopedTransformers/pull/295).

## Scientific scope

Read exactly six original NEC04A **calibration** world roots 0..5, two existing native teacher arms (composed, additive), and all four historical optimizer checkpoints 120/240/480/900. The pretraining ledger, original 2,560 source cases, 768 owned validation contexts, original checkpoints and prior independent native replay receipts must come from one fixed historical run. Their byte hashes are verified. Each complete original source/validation case is assigned to the originally committed MUX selector's parity or alternate branch. No new model inference is required.

Derive native predictions again from archived two-class logits and independently reconstruct each label from the original NEC04A composed AST or the original frozen additive signed linear-threshold oracle. Require that the new branch totals reproduce each original whole-split correctness and balanced accuracy, and that they agree with the earlier independent model replay. Compute root-level branch accuracies, branch positive-class priors, constant majority-class baselines, balanced accuracy, positive prediction counts and one explicitly descriptive pooled case fraction. Preserve all unsuccessful roots, budgets and arms in the report.

This historical secondary analysis is **posthoc** and **does not measure** causal neuron-community necessity, SAE reconstruction, or generalization to unseen world roots. Six roots are the experimental unit for any root-macro descriptive averages. Pooled masks and contexts are correlated within root.

### Historic measurements that must reproduce

- Composed 900-step pooled parity validation: **1,139/2,272 = 50.13%**.
- Composed 900-step pooled alternate validation: **1,655/2,336 = 70.85%**.
- Original composed six-root macro validation: **60.63%**.
- All 48 original checkpoints must match archived byte SHA256 and their previous independent replay receipts.

If any fixed original root, model arm, checkpoint, saved logit, validation label, context ownership, class prior or old replay receipt disagrees, fail the job and withhold the diagnostic result.

### Protected experimental boundaries

NEC04C treatment training will use newly untouched **NEC04A calibration roots 6–11** under its own precommitted equal-optimizer-budget protocol in issue #296. This stage consumes exclusively prior roots 0–5. The 24 **development WORLD roots** and 16 **confirmation WORLD roots**, the original calibration-world 24 reserved context groups, old NEC03Z groups 56–63 and NEC03P recruited-world evaluations remain untouched.

The status of the broader causal study stays **FSI-01: REDESIGN_OR_PILOT_ONLY**. NEC04C stage 0 adds zero native model optimizer updates, zero source-SAE updates and zero new-root outcomes.

## Reproduction

Use the GitHub Actions historical read-only replay job defined in .github/workflows/nec04c-archive-parity.yml. It downloads only the already completed NEC04B workflow artifacts and executes:

~~~bash
python -m mechinterp.neuron_communities.nec04c_archived_parity_audit \
  --old-roots /tmp/nec04c/roots \
  --pretraining /tmp/nec04c/old/oracles \
  --out /tmp/nec04c/audit
~~~

The job uploads nec04c_archived_parity_audit.json. A separate training and causal-gate protocol will be frozen on a subsequent branch and new commit, with no change to the archived original benchmark or its protected-world allocations.
