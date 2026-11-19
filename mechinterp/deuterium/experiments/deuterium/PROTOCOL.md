# DEUTERIUM-FOLLOWUP-01

## Scope and status

This protocol defines two separate lanes. The recovered-checkpoint lane audits the
serialized model and its learned decomposition. Original forward code and the
original dataset/tokenization implementation are missing, so task performance and
steering in that lane stop at `BLOCKED_ARCHITECTURE_PROVENANCE`.

The executable follow-up uses the repository's `RecursiveGateTRM` on a **new
synthetic commit/veto/repair task**. This is a new toy method study. Its outcomes
cannot establish historical Intellect-3, ARC, Wordle, adapter, or governance claims.
Its shared transition runs three times; edits affect only the output head.

All numerical settings and endpoints live in `protocol.json`. A pilot uses seed
997 and creates only training and calibration data. Seeds 103, 211, and 307 are
reserved for the registered comparison. Test examples are generated inside the
scoring phase after the source/registration checks. Local pilot observations may
help fix implementation errors before freezing. Once scoring has started,
changes require a new protocol and fresh test seeds.

## Recovered checkpoint prerequisites

Load checkpoint dictionaries with `torch.load(..., weights_only=True)`. Confirm
each frozen target tensor in the learned decomposition exactly equals the
corresponding original checkpoint tensor. Record model parameter count, configured
reasoning depth, learned component counts and shapes, checkpoint hashes, and each
module's residual norm. Preserve all archive bytes.

The available powered classifier has a single reasoning step. A future recovered
comparison must retain that setting, recover a pinned original forward and
tokenizer, and match original reference logits before testing edits. Tensor
identity establishes weight lineage; it does not establish a faithful forward
pass. Full-rank residual replay must be distinguished from a decomposition's
components-only fidelity. Other recovered checkpoints require the same provenance
review before use.

Historical train/eval row identifiers are inadequate as the holdout boundary.
Group powered rows by the literal text before `\nMODEL_ACTION:`. Source grouping
leaves 51 previously unseen evaluation tasks, totaling 153 rows. Those examples
remain unscored in this work. Any future recovered-task study must prospectively
allocate these groups or acquire a new source-disjoint task sample.

## New recurrent toy task

Each example contains sixteen independent uniform coordinates in [-1,1]. The
label is `veto` when x3 + 0.5*x4 exceeds 0.35. Otherwise the label is `commit`
when x0 + 0.5*x1 - 0.25*x2 exceeds 0.1, and `repair` for the remaining examples.
This known generator provides independently reproducible labels. Train,
calibration, and heldout use distinct PRNG seeds and record exact data digests.
Every model seed gets fresh examples, model initialization, learned decomposition,
random control bank, and bootstrap seed. The intended unit of replication is the
whole model/decomposition experiment.

A 32-dimensional `RecursiveGateTRM` trains on 2,048 examples with the fixed
500-step schedule. Calibration contains 512 examples. Clean calibration accuracy
must reach 90%, with at least 32 examples in every class, before any edit can be
scored on the heldout split. The untouched heldout model must also reach 90% on
2,048 examples. Any failure is retained as a failed prerequisite and excludes a
complete three-seed comparison verdict. Failed seeds are never silently dropped.

## Learned directions and controls

Use the repository's actual `param_decomp.run_param_decomp.optimize` with learned
rank-one head components. The loss includes its existing `PGDReconLoss`, with two
adversarial mask steps, its importance minimality loss, and weight faithfulness.
The original model is frozen during decomposition. VPD trains on training
examples only. A saved configuration and loss trace identify the implementation;
the study does not substitute SVD components for the learned VPD arm.
The upstream optimizer records indices zero through its configured terminal step
and skips `optimizer.step()` at that terminal index. A terminal step of 300
therefore executes 300 main optimizer updates after 200 faithfulness warmup
updates.

**Documentation erratum, 19 November 2026, after scoring:** the prose at the
freeze commit counted 301 main updates. Source inspection established the
terminal-index skip. This correction changes the description only. The frozen
numerical configuration, implementation, dependency pins and measured results
remain unchanged; Git preserves the earlier wording at the registration commit.

There are 32 learned candidate matrices. Normalize each to a Frobenius norm of
1% of the original head's weight norm and evaluate both signs on calibration veto
examples. Select the candidate/sign minimizing mean cross-entropy. An independent
bank of 32 Gaussian outer products receives the same normalization, both signs,
and identical selection budget. The random arm is therefore a selected control.
The simple task-gradient baseline takes the negative best rank-one SVD
approximation of the training veto-loss gradient and normalizes it to the same
dose. SVD here belongs only to the explicitly named gradient baseline.

Clean inference and a zero edit form the baseline and sham. Every edit's matrix
rank and Frobenius norm are checked. Save the literal edit matrices, selected
indices/signs and selection scores. Verify both cached-head and native full-model
intervention paths, restore weights after every edit, and check the maximum logit
error against 2e-6. Evaluate components-only replay and residual-restored replay
separately. The residual can ensure full replay while the components still fail
to represent useful task directions; components-only fidelity is reported and
does not determine scientific success.

The selector does not filter candidates by observed collateral loss. This keeps
the failure modes visible. The selected edit, its dose, and every endpoint are
fixed before generating heldout examples.

## Endpoints, uncertainty, and verdicts

The primary seed effect is random-arm target-veto NLL minus VPD-arm target-veto
NLL. A positive value favors VPD. Report a paired 2,000-resample percentile
bootstrap interval within each seed. Across the three independent seed effects,
report their mean and a 95% Student t interval with two degrees of freedom. The
small seed count limits inference even when per-example intervals are narrow.

Secondary outcomes include the comparison to the gradient baseline, target
accuracy, accuracy across all labels, collateral accuracy on commit/repair
examples, and false commits on veto examples. Each arm retains its per-example
logits, predictions, losses, labels, and data identifier. Record runtime,
hardware, exact package versions, all failed gates, and exception details.

A positive study verdict requires every prerequisite to pass, a positive primary
effect in all three seeds, an across-seed lower interval bound above zero, and
at most two percentage points of collateral accuracy loss for VPD in every seed.
Any unmet criterion is retained. A completed comparison may be negative or
inconclusive. An Actions job's exit status is an engineering status.

## Execution contract

`python -m experiments.deuterium.steering preflight --records ROOT --out OUT`
audits the recovered checkpoint without evaluating task examples.

`python -m experiments.deuterium.steering toy --phase pilot --out OUT`
runs the auxiliary seed and never generates heldout data.

`python -m experiments.deuterium.steering toy --phase score --registration-commit SHA --out OUT`
runs the frozen three-seed comparison. The command requires a full commit SHA,
checks the protocol and source against that commit, and refuses a dirty tracked
tree or local source changes. The workflow records its commit and run URL. To
change the frozen experiment after looking at heldout results, write a new
protocol identifier with new heldout seeds and keep the prior results.

Use the supplied pinned dependency file and CPU environment. No model training,
decomposition, or performance claims from the new toy task are attributed to the
historical archive. Original large artifacts remain externally hosted and
hash-verified on retrieval.
