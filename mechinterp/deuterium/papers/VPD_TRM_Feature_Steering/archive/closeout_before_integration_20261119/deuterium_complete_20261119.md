# Deuterium: Auditing Parameter Steering in Recurrent Models

## Historical control studies and matched native-weight experiments

Patrick Dugan · MoralityLabAI

Research report · 19 November 2026

Completed evidence review through the first native recurrent-weight experiment and source-only adapter schedule audit. Historical source is preserved at commit `2df0092d5be4d8e4748b3c66ef8325dedef17549`.

## Abstract

We audit a sequence of parameter-steering studies in small recurrent models and evaluate learned adVersarial Parameter Decomposition (VPD) directions under matched intervention budgets. The recovered archive resolves an apparent discrepancy in a six-seed replication: all 16 accepted grafts appear in the raw sweep, while a summary preview retained 12. Every accepted graft ties its strongest guarded random comparator on task metrics and wins on smaller output movement. Historical ARC Challenge cards share many examples across nominal validation folds. Wordle score records support a constrained-decoding gain from 24/72 to 42/72 under their recorded identity rule, with incomplete upstream episode lineage. A frozen 1,763-parameter recurrent-gate experiment compares equally searched rank-one directions; its hosted reproduction yields a mean random-minus-VPD target loss of 0.000890 nats and an across-model 95% Student t interval of [-0.000160, 0.001940]. A subsequent experiment edits the shared recurrent weight of a native four-pass, 44,001-parameter model. Across three previously examined teachers, the corresponding mean is -0.00002608 nats, with interval [-0.00028738, 0.00023521]. Both experiments fail their registered advantage criterion. Independent audits reproduce the saved-score arithmetic. Source instrumentation then establishes that the pinned VPD implementation retains the last input to a reused module and applies its importance mask at every recurrent pass. A pass-indexed adapter passes pointwise source-only tests. A subsequent full-schedule comparison exceeds its fixed trajectory tolerance, leaving new-root experiments blocked at the engineering gate.

## 1. Scope and evidence standard

The research question is whether an interpretable parameter basis can help a controller improve a task-recursive model while preserving useful behavior elsewhere. An outer loop could select a failure, propose a parameter intervention, measure its effect, and retain successful changes. Establishing that process requires persistent gains across independent evaluation examples and repeated edits. The historical experiments reached several earlier steps along that path.

This revision treats the June manuscript as a research record. Its chronology, discarded attempts, and positive observations remain in an archival copy [R1]. The present text makes each evidential claim at the level supported by a corresponding artifact. A stored summary supports a statement about what a historical run reported. Recomputing counts from raw rows supports a stronger statement about those rows. A new execution adds evidence about the current implementation, model, data, and environment. These levels are recorded alongside results so that a reader can tell which computations have been repeated.

| Result family | Evidence in this revision |
| --- | --- |
| Earlier full arena | Recovered summary only; sweep outside supplied package |
| Six-seed replication | Complete archived-row recount and acceptance-rule audit |
| ARC and Wordle policies | Recomputed cached-score decisions and source-identity overlap |
| Governance | Preserved report values with source-code audit |
| Component adapters | Recovered result JSON and dataset overlap; exact invocation missing |
| Later basis-search addendum | Recovered manuscript report only |
| Synthetic gate follow-up | First local execution and hosted reproduction; NO_REGISTERED_ADVANTAGE |
| Native recurrent-weight steering | First held-out execution and independent arithmetic recovery; DEVELOPMENT_NO_REGISTERED_ADVANTAGE |
| Recurrent importance-mask semantics | Instrumented source-only observation using the original fitted checkpoints |
| Pass-indexed adapter | Pointwise source tests passed; full-schedule parity gate blocked; new-root comparison unexecuted |

The word *TRM* covered several architectures during this project. The small arena models are recurrent decision gates with a projected input, repeated transition and layer normalization, and an output head. Later work uses a larger recursive classifier. The methods and model identities are kept explicit throughout. “ARC” here means the multiple-choice **ARC Challenge** task; the recovered cards provide answer-label scores. They do not measure performance on ARC-AGI visual reasoning.

Parameter-basis terminology also needs precision. The arena source constructs singular-vector and class-row components of a head. Their historical names include `vdp_or_class_row` and `vdp_graft`. Those labels are preserved in machine-readable outputs for traceability. In this paper they are called **head-component grafts**. Later learned decomposition checkpoints form a separate evidence family. A convenient field name cannot establish that a learned adversarial decomposition was fitted or that its full-model reconstruction is faithful [R2, R3].

### 1.1 Method context and contribution

VPD was introduced by Bushnaq et al. as adVersarial Parameter Decomposition. It learns simple parameter subcomponents while encouraging sparse input-dependent importance and reconstruction under adversarial partial ablation. The published method also includes a manual language-model editing demonstration [R15]. Our experiments use a pinned implementation of that method to study a narrower question: whether a fixed calibration search over learned rank-one directions improves a specified task loss more than an equally searched random bank.

The audit contributes an account of what the historical acceptance rules measured and how example reuse affected their interpretation. The new experiments add actual parameter interventions with explicit dose and source identities. Recurrent instrumentation further records what one module-path cache means when the same physical layer runs repeatedly. These contributions remain tied to the architectures and tasks reported here. The term *causal importance* names the learned function in the VPD implementation; a task-level causal interpretation depends on the intervention and endpoint used to test it.

## 2. Archive recovery and custody

The recovery input is `VPD_TRM_original_records_202606.zip` from the Hugging Face dataset `AlephFunk/vpd-trm-original-records-202606`. The fetch configuration pins the dataset revision and the complete archive SHA-256. The reproduction script verifies the digest before extracting members, rejects unsafe archive paths, and emits an inventory with a SHA-256 for every member. Derived tables and experiments are written outside the extraction directory. Original checkpoints and large sweeps stay in the dataset host; Git contains their manifests and the analysis needed to verify them [R2].

| Item | Verified value |
| --- | --- |
| Archive bytes | 1,326,795,233 |
| Extracted members | 406 |
| Original source members | 401 |
| Recovery metadata members | 5 |
| Uncompressed bytes | 3,716,109,041 |
| Dataset revision | `b208d547d20e89dc6c221bc640525320cc67a2f8` |
| Digest check | PASS |

The archive contains a summary for the earlier run `metta_organelle_arena_full_20260531T180830Z`. Its sweep is absent from this package. The package's own missing-evidence note describes that sweep as “not requested.” This establishes the scope of the supplied recovery, while leaving the wider availability of that earlier sweep unresolved. Consequently, the earlier counts remain summary-only. They are kept separate from the later `metta_organelle_arena_paper_20260601T024147Z` replication, whose sweep is available for a complete recount [R2, R3].

The archive also preserves two earlier export ZIPs. They count as source members, and their contents may duplicate other records. The inventory reports files and bytes rather than treating every contained copy as an independent observation. The archive digest identifies the bytes used in this analysis; it does not recover an unrecorded command line, establish when a model first saw an example, or reconstruct a missing runtime state.

## 3. Historical head-component transfer

### 3.1 Task and intervention

The arena trains small organelles for decisions such as route, repair, verify, commit, and abstain. Related tasks expose compatible decision labels. A source head component is mapped into a target's label space and added at runtime. Candidate scales and component choices are scored against the target's baseline. The later replication uses six seeds, 140 training epochs, hidden width 24, hard slices, and eight random controls according to its recovered configuration [R3].

The primary raw record contains baseline metrics, intervention deltas, source and target identities, and acceptance flags. Its rows represent candidate evaluations, often related through shared models, examples, components, and scales. The resulting denominator measures search volume. It supplies no count of independent statistical trials.

### 3.2 Reconciliation of 16 and 12

The summary has a total `accepted_count` and a bounded `best_grafts` list. The code constructs that list by placing accepted candidates first and then truncating it to `top_grafts`. The paper copied the accepted total from one field and a seed distribution from the preview. The streamed audit counts all sweep rows, checks the acceptance predicate, and compares the resulting totals with the summary and preview [R3].

| Seed | Sweep rows | Accepted in full recount | Accepted in preview |
| --- | --- | --- | --- |
| 20260601 | 217,497 | 3 | 2 |
| 20260602 | 217,539 | 4 | 2 |
| 20260603 | 217,546 | 1 | 1 |
| 20260604 | 217,546 | 0 | 0 |
| 20260605 | 217,504 | 8 | 7 |
| 20260606 | 217,539 | 0 | 0 |
| Total | 1,305,171 | 16 | 12 |

| Sweep | Accepted | Stable neutral | Evidence |
| --- | --- | --- | --- |
| Earlier full arena | 23 | 54,295 | Recovered summary; sweep absent from package |
| Six-seed replication | 16 | 110,731 | Streamed raw-row recount |

The replication registry contains 186 organelles. The 16 accepted rows reduce to 14 distinct seed/source/target/component combinations when scale variants are collapsed.

![Accepted replication grafts by seed and the summary preview](figures/replication_by_seed.png)

*Figure 1. Each bar counts accepted candidate rows. The preview is a subset of the same sweep. The difference between the two bars exposes truncation; it carries no independent replication evidence.*

The accepted rows preserve a positive local observation under the historical acceptance rule. Their recurrence depends on seed and task family. Several accepted rows can describe nearby scales or closely related transfers, so a count of accepted rows cannot by itself estimate the probability that a new model or task benefits from a graft. The generated accepted-row file retains identities and line numbers to permit analyses at coarser units.

### 3.3 What the acceptance rule measures

The historical gate requires the designated component family, a guardrail pass, a positive signal, and a rank superior to the best guarded random row for the same seed and target organelle. Random directions are generated in groups of eight for each source-component/target combination, matched to that component's rank and Frobenius norm, then evaluated over the scale grid. The acceptance comparison pools all of those source components and scales for the target. The rank is lexicographic: it orders the task signal first, then false commits, false rejects, accuracy, and finally the negative mean absolute logit change. When all task entries tie, the last quantity rewards the smaller perturbation. A task-metric tie can therefore satisfy acceptance [R3].

| Diagnostic among accepted replication rows | Count |
| --- | --- |
| Strictly better task-metric rank than best guarded random | 0 |
| Equal task-metric rank; wins on smaller mean absolute logit change | 16 |
| SVD head components | 9 |
| Class-row head components | 7 |
| Repair-step targets | 14 |
| Candidate-verification targets | 2 |

All 16 accepted rows tie the best guarded random control on the first four rank entries. Their winning entry is the smaller output perturbation. The reconciled result records lower output perturbation under the historical rank rule. Strict task superiority over the best guarded random row for the target has count zero.

The guardrail permits some degradation. In the recovered source, accuracy can fall by as much as 0.05 and the false-reject rate can rise by as much as 0.05. The acceptance label therefore describes a bounded tradeoff under that implementation. Describing every accepted graft as preserving all task behavior would exceed the criterion.

A sweep over roughly 1.3 million dependent candidates raises a further selection issue. The target-wide maximum over many random candidates supplies a stringent empirical comparison, while pooling also changes which direction and dose define the winning random row. This rank threshold supplies no calibrated family-wise error rate over the full search. The historical acceptance rule lacks a multiplicity correction and an independently sealed confirmation set. We retain the accepted candidates as findings from an exploratory search and require separate held-out confirmation for a steering advantage.

Stable neutral rows remain a distinct count. They capture interventions that pass the guardrail while producing no positive signal. Their abundance is relevant to the search landscape and has no direct implication for task improvement.

## 4. Cached controllers and task-level claims

### 4.1 ARC Challenge overlap

The ARC experiments work with cached per-action scores. A route rule suppresses a selected answer label under a predicate built from the top choice, runner-up, and score margin. These operations can expose useful structure in the recorded scores. Their relationship to a parameter intervention requires a separate test.

The four ARC cards contain 32 rows each. The audit matches trajectory identities across cards, reports pairwise overlap, and computes the examples unique to each held-out file relative to the other files. The cross-validation source builds its training union and its held-out card separately, deduplicating within each. It does not remove held-out identities from the training union [R4].

| Held-out seed | Rows | Seen in other cards | Unique to this card | All-row delta | Unique-row delta |
| --- | --- | --- | --- | --- | --- |
| 23 | 32 | 27 | 5 | 0.0625 | 0.0000 |
| 37 | 32 | 28 | 4 | 0.0625 | 0.0000 |
| 101 | 32 | 30 | 2 | 0.0938 | 0.0000 |
| 151 | 32 | 26 | 6 | 0.0938 | 0.0000 |

The 128 scored appearances contain 62 distinct trajectory identities. Replay of the fixed `top D, runner-up A, margin <= 1.0` condition changes 44/62 correct to 48/62, with 4 rescues and 0 damages. Every fold's examples unique to that card have zero gain under this fixed rule. These post hoc slices describe the previously selected rule; their labels have already entered the research record.

![ARC Challenge overlap and examples unique to a card](figures/arc_overlap.png)

*Figure 2. Shared trajectory identities quantify overlap in the historical leave-one-score-file-out procedure. Fresh random sampling seeds can still select the same examples from a small pool.*

The reported four accepted folds describe four score-file evaluations with extensive example reuse. Their acceptance cannot estimate performance on four independent sets of unseen tasks. The rule was also selected after inspecting scores and labels during the surrounding research program. Reconstructing a stricter split today would constitute a new retrospective analysis of already explored data. The present audit exposes overlap directly and reserves prospective generalization claims for a new, independently separated endpoint.

Historical ARC reports include a first cached-rule gain and a subsequent failure to pass the stacked-edit control. A broad fixed-label control can match the selective rule's score while touching more examples. That specificity observation remains useful for designing an internal steering target. It gives no evidence that the internal parameter basis already realizes the rule.

### 4.2 Wordle scoring and repeated examples

The Wordle policy penalizes `[crane]` after the first step, while retaining its availability as an opening action. Its live scorer assigns likelihoods to a small legal-action set; later scripts reuse those scores to evaluate the policy. The original reports describe a four-example card, an eight-example card, and two cards of 32 examples. The recovered scores permit arithmetic replay and an overlap audit [R5].

| Card | Rows | Baseline correct | Policy correct | Delta | Best control delta |
| --- | --- | --- | --- | --- | --- |
| 4sample | 4 | 2/4 | 4/4 | 0.5000 | 0.0000 |
| 8sample | 8 | 4/8 | 8/8 | 0.5000 | 0.0000 |
| 32_seed23 | 32 | 11/32 | 17/32 | 0.1875 | 0.0312 |
| 32_seed37 | 32 | 9/32 | 17/32 | 0.2500 | 0.0000 |

The mean card delta is 0.359375. Weighting scored appearances by card size gives 0.263158 across 76 rows. The four-example card is contained in the eight-example card; the two later cards are disjoint from those discovery cards and from each other by recorded trajectory ID. The audit's stated identity rule yields 72 retained records with 24 baseline hits and 42 policy hits. The full audit preserves both identity definitions and repeated-record diagnostics. Identity rule: `env:trajectory_id:step`. These events span 71 recorded episodes; one episode appears at two different steps. All four cards replay their stored per-row decisions exactly.

The unit of observation matters. Card totals count scored appearances; shared trajectory identities reduce the number of distinct examples. The headline mean across cards gives a four-row card and a 32-row card equal weight. The generated results state whether an estimate is a card mean, an appearance-weighted mean, or a deduplicated estimate. These quantities answer different questions and remain separately labeled.

This result describes constrained decoding over a specified candidate set. It supplies a positive behavior-level target for later internalization. Token penalties can improve an answer choice without locating a faithful component inside a recurrent model. The archive contains score records for this policy, while a fresh execution of the historical autoregressive model and scorer would require its complete model, adapter, and runtime provenance.

### 4.3 Outer-loop and eval-alignment records

The recovered roundout tables preserve a wider sequence of feedback and search results. The outer-loop experiments combine observed arena candidates with synthetic or estimated effects for some actions. Filtered policy totals also depend on source-family eligibility. A rule that requires an accepted component-family source excludes several controls by construction. Its resulting accepted-gain column describes that filter's behavior; a method comparison needs symmetric eligibility and independently measured task effects [R6].

| Historical family | Metric | Value | Evidence |
| --- | --- | --- | --- |
| eval_alignment_boundary | accepted_eval_aligned_count | 0 | Recovered summary/table |
| edit_bootstrap_microcycle | accepted_edit_count | 1 | Recovered summary/table |
| edit_bootstrap_microcycle | stacked_bootstrap_success | False | Recovered summary/table |
| multi_family_bootstrap_sweep | family_count | 15 | Recovered summary/table |
| multi_family_bootstrap_sweep | first_edit_family_count | 3 | Recovered summary/table |
| multi_family_bootstrap_sweep | stacked_bootstrap_family_count | 0 | Recovered summary/table |

| Historical policy | Raw total gain | Filter accepts | Filtered gain |
| --- | --- | --- | --- |
| failure_replay | 0.231252 | 0 | 0 |
| llm_edit_proxy | 0.309252 | 1 | 0.04 |
| mixed_controller | 0.634724 | 1 | 0.04 |
| random_control | 0.660724 | 0 | 0 |
| vpd_greedy | 0.660724 | 5 | 0.2 |

These values reproduce the archived roundout CSVs. The model executions and estimated-effect calculations have not been rerun.

The eval-aligned search reported zero accepted rows in its initial gate and strict hill climb. A targeted generator then reported 36 accepts before matched controls were available. Adding targeted random controls returned the accepted count to zero. The targeted-search counts remain historical manuscript reports [R1]. The relevant metric was a deterministic estimate of downstream effect. These records describe a failure to establish superiority under that estimated scoring procedure. A live task outcome would require a separate run.

The cached ARC bootstrap likewise reported one accepted edit before its next edit lost to a broad control, and the multi-family search reported zero stacked successes. These results constrain the recursive-improvement thesis. They record where a search stopped and why. Green execution status, empty stderr, or a completed search provides evidence about execution and supplies no scientific acceptance criterion by itself.

### 4.4 Later basis-search report preserved in the recovered manuscript

The archive contains a longer manuscript than the repository snapshot. Its additional sections describe June 9-10 experiments on basis-constrained gated edits. Both manuscript versions are preserved byte-for-byte. This audit has parsed the recovered text; the later experiment's candidate-level sweep and invocation have not been independently reconciled. The following values therefore retain **historical report-only** status [R10].

| Reported arm | Accepts | Accepts / eval | Evals to first | Accepted delta |
| --- | --- | --- | --- | --- |
| free_rank1 | 61 | 0.0046 | 15.5 | 0.750 |
| basis_constrained | 32 | 0.0024 | 24.0 | 0.953 |
| matched_random_subspace | 116 | 0.0088 | 48.5 | 0.707 |
| attribution_shuffled | 180 | 0.0136 | 7.5 | 0.850 |

Source: the recovered expanded manuscript, parsed without modifying its numbers. Each row is report-only.

The reported rank-three basis search accepts fewer rows than the free-rank-one, random-subspace, and attribution-shuffled arms. Its reported mean delta among accepted rows is higher. The report also describes a rank-one follow-up with 193 basis accepts and zero matched-random accepts, while the attribution-shuffled arm reaches 194. These observations motivated a distinction between useful read directions and unproven read/write pairing. They concern repeated searches over cached answer-score surfaces whose source-task independence requires separate validation.

Other additions preserve both favorable and unfavorable outcomes. A recombination probe reports 37 accepted children at 6.49 evaluations per accept, evaluated only on cells where a parent had already succeeded. A support-overlap predictor loses to a norm-product baseline in all 24 modules, with reported Spearman correlations 0.11 and 0.51. The D=72 organelle search reports zero accepts for every arm because all 2,400 evaluations per arm fail its touch cap. An Intellect-3 lane reports a best guarded estimated delta of 0.087 at edit norm 4.0, alongside zero accepts under the recurrence criterion. The recovered text discloses superseded trigger and dose settings. Those corrections remain part of the historical account.

The same added text reports reconstruction of both earlier arena runs. That statement has manuscript-level support here. It does not change the evidence tier assigned to the earlier full-arena sweep in Section 2, because the present recount uses only the supplied replication sweep.

## 5. Later learned-decomposition experiments

### 5.1 Governance patch removal

The governance experiment uses a feedforward TF-IDF classifier and trains a low-rank “dirty” update intended to improve professional-law answers while damaging ARC Challenge. A selector then subtracts component projections from that update. The report records target accuracy 0.6500 at baseline, 1.0000 after the dirty patch, and 1.0000 after the decomposition-informed cleanup. Collateral accuracy moves from 0.6775 to 0.2975 and then to 0.0100 [R7].

| Reported arm | Target accuracy | Collateral accuracy | Evidence |
| --- | --- | --- | --- |
| baseline | 0.6500 | 0.6775 | Historical report + source audit |
| raw_dirty | 1.0000 | 0.2975 | Historical report + source audit |
| decomp_ci_informed | 1.0000 | 0.0100 | Historical report + source audit |
| norm_only | 0.9500 | 0.0100 | Historical report + source audit |
| random | 1.0000 | 0.2675 | Historical report + source audit |
| svd_topk_removed | 0.6500 | 0.6775 | Historical report + source audit |

The cleanup preserves the target gain and amplifies the collateral failure. The script trains the update on the same target slice later scored and uses both scored slices to select removed components. That design supports a diagnostic about the tested slice. The exact invocation, complete selected row identities, dirty-update checkpoint, and projection coefficients remain absent from the recovered provenance.

The projection method adds another limitation. It computes each coefficient as an independent inner product divided by the component's squared norm, then subtracts the chosen weighted components. An overlapping basis generally requires a joint least-squares projection to represent the same geometric operation. The independent subtraction can change the update outside the intended residual direction. The SVD control removes up to eight directions from a rank-four patch under the script defaults, which restores the baseline by removing the patch. A comparison about selective governance must account for retained target gain and the dose of the residual update.

### 5.2 Gated adapter reports

The adapter evaluates critic reward buckets; solving a logic task requires a separate endpoint. Its code applies parameter changes through selected component directions and trains their scalar coefficients. The validated report uses four selected directions and four randomly selected directions. It reports 64 training rows, 32 validation rows, and 64 evaluation rows. All arms have bucket accuracy 0.515625 after the reported 12 steps. The selected arm's loss is 1.004116, compared with 1.001876 for the frozen model and 1.004754 for the random-component arm [R8].

| Historical experiment | Arm | Eval loss | Bucket accuracy |
| --- | --- | --- | --- |
| SVD static | base_frozen | 1.001876 | 0.515625 |
| SVD static | vpd_static_selected | 0.972768 | 0.531250 |
| SVD static | random_trunk_control | 1.002200 | 0.515625 |
| Learned static | base_frozen | 1.001876 | 0.515625 |
| Learned static | vpd_selected | 1.016567 | 0.500000 |
| Learned static | random_trunk_control | 0.991228 | 0.515625 |
| Learned validated | base_frozen | 1.001876 | 0.515625 |
| Learned validated | vpd_validated | 1.004116 | 0.515625 |
| Learned validated | random_trunk_control | 1.004754 | 0.515625 |
| Learned validated (0 steps) | base_frozen | 1.001876 | 0.515625 |
| Learned validated (0 steps) | vpd_validated | 0.999113 | 0.515625 |
| Learned validated (0 steps) | random_trunk_control | 1.001876 | 0.515625 |

| Identity used for dataset comparison | Shared distinct identities | Evaluation rows sharing a training identity |
| --- | --- | --- |
| exact text | 637 | 666 / 3,270 |
| row id | 0 | 0 / 3,270 |
| task prefix before MODEL ACTION | 2591 | 3,117 / 3,270 |

The archived dataset has disjoint row identifiers. Exact text overlaps remain, and task context before the literal `MODEL_ACTION:` delimiter is shared by most evaluation rows. This source-task identity check exposes a different level of dependence from a row-ID split. It also leaves the historical 64-row evaluation subset unauthenticated because the exact invocation and subset IDs are absent.

The static SVD arm records a gain of one correct bucket prediction among 64 rows and lower loss. Its learned-component counterpart falls from 0.515625 to 0.500000 bucket accuracy, while the random-component arm retains 0.515625. These are separate historical experiments with incomplete execution provenance. The zero-step validated report gives a small selected-arm loss decrease. Selected directions start with signed coefficients of magnitude 0.1; the random arm starts at zero. The reported difference therefore combines direction selection and unequal initialization dose. Equality in component count also leaves matrix norm and effective rank unconstrained. The archive recovers model-related artifacts and dataset rows, but the historical invocation and explicit train/validation/evaluation row-ID manifest remain missing. Script defaults cannot fill those gaps.

These experiments motivate a new comparison with a fixed edit budget, symmetric controls, explicit source identities, and a model replay test. The follow-up below uses those requirements as prerequisites.

## 6. Frozen synthetic follow-up and hosted reproduction

The follow-up is specified in `experiments/deuterium/PROTOCOL.md` and `protocol.json`. Local commit `f1e3e9ba663ea906a6468d76426bb7381a6e23ad` fixed its design before held-out scoring and was recorded in the task session before the scoring command. Hosted reproduction followed after repository access became available. The initial local freeze and the later Actions replay have distinct dates and receipts; the replay uses already exposed seeds. Section 6.3 preserves the first local measurements, and Section 6.4 reports the hosted execution [R9, R11].

### 6.1 Recovered-checkpoint prerequisites

The recovered lane audits the powered classifier and its learned decomposition. Safe tensor loading permits a comparison of frozen tensors, component shapes, and residual norms. The original forward class and dataset/tokenization implementation are prerequisites for reproducing predictions. Weight equality alone cannot identify those computations. The recovered model's configuration uses a single reasoning step, which also limits any claim about multi-step recurrence.

The historical dataset's source-task overlap requires a stricter boundary for future testing. Grouping text before the literal `MODEL_ACTION:` field leaves 51 previously unseen evaluation task roots, comprising 153 rows. These remain unscored. A recovered-model steering experiment would first need the pinned original source, reference-logit agreement, and a prospective allocation of those task groups or newly acquired independent tasks. A failed prerequisite records a blocked comparison with its cause.

### 6.2 New synthetic recurrent-gate method

The local comparison uses the repository's `RecursiveGateTRM` on a newly generated commit/veto/repair task. The 16 input coordinates determine labels through fixed linear threshold rules. Model width is 32 and the shared transition runs three times. Each registered seed receives its own training, calibration, and held-out examples, model initialization, learned decomposition, and random bank. This lane tests the steering method on a defined toy distribution. Historical Intellect-3, ARC, Wordle, adapter, and governance outcomes retain their separate evidence status.

The comparison used model/decomposition seeds 103, 211, and 307. Seed 997 served only the calibration/runtime pilot, which left its held-out data unopened. The fixed schedule trains a model on 2,048 examples for 500 steps and calibrates on 512. Clean calibration and held-out accuracy must each reach 90%, and every class must supply at least 32 examples per scored split. An unsuccessful seed remains in the report and prevents a completed three-seed verdict.

The learned arm calls the repository's `param_decomp.run_param_decomp.optimize` to fit 32 rank-one head components. Its objective includes the existing adversarial reconstruction term, importance minimality, and weight faithfulness. The fixed schedule performs 200 warmup updates followed by 300 main updates. Head SVD appears only in the explicitly named task-gradient baseline, where it supplies the best rank-one approximation to the training-loss gradient.

Each learned candidate and each of 32 independent Gaussian outer-product controls is normalized to 1% of the original head's Frobenius norm. The selector tests both signs on calibration veto examples with equal candidate budgets and chooses the lowest cross-entropy. The gradient baseline receives the same rank and dose. Clean inference and a zero edit provide baseline and sham. Full-model and cached-head intervention paths must agree within `2e-6` maximum absolute logit error, and restored weights must reproduce clean inference. Components-only fidelity and residual-restored fidelity are recorded separately. The residual's ability to restore a forward pass supplies no conclusion about the usefulness of individual learned directions.

The primary seed effect is target-veto negative log-likelihood for the selected random arm minus that for the learned arm. A positive value favors learned steering. Each seed receives a paired 2,000-resample bootstrap interval; the mean across independent seeds receives a 95% Student t interval with two degrees of freedom. Secondary outcomes include the task-gradient comparison, target accuracy, overall accuracy, commit/repair collateral accuracy, and false commits on veto examples. The complete report retains literal edit matrices, selected directions, per-example scores, runtime, and failed gates.

The frozen positive criterion requires every prerequisite to pass, a positive primary effect in every seed, an across-seed lower interval bound above zero, and at most two percentage points of collateral accuracy loss in each seed. A completed run may produce a negative or inconclusive finding. Execution success remains an engineering status.

**First local execution: NO_REGISTERED_ADVANTAGE.**

The frozen synthetic recurrent-gate protocol was executed on local CPU. The comparison completion flag is True. Registration commit: `f1e3e9ba663ea906a6468d76426bb7381a6e23ad`. Protocol SHA-256: `c6bd49b7423091f02fd55812a51a24aae8f4f07332b3178e006a64a015f908a9`. These tables retain the first local execution values. Section 6.4 gives the subsequent hosted reproduction.

| Prerequisite or pilot | Verdict | Observed evidence |
| --- | --- | --- |
| Recovered checkpoint task/replay gate | BLOCKED_ARCHITECTURE_PROVENANCE | 11,191,328 parameters; 91 frozen tensors checked; exact equality=True; task performance evaluated=False. |
| Recovered residual: bucket_head | MATRIX AUDIT | 16 learned components; residual/weight Frobenius norm=0.203684. |
| Recovered residual: networks.0.layers.0.1.to_out | MATRIX AUDIT | 32 learned components; residual/weight Frobenius norm=0.783704. |
| Recovered residual: networks.0.layers.1.1.ff.2 | MATRIX AUDIT | 32 learned components; residual/weight Frobenius norm=0.798766. |
| Auxiliary pilot seed 997 | PILOT_CALIBRATION_ONLY | Clean calibration accuracy=0.966797; runtime=4.48 s; held-out evaluated=False. |
| Pilot replay diagnostics | CALIBRATION ONLY | Components-only KL=0.000003; argmax agreement=1.000000; residual-restored maximum logit error=0. |
| Retained pilot failure, seed 997 | ENGINEERING_FAILURE | The initial pilot failed Config validation: slow_eval_freq must be a multiple of eval_freq. The failed receipt is retained. |

### 6.3 Local held-out comparison

| Seed | Clean test accuracy | Veto rows | Random minus VPD NLL | 95% paired-row interval | Runtime (s) |
| --- | --- | --- | --- | --- | --- |
| 103 | 0.9771 | 641 | 0.001378 | [0.000978, 0.001800] | 4.17 |
| 211 | 0.9712 | 627 | 0.000629 | [0.000303, 0.000981] | 3.13 |
| 307 | 0.9673 | 669 | 0.000664 | [0.000361, 0.001000] | 3.43 |

The mean primary effect across seeds is 0.000890402, with a 95% t interval [-0.000159859, 0.001940663]. The recorded verdict is `NO_REGISTERED_ADVANTAGE`. Every seed passes its prerequisites. Collateral gate passed: True. The across-seed interval includes zero. Each within-seed interval is positive; those intervals quantify row variation conditional on one trained model. The across-seed interval treats the independently trained model/decomposition as the replication unit.

![Local synthetic follow-up effects and uncertainty](figures/followup_effects.png)

*Figure 3. Positive values favor learned steering over the selected random arm. Seed intervals use paired-row bootstrap; the bottom interval uses the mean of three independent seed effects and Student t with two degrees of freedom. Both estimate the same contrast at different levels of variation.*

| Seed | Arm | Target NLL | Target accuracy | Collateral delta (pp) | False commits / veto |
| --- | --- | --- | --- | --- | --- |
| 103 | clean | 0.066945 | 0.9704 | 0.0000 | 0.0094 |
| 103 | sham | 0.066945 | 0.9704 | 0.0000 | 0.0094 |
| 103 | vpd | 0.064486 | 0.9750 | 0.0711 | 0.0062 |
| 103 | random | 0.065864 | 0.9704 | 0.0000 | 0.0094 |
| 103 | task_gradient | 0.062824 | 0.9735 | 0.0000 | 0.0094 |
| 211 | clean | 0.033766 | 0.9856 | 0.0000 | 0.0080 |
| 211 | sham | 0.033766 | 0.9856 | 0.0000 | 0.0080 |
| 211 | vpd | 0.032650 | 0.9856 | -0.0704 | 0.0080 |
| 211 | random | 0.033279 | 0.9856 | -0.0704 | 0.0080 |
| 211 | task_gradient | 0.032200 | 0.9872 | -0.1408 | 0.0080 |
| 307 | clean | 0.024806 | 0.9940 | 0.0000 | 0.0030 |
| 307 | sham | 0.024806 | 0.9940 | 0.0000 | 0.0030 |
| 307 | vpd | 0.023592 | 0.9940 | -0.1450 | 0.0030 |
| 307 | random | 0.024256 | 0.9940 | 0.0000 | 0.0030 |
| 307 | task_gradient | 0.023246 | 0.9940 | -0.1450 | 0.0030 |

The task-gradient baseline has lower target NLL than learned VPD in 3 of 3 seeds. Clean and sham outputs agree. VPD collateral accuracy changes in percentage points are 103: +0.0711, 211: -0.0704, 307: -0.1450. Model parameter counts: 1,763. The measured seed runtimes include model training, decomposition, selection, and scoring; environment setup and archive recovery are outside that timing. The CPU was AMD EPYC 9V74 80-Core Processor with 2 PyTorch threads. The frozen source used Python 3.12.14 and PyTorch 2.5.1+cpu. Eleven package versions accompany the result receipt; a contemporaneous full transitive package freeze is absent. The later Actions workflow also archived `pip freeze` [R11].

Residual-restored replay has maximum absolute logit error 0.0. Components-only calibration KL ranges from 1.900e-06 to 2.024e-06, with minimum argmax agreement 1.0. The literal edits, checksummed per-example held-out scores, loss traces, and selected candidate receipts remain available for replay.

Recovered-model missing prerequisites: experiments.hermes_skill_gym.train_hrm_hermes.HermesHRMClassifier; experiments.hermes_skill_gym.hermes_dataset.HermesSkillDataset; independent original forward logits for replay validation; historical x-transformers/package versions. These gaps prevent a recovered-task steering comparison.

Positive NLL differences favor VPD. The primary interval uses three independent model/decomposition seeds and two degrees of freedom; within-seed row intervals describe a separate level of variation. Secondary outcomes retain descriptive status.

The new experiment concerns a generated commit/veto/repair distribution with output-head edits. Its evidence is separate from the recovered arena, ARC, Wordle, governance and adapter records.

### 6.4 Hosted reproduction

The hosted archive audit in Actions run 44449171571 regenerated 38 comparable audit and paper-output files byte-for-byte. Steering replay 44449171565 executed the frozen synthetic method on GitHub Actions and passed 265 independent verifier checks over 6,144 held-out example records. The first hosted attempt had completed scoring before its verifier failed because a Git lookup still used the prior repository's paths. A relocation wrapper corrected that lookup while preserving the registered scientific code. Both attempts remain identified in the replay ledger [R11].

| Seed | Hosted clean accuracy | Random minus VPD target NLL (nats) | VPD collateral accuracy change (percentage points) |
| --- | --- | --- | --- |
| 103 | 0.977051 | +0.001377650 | +0.071073 |
| 211 | 0.971191 | +0.000628515 | -0.070375 |
| 307 | 0.967285 | +0.000663796 | -0.145030 |

The hosted mean is +0.000889986967 nats, with a 95% Student t interval of [-0.000160050097, +0.001940024031]. The difference from the first local mean is -0.000000414885 nats. The prerequisite gates pass, and the verdict remains `NO_REGISTERED_ADVANTAGE`. This reproduction establishes numerical portability of the frozen comparison across the recorded environments. Its evaluation examples were already exposed during the first local run, so the evidence concerns replay of that experiment [R11].

## 7. Native recurrent-weight experiment

### 7.1 Model and physical intervention

DEUTERIUM-NATIVE-01 was registered separately at commit `d808db9122a60131c395804d7780bb3adec80fa7`. It uses the original `TinyRecursiveModel(48,4)` from the LOOPSCOPE source, with 44,001 parameters and one reasoning block reused four times. Each input contains 81 independent digits from 1 through 9. The target at each position is the previous digit under a cyclic shift. A deterministic copy operation solves this task exactly; the trained network supplies a controlled setting for measuring how a recurrent parameter edit affects its learned computation [R12].

The frozen teacher roots are 103, 211, and 307 from the original LGA-02 artifact. The teacher bundle SHA-256 is `d32f249385df2eabefcacbee15c9ab2cccc73f01452300f275a11f621106a266`. Source classes and metadata were checked before fitting. Direct forward inference and a manual execution of the same four tied blocks agreed within the registered tolerance, with zero-edit identity and complete weight restoration checked separately. These checks authenticate internal execution of the recovered native architecture. Independently archived per-token reference logits from its original training run remain unavailable [R12].

The intervention targets `block.chan_mix.2.weight`, a 48 by 128 matrix inside the shared reasoning block. A candidate adds the same physical matrix at all four uses of that weight. The VPD fit uses the unchanged pinned upstream implementation at `2df0092d5be4d8e4748b3c66ef8325dedef17549`, with 16 rank-one components fitted on the teacher's original 512 training strings. There are 32 faithfulness warmup updates and 64 main AdamW updates at learning rate 0.001, using batches of eight strings and two projected-gradient adversarial mask steps. The warmup applies weight decay 0.1 to the component factors; the main optimizer uses zero weight decay. Source configuration preserves the loss coefficients and gradient-clipping defaults. The terminal index performs logging and saving after the 64 main updates [R12].

Each learned candidate is an outer product of one fitted factor pair. Its Frobenius norm is scaled to 1% of the original target weight's norm, with the actual FP32 applied delta checked against that dose. Sixteen Gaussian outer products receive the same normalization. Both banks search both signs on 256 calibration strings from seed 731055, for 32 comparisons per bank. Selection uses the mean correct-digit negative log likelihood at eight fixed positions, `(0,10,20,30,40,50,60,70)`, with the registered index/sign tie rule. A rank-one negative task-gradient control uses the original training labels. Its supervised objective gives it a different source of information, so its outcome is reported descriptively [R12].

### 7.2 First scoring and audit recovery

First scoring in Actions run 44465490737 used the 256 reserved strings from seed 731057. All teachers shared that input cohort. The scorer saved the complete 81 by 10 output-logit array for each source under 17 declared arms, including the primary shared-weight edits and the descriptive dose/depth controls. An exposure marker preceded test generation. The three teacher checkpoints had already been examined in earlier LOOPSCOPE work, giving this study its registered development-only scope [R12].

The first scoring jobs completed, after which all independent auditors failed on an unbound variable in a target-row shape check. The recovery copied the original auditor and changed the expression `len(row)` to `len(y[i])`. It read the first saved artifacts, retained their source locks and checksums, and generated no model predictions or new source strings. Recovery run 44467552827 passed. Its standard-library implementation recomputed the metrics from all 10,575,360 saved scalar logits across the three teachers. The original failed audit remains preserved, together with the correction's distinct provenance [R12].

The primary within-teacher quantity is the paired difference between selected-random and selected-VPD goal NLL, averaged within each whole string. There are 2,000 bootstrap resamples of the 256 paired string means. The across-teacher interval uses the three mean effects and a Student t critical value with two degrees of freedom. All numbers below come from the recovered arithmetic receipt, whose floating-point precision differs slightly from the original FP32 scorer [R12].

| Teacher | Clean digit accuracy across all 81 positions | Random minus VPD goal NLL (nats) | 95% paired-string interval |
| --- | --- | --- | --- |
| 103 | 98.7895% | +0.000041767 | [-0.000111491, +0.000196867] |
| 211 | 98.4182% | -0.000147249 | [-0.000268377, -0.000021654] |
| 307 | 98.7124% | +0.000027239 | [-0.000115815, +0.000181753] |

The mean is -0.000026081008 nats. Its 95% across-teacher interval is [-0.000287376525, +0.000235214509]. The registered criterion requires a positive effect in every teacher and a positive lower across-teacher interval bound, subject to the collateral gate. The verdict is `DEVELOPMENT_NO_REGISTERED_ADVANTAGE`. VPD changes off-goal digit accuracy by +0.010702, 0.000000, and -0.010702 percentage points for teachers 103, 211, and 307 respectively, so collateral behavior passes the registered two-percentage-point limit. The label-supervised task-gradient control reaches lower goal NLL than VPD in every teacher [R12, R13].

![Steering effects in completed studies](figures/completed_steering_effects.png)

*Figure 4. Random-minus-VPD loss effects in the completed synthetic head-edit and native recurrent-weight studies. Positive values favor the selected VPD direction. Each panel shows three model effects and the across-model mean with its 95% Student t interval. The panels use their respective effect scales. The native teachers were previously examined and share one 256-string test cohort, so their interval is a descriptive summary of those checkpoints. Hosted reproduction of the synthetic study repeats the previously exposed experiment. The panels represent distinct tasks and are reported separately.*

Clean accuracy here averages digit correctness over positions and strings. Exact whole-string success would require every one of the 81 predictions to be correct and is a different endpoint. The common test cohort also creates a shared source of variation across teachers. The three-root interval is the registered descriptive summary over these previously examined checkpoints; the 20,736 digit positions per teacher do not supply additional independent model replications. An equivalence claim would require a declared equivalence margin and enough precision to place the entire interval within it [R12, R16].

## 8. Recurrent importance masks and adapter validation

### 8.1 What the fitted stock implementation does

A source-only diagnostic loaded each original fitted VPD checkpoint from the first calibration run, requiring its saved teacher tensors to match the original bundle. The diagnostic used eight old training strings from seed 31041. A separate native hook recorded four distinct inputs to the shared target module. The stock `ComponentModel` cache contained one tensor under that module path, exactly equal to the fourth input. Its importance network therefore computed a mask from the final clean pass, and masked replay supplied the identical selected mask tensor at every native application [R13].

The same fitted importance network was then evaluated on each recorded clean input. The table gives mean absolute differences from its fourth-pass mask, conditional on this eight-string batch.

| Teacher | First application | Second application | Third application | Fourth application |
| --- | --- | --- | --- | --- |
| 103 | 0.27464578 | 0.18320383 | 0.08888695 | 0 |
| 211 | 0.26785272 | 0.18549871 | 0.09476270 | 0 |
| 307 | 0.29798546 | 0.20903246 | 0.10494941 | 0 |

These measurements identify how the pinned implementation handles a reused layer and quantify the masks that the original learned network assigns to its earlier states. Each row uses one fitted importance network shared across depth. The instrument did not fit separate per-pass networks. Its findings motivate a controlled change in mask semantics; attributing the previous steering outcome to that change would require the paired experiment described below [R13].

### 8.2 Pass-indexed source-only implementation

DEUTERIUM-NATIVE-02 registers that comparison before opening new training roots. Its primary adapter records the clean input at each actual invocation and uses one shared importance network to produce a mask for that invocation. Factor parameters and the physical target matrix remain shared. The projected-gradient adversary retains the stock source shape of `[batch,81,17]`, comprising 16 component coordinates and one residual coordinate. This same adversarial source is combined with the four depth-specific importance floors. The policy obtains its masks from the original clean states [R14].

PR 225 merged the adapter at `7de0526747fd7553a6c3b048ebdcf511f05b1631`. The final source-only Actions job, run 44483849471, passed nine tests on an untrained excluded model and data seed 61077. They cover four-call activation capture, native output parity, and matching losses and parameter/importance gradients when the masks are forced identical across passes. A one-update AdamW comparison checks the complete configured loss combination, while the native per-pass policy receives finite-loss and gradient checks. These tests establish a specific engineering result at the implemented boundary [R14].

The registered scientific comparison reserves 12 new teacher roots, each with independent training and evaluation source seeds. Its primary teacher effect is stock-last goal NLL minus pass-indexed goal NLL after a fixed equal-dose candidate search. Both learned arms share the same selected random bank. The protocol specifies a 0.0002-nat development effect threshold with a positive across-root interval and at least nine positive roots, together with clean-competence and collateral gates. These thresholds describe an exploratory engineering target. A root-level power and sham-resolution study would be required for a calibrated generality claim [R14, R16].

### 8.3 Full-schedule engineering gate

A subsequent source-only comparison exercised the full pinned optimizer schedule using an excluded untrained teacher and input seed 61077. The stock and adapter arms started from the same factor and importance-network state. Both paths used 32 faithfulness warmup updates and 64 main updates with matched source batches and random-number states. An identity policy supplied the final-pass mask at every depth, allowing the adapter trajectory to be compared with stock under the same mask semantics. The full-schedule check retained the 0.00002 numerical tolerance used in the pointwise tests. The original protocol specifies pointwise equivalence; applying that threshold to the entire optimizer trajectory is an added source-only engineering criterion [R14, R17].

The warmup states matched exactly. At the first main step, the recorded loss was equal, the maximum gradient difference was approximately 9.3e-10, and the first parameter-update difference was approximately 7.45e-9. Differences accumulated during the later adversarial fitting trajectory. The largest loss difference was 0.000583649, the largest parameter difference 0.000358582, and the largest gradient difference 0.000115089. These values exceed the fixed threshold. The receipt records `BLOCKED_UNMATCHED_OPTIMIZER`, and the scientific release remains closed [R17].

This extended check compares entire training trajectories. The earlier pointwise identity tests retain their passing result. A new guard test also accommodates negative lower-mask leakage allowed by the pinned `leaky_hard` function, and the resulting source-only suite passes 10 tests. The native per-pass policy completes a finite-loss trajectory. Small floating-point differences interacting with sign-based projected-gradient updates are a hypothesis for the observed drift; the comparison leaves its cause unresolved. Establishing whether the two paths retain an equivalent objective at a scientifically adequate numerical resolution requires further source-only work [R17].

All 12 newly registered teacher roots and their reserved calibration and held-out inputs remain unopened at this report's cutoff. Their prospective comparison has produced no task result. Resolving the engineering gate, with any changed criterion declared before new-source exposure, is a prerequisite for that separate study.


## 9. Interpretation and limits

The historical acceptance rule identifies small output changes among interventions tied on its task rank. The new studies ask whether a learned direction improves a supervised endpoint under a symmetric calibration search. The shared term *steering* covers these distinct measurements, so their results require separate denominators and evidence labels. The native experiment expands the physical intervention into the recurrent block, which changes the state trajectory at each of its four uses. Its registered advantage criterion remains unmet.

The trained models and selected candidate banks are the units needed to assess variability in method performance. Multiple tokens within one input share both computation and intervention. The source bootstrap respects that dependence within each native teacher, while the across-teacher summary remains limited to three previously studied checkpoints and one generated task family. Neither study supplies a calibrated estimate of the probability that VPD improves a new language model. Claims about broad method effectiveness would require independently sampled model roots and a task population that supports the intended generalization [R16].

The cache diagnostic narrows an implementation question to a measurable choice: which reached state supplies each importance mask for a tied recurrent layer. A new comparison requires fitting paths that match in effective update count and objective scale, together with an accepted numerical verification gate. Factor count, adversarial-source dimensionality, and the search budget also enter that comparison. Reusing a fitted importance network on earlier activations gives descriptive mask differences; a randomized or prospectively paired task experiment is needed to connect those differences to steering efficacy.

The recovered 11,191,328-parameter classifier retains its blocked execution status. Matching 91 frozen tensors authenticates the stored parameters. Its original forward class and dataset/tokenization implementation are still required to reproduce task predictions, alongside independent reference logits and the historical package environment. The 51 recovered task groups that had no training-context match remain unscored. Missing provenance limits the claims supported by those records and defines what a future recovery would need to supply.

Recursive improvement would additionally require a sequence of retained edits whose gains survive fresh evaluation after each update. The historical stacked-edit searches and the controlled single-edit studies in this report leave that claim open. Search effort, source-family reuse, and changes in the evaluation set would need to be recorded throughout such an experiment. The current report closes the historical audit and the completed single-edit comparisons at their measured scope.

## 10. Reproduction and data availability

The active project is under `mechinterp/deuterium/` in `MoralityLabAI/LoopedTransformers`. The original retrospective manuscript, its generated tables, and its render manifest are retained as the first local version. This completed revision adds the hosted replay and native-model evidence without altering the historical archive or either scored protocol. The record renderer and the completion renderer execute from saved evidence; neither rendering step generates reserved test data.

`experiments/deuterium/README.md` contains the archive-recovery commands and pinned dependency instructions. The hashed 1.33-GB source archive remains at its pinned dataset revision [R2]. The hosted audit ledger identifies the relocation wrappers and successful replay artifacts [R11]. Native source locks, the first scoring job, and the independent recovery identify the exact experiment behind Section 7 [R12]. Actions artifact IDs and digests accompany those records, subject to the host's retention policy. The completed report preserves the numerical audit receipts and source references needed to distinguish fresh scoring from later arithmetic verification.

| Evidence boundary | Status at this revision |
| --- | --- |
| Earlier full-arena sweep | Outside the supplied package; its counts retain summary-only status. |
| Historical run-to-commit identity | Recovered source supports the reviewed logic; the exact execution attestation is incomplete. |
| Arena and ARC generalization | Repeated search and example reuse leave prospective task-disjoint confirmation open. |
| Wordle lineage | Recorded identifiers support the reported overlap audit; upstream episode lineage is incomplete. |
| Governance and historical adapters | Exact invocation and selected-row manifests are incomplete; intermediate-update artifacts are missing for governance. |
| Later basis-search addendum | Values retain recovered-manuscript status; candidate-level reconciliation is outside this audit. |
| Recovered large-model predictions | Blocked by original execution and reference-logit provenance. |
| Synthetic learned-direction comparison | Executed locally and reproduced on Actions; registered advantage criterion unmet. |
| Native recurrent-weight comparison | First scored artifacts independently audited; registered development advantage criterion unmet. |
| Per-pass importance policy | Pointwise source tests pass; full-schedule parity exceeds its fixed tolerance and blocks new-root release. |

The source-only adapter tests use excluded development data and are suitable for repeated engineering verification. Any later scoring of the already exposed native seed 731057 is a reproduction. DEUTERIUM-NATIVE-02 retains its separate pre-exposure source-lock requirement and its own reserved cohorts [R14].

## References and artifact ledger

**[R1] Historical manuscript.** `archive/original_vpd_trm_feature_steering_paper_2df0092.md`, copied byte-for-byte from repository commit `2df0092d5be4d8e4748b3c66ef8325dedef17549`. SHA-256: `45822fdda4460fa36641aac54af5b8a5d828f98d69c07cf8bba5a40e0f3072da`. The adjacent archive manifest records its original path and Git blob. The longer recovered manuscript is preserved separately [R10].

**[R2] Recovered records.** [Hugging Face dataset at the pinned revision](https://huggingface.co/datasets/AlephFunk/vpd-trm-original-records-202606/tree/b208d547d20e89dc6c221bc640525320cc67a2f8). Archive SHA-256: `a45db5ef5e603852efb4f247112655c5572eff5aa85f9ab94674a4ae9dc8f003`. Member paths and hashes: `../../experiments/deuterium/results/archive_manifest.json`. The archive's `MANIFEST.json`, `reconciliation.json`, and `MISSING_FILES.md` retain the recovery author's metadata. `tables/source_ledger.csv` maps the files used in this audit to complete member paths and digests.

**[R3] Arena records and source.** Archive prefix `sources/D/Research_Engine/runs/`; replication directory `metta_organelle_arena_paper_20260601T024147Z`; earlier summary directory `metta_organelle_arena_full_20260531T180830Z`. The recovered implementation is `sources/D/projects/VDP/param_decomp/experiments/metta_organelle_arena.py`. Recomputed counts and line-addressable accepted rows are under `../../experiments/deuterium/results/retrospective/`.

**[R4] ARC score cards and cross-validation.** Four recovered ARC Challenge cards for seeds 23, 37, 101, and 151; raw paths and hashes appear in the retrospective audit. The condition cross-validation source is preserved in the repository. Generated overlap and held-out-identity tables record the comparison used here.

**[R5] Wordle.** Archive run `sources/D/Research_Engine/runs/trm_wordle_vpd_hook_probe_20260603/`, including the four- and eight-example cards and `overnight_validation_32x2_20260604_retry_smallgpu`. Scoring and replay source appear in `sources/D/projects/VDP/param_decomp/experiments/`. The retrospective outputs identify exact card paths and distinguish stored summaries from recomputed policy scores.

**[R6] Historical roundout.** Archive run `sources/D/Research_Engine/runs/vpd_trm_paper_roundout_20260604/`, including `paper_experiment_manifest.csv`, `paper_metric_table.csv`, `paper_policy_comparison.csv`, and `paper_claim_ledger.csv`. Recovered summaries carry historical claim labels that this revision reassesses.

**[R7] Governance.** Recovered files `reports/trm_governance_report.md` and `scripts/trm_governance_test.py` within the original project tree, with their complete archive-member paths identified in the inventory. Missing execution and intermediate-update provenance is documented in the archive's `MISSING_FILES.md`.

**[R8] Component adapters.** Recovered original-project files `reports/powered_int3_validated_vpd_zero_step_report.md`, `reports/powered_int3_validated_vpd_gated_adapter_report.md`, and `scripts/run_powered_int3_vpd_gated_adapter.py`. Their archive-member paths, result JSON, and dataset identities appear in the retrospective audit.

**[R9] Follow-up and reproducibility.** `../../experiments/deuterium/PROTOCOL.md`, configuration and dependency pins, workflow source, and recorded outputs under `../../experiments/deuterium/results/`. The reproducibility README records the local freeze and hosted-run status.

**[R10] Recovered expanded manuscript.** `archive/recovered_vpd_trm_feature_steering_paper.md`, copied from archive member `sources/D/projects/VDP/papers/VPD_TRM_Feature_Steering/vpd_trm_feature_steering_paper.md`. SHA-256: `0bcfa35bf6f4599ce2b521ee5655bda6370a86cee8529cef9701f7f9dd01dc6b`. This copy contains additional June 9-10 basis-search sections absent from [R1]. A historical Git commit has not been inferred for this version.

**[R11] Hosted reproduction.** [Replay audit and source ledger](https://github.com/MoralityLabAI/LoopedTransformers/blob/main/mechinterp/deuterium/HOSTED_REPLAY_AUDIT_20261119.md). Successful steering [Actions run 44449171565](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44449171565), artifact 14493919867. Historical audit [run 44449171571](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44449171571), artifact 14494951200. The ledger retains the failed first hosted verifier run and source relocation details. The byte-preserved [hosted paper summary](evidence/hosted_gate/paper_summary.json), [verification receipt](evidence/hosted_gate/verification.json), and adjacent runtime files supply the numerical replay evidence.

**[R12] DEUTERIUM-NATIVE-01.** [Frozen protocol](../../native_trm/PROTOCOL.md), original teacher source and fitting implementation under `../../native_trm/`, and [first-audit recovery](../../native_trm/stagec/recovery/README.md). First calibration [run 44463652662](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44463652662); first scoring [run 44465490737](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44465490737); recovered audit [run 44467552827](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44467552827). Recovery artifact 14502718352 has ZIP SHA-256 `9e3e71808d3e1df8e21edcbd65ff57132afcf5d46cb88c0559e2106408848e0f`. The byte-preserved [registered verdict](evidence/native01/registered_verdict.json) and adjacent per-teacher audit JSON provide the numbers in Section 7; [artifact provenance](evidence/native01/provenance.json) records their retrieval.

**[R13] DEUTERIUM-ADAPTER-01.** [Original fitted-checkpoint cache diagnostic](../../native_trm/adapter_audit/RESULTS_SOURCE_ONLY_20261119.md), [Actions run 44468845945](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44468845945), artifact 14503228548. ZIP SHA-256 `4c6b72d949dddd3d0e78880f9c80174633a062467e390d9a443fd68f5c074f97`. The byte-preserved [fitted-checkpoint observation](evidence/cache_audit/old_train_source_only.json) and [audit receipt](evidence/cache_audit/trained_audit.json) preserve the measured masks and their provenance. The distinct earlier untrained-CI diagnostic is labeled separately in the source report.

**[R14] DEUTERIUM-NATIVE-02.** [Prospective protocol](../../native_trm/adapter02/PROTOCOL.md), initial registration commit `0512c69d63a16b58fbcf838292d1394f4d8d18e3`; adapter [PR 225](https://github.com/MoralityLabAI/LoopedTransformers/pull/225), merge commit `7de0526747fd7553a6c3b048ebdcf511f05b1631`. Final source-only [run 44483849471](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44483849471), job 135422360677, records nine passing tests. The scientific protocol is separately preserved from subsequent engineering changes.

**[R15] VPD method.** Bushnaq, L.; Braun, D.; Clive-Griffin, O.; Bussmann, B.; Hu, N.; Ivanitskiy, M.; Linsefors, L.; and Sharkey, L. [Interpreting Language Model Parameters](https://www.goodfire.com/research/interpreting-lm-parameters). Goodfire, May 5, 2026. Experiments here pin the implementation fork to `MoralityLabAI/param-decomp` commit `2df0092d5be4d8e4748b3c66ef8325dedef17549`.

**[R16] Experimental-unit and resolution policy.** [EvalMath mechanistic-resolution preflight](https://github.com/MoralityLabAI/EvalMath/tree/main/skills/evalmath-mechinterp-power) and [FSI-01 finite-sample inference](https://github.com/MoralityLabAI/EvalMath/blob/main/paper/finite_sample_inference.md). These project methods require explicit independent units and distinguish descriptive small-root intervals from prospectively supported generalization.

**[R17] Full-schedule source-only audit.** The [schedule receipt](../../native_trm/adapter02/results/source_schedule_20261119/schedule_receipt.json), [paired update comparison](../../native_trm/adapter02/results/source_schedule_20261119/trajectory_comparison.json), and [independent audit](../../native_trm/adapter02/results/source_schedule_20261119/schedule_audit.json) preserve the stock/adapter result. The fixed 0.00002 trajectory threshold yields `BLOCKED_UNMATCHED_OPTIMIZER`; source roots are excluded from the prospective scientific study.
