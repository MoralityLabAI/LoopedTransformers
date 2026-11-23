# LSA-02G: Native Lean fork-choice competence, source weighting, and matched sham

**Registration:** November 23, 2026. **Status:** development-only CPU research before any full gym. This follows the [LSA-02F 96-byte pretty-dose study](../reports/lsa02f_pretty_dose_uniform_cpu_20261123.md). That experiment trained ten native Plan B checkpoints, found a 96-byte public pretty-text development maximum at 7/32 certified proofs, and established a genuine model-free uniform-menu native baseline at 15/96 = 15.625%. Its 21.875% best model completion failed the existing ≥40% competence requirement, and the predeclared pp96-versus-pp0 +5-point screen missed by 0.3125 percentage points.

## Question

The public candidate menu often contains meaningful proof-choice forks. The LSA-02B audit found that a large fraction of source frames were forced singletons and that in-sample root-weighted optimal-action accuracy on real forks only modestly exceeded random choice. Can increasing the **relative training loss weight of verified shortest-continuation forks** improve correct action ranking on independently generated *native Lean* states and, ultimately, standalone-certified proof completion?

This experiment measures that claim with a **same-source shuffled-loss sham** to separate useful fork targeting from generic stronger loss emphasis.

## Frozen source and training arms

Freeze a new suite with generator seed `20261129`, 64 roots in each `train`, `calibration`, `ordinary_test` and `ood_family` shard. The first **64 independent training roots** are the only roots used to build the teacher and fit model parameters. Native Lean 4.19 `lgym` explores their AND/OR proof graphs and certifies every reference proof with a separately compiled axiom-free Lean declaration. The **first 32 calibration roots** are a disjoint development cohort, held out from optimizer updates. Ordinary-test/OOD proof outcomes remain sealed.

For a train-only reference-path decision frame \((s,A)\), a tactic receives shortest-continuation label \(y_a=1\) iff its Lean-verified transition achieves the minimum proof size from that state. A frame is a **discriminating fork** precisely when its public action menu has at least two tactics and its native labels contain both 0 and 1. The definition does not interpret \(y_a=0\) as a Lean rejection: a nonshortest tactic may still eventually produce a proof.

Three training conditions all consume *identical* model-visible inputs and sampled source frames:

| Condition | Training loss weight per native source frame |
| --- | --- |
| `plain` | Every frame weight 1, historical reference-path baseline |
| `fork4` | Discriminating fork weight 4, other frames weight 1 |
| `sham4` | Exactly the same number of randomly selected train-only frames assigned weight 4, others weight 1 |

The `sham4` selection is a separate seeded source permutation of frame lineage/step identifiers. It uses the frozen teacher only for the **count** of boostable frames, and it never consults calibration labels or model-generated proof results to choose boosted rows. The total source loss mass is **identical** for `fork4` and `sham4` by dividing both per-sample losses by \(1+3K/N\), where \(K\) is the number of native training forks and \(N\) the number of training frames. The `plain` arm retains weight 1. Source sampling stays **root-uniform, then within-root uniform**, with exactly the same deterministic root/frame draws across all three arms.

## Matched CPU contract

| Axis | Fixed registration |
| --- | --- |
| Model | Plan B, 57,475 parameters, width 32, four heads, 384-byte text encoder |
| Input view | `pp96` for every model: mandatory typed goal/candidates + up to 96 public Lean pretty bytes |
| Recurrence | AB, two middle-block passes and identical tied parameters |
| Seeds | 149, 151 |
| Training budget | 720 FP32 AdamW optimizer steps, LR 0.001, global gradient clip 1 per model |
| Source corpus | 64 independent native Lean proof roots |
| Execution | One GitHub-hosted CPU worker, one Torch thread, deterministic algorithms and disabled MKLDNN |
| Repeatability | Each seed/condition independently fitted **twice** on the same worker, 12 fits total |
| Equality checks | Same-seed bitwise initial state, identical optimizer source-root/frame stream, same physical model architecture |
| Proof evaluation | 32 fresh independent calibration roots, eight tactics, true Lean kernel and standalone compilation |
| Native controls | AB/BA frozen-order intervention, public-order `auto8`, native model-free shortest attainability |
| Uniform benchmark | 5 source-blind random-menu seeds 17,31,47,61,83, 160 native episodes on same 32 roots |
| GPU/payments | None: GitHub CPU and RAM only |

The CPU run checks exact tensor and optimizer-trace equality for each duplicate fit. The native model uses frozen weights during inference. Every completed proof is independently compiled as a standalone Lean declaration; every episode is replayed without the neural model. The uniform policy selects tactics only from the public live menu, independent of native proof labels and logits, and undergoes the same kernel/compilation/replay gates.

## Independently verified calibration fork accuracy

A separate **post-native-scoring** job opens the exact 32 development calibration theorem roots under Lean 4.19, explores their shortest-path reference proofs and publicly available action transitions, and **independently compiles every reference proof again**. Its outputs are recorded in a separate, hashed native calibration fork corpus.

The metric includes only **discriminating reference-path tactic states on roots whose shortest reference proof fits within the eight-tactic budget**. Each frozen checkpoint then predicts a tactic by argmax of its AB readout on the same public `pp96` state and menu. A prediction is marked correct only when the corresponding **independently Lean-verified shortest-continuation label** is 1.

Report exact numbers of eligible roots, forks, and root-weighted action accuracy, along with the uniform-in-menu shortest-choice probability averaged by theorem root. These are **reference-trajectory** metrics, so their interpretation is limited to independently verifiable states *along shortest reference proofs*, rather than all off-policy model-visited states. Calibration fork labels are never sent back to the optimizer, source sampler, hyperparameter tuning job or inference-time model.

## Registered endpoints and decisions

The **primary scientific-development endpoint** is **`fork4` minus `plain` root-weighted correct optimal-action rate** on fresh certified calibration forks. The **matched-sham contrast** `fork4 - sham4` tests whether any improvement follows meaningful fork targeting rather than generic sample weighting. All these predictions use fixed frozen AB models and identical reference-path states.

Secondary endpoints are **whole-proof closure** on the same 32 roots, comparing all three arms under the same Lean verifier and action budget. Each root's model-seed outcomes are averaged before a descriptive 2,000-draw bootstrap over **unique theorem lineages**. Report training-only fork accuracy separately; it cannot substitute for held-out development action choice or native proof closure. The independent uniform-menu native closure mean is a separate full-proof baseline, not a substitute for reference-path action accuracy.

**Predeclared development screen:** at least **+0.05 absolute root-weighted reference-path fork-action accuracy** for `fork4` over **both `plain` and `sham4`**, with nonnegative mean native full-proof completion effects against both. A screen pass is **not** confirmatory architecture evidence. A failure, including reversal on full-proof closure, is a valid negative development result and stays archived.

**Formal competence gate:** at least 32 independent development roots, ≥0.40 certified trained-order full-proof completion, and at least +0.10 over the registered five-stream native uniform-menu closure rate. Satisfying that floor would permit registering **fresh sealed confirmation plus FSI-01** after independent root ancestry, random-baseline uncertainty and matched-sham audit. It does not automatically promote the architecture or justify a Mathlib transfer claim.

## Evidence and fail-closed protocol

The [LSA-02G GitHub workflow](../.github/workflows/lsa02g-fork-competence-cpu.yml) has native source compilation, same-worker duplicate training, independent uniform-menu scoring, six paired native model proof jobs, and the **post-proof native reference-fork assessor**. The assessor is dependency-gated on all native model proof jobs to prevent use of development labels during fit or inference.

All file hashes, source lineages, checkpoint/seed/emphasis identities, source weight counts, native certificates, independent replay receipts, identical calibration theorem rosters and pretrained-model-free conditions are authenticated before statistical aggregation. The network-isolated suite generator's external corpus ancestry remains unverified; even a pass would be scoped to generated Lean Horn proofs.

Save the numerical CI receipts as GitHub Actions artifacts with 14-day retention, then commit a permanent research report. The calibration source is exploratory and cannot be repurposed as a sealed test set in later experiments. The `ordinary_test` and `ood_family` shards remain untouched.
