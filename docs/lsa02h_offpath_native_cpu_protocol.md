# LSA-02H: Native Lean recoverable off-reference proof states

**Registration:** November 23, 2026. **Evidence scope:** development-only, CPU/RAM before a full gym. This study follows [LSA-02G](../reports/lsa02g_native_fork_competence_20261123.md), where 4× emphasis on training-reference forks failed the root-weighted Lean calibration fork-accuracy gate twice. The previous verified proof policies visited states outside their shortest-reference training trajectories, motivating a falsifiable *state-coverage* experiment.

## Hypothesis and source validity

**Hypothesis:** adding truly novel, independently certified **recoverable off-reference proof states** to training improves theorem completion and decision accuracy at such states more than an equal-compute control using another already-known reference-path state of the same theorem root.

The existing native teacher derives shortest-continuation labels on the exact reference proof trajectory. LSA-02H takes a controlled, legal alternative tactic from that path. It uses the actual Lean 4.19 tactic prover to reach the alternative successor and then computes the shortest finite completion using the Lean-reported AND/OR action graph. The collector retains a correction state only if:

- the alternative tactic is in the public menu and succeeds in real Lean;
- the full *root-to-correction-to-terminal* proof has a finite continuation requiring at most **eight tactic checks**;
- the public state fingerprint is absent from that theorem's certified reference trajectory;
- the model-visible action menu has at least one native shortest-continuation witness;
- the **entire altered theorem proof**, including the deviation, is replayed in Lean, kernel-closed, and compiled separately as an axiom-free Lean declaration.

**Unrecoverable alternatives are recorded as misses.** We never manufacture success labels from the generator's private hints or treat an unprovable dead end as a solvable training example. An off-reference state is a controlled, recoverable deviation from a known shortest proof, not a claim that a particular neural model visited that state. This distinction remains explicit in every receipt.

The source collector can retain **at most two** successful distinct alternative recovery scripts per training root. Repeated public state/menu/budget identities within a theorem are deduplicated before training. Reference-only states are not relabeled as off-reference. Native source hash, independent terminal certificate status, proof script and lineage are frozen into a supplemental dataset.

## Fresh theorem roots

Freeze a **new** Lean-CGRT Horn suite with generator seed `20261201` and 96 independent roots in each of its train, calibration, ordinary-test and OOD-family shards. Compile and freeze a reference-path training teacher from **all 96 training theorem roots**, then collect at most two recoverable legal alternatives per root. The source's precise coverage is a measured outcome.

Reserve the first **32 independent calibration roots** for the development experiment. The model/native proof evaluation runs first; only when all six checkpoints have completed their Lean proof evaluations does a separate job use native Lean to discover correction states on those calibration roots. Its results are **read-only and never used for optimizer updates or model policy selection**. Ordinary-test and OOD proof outcomes remain sealed.

A source corpus with zero independently certified novel states fails the training gate. The calibration audit permits a valid low-coverage negative result and registers the absence of independent off-reference forks explicitly.

## Same-worker training controls

All models share the original Plan B architecture: **57,475 parameters**, width 32, four attention heads, two AB tied recurrent passes, fixed 384-byte input and the **`pp96`** public typed-plus-pretty representation. Training seeds are **157 and 163**, each with 720 CPU FP32 AdamW updates, learning rate 0.001 and gradient clipping 1. Every seed receives the **identical starting parameter tensors and theorem-root/reference-index/switch sampling schedule** across arms.

| Arm | Training state used |
| --- | --- |
| `reference` | Only the existing native shortest-reference trajectory frame |
| `offpath` | On a registered switch draw, a **novel certified correction state** from that same theorem root; otherwise the original frame |
| `sham` | On the identical registered switch draw, another certified **reference-trajectory frame from that same theorem root**; otherwise the original frame |

The sampler picks a theorem root uniformly and then a reference-path index uniformly. A half-probability switch is eligible only when the root has certified off-reference corrections. Switch/supplement indices are selected from a source-only seeded RNG shared across all arms. The sham reproduces the *number and root locations of opportunities* for new state exposure while keeping supervision on the old reference support. For a single-reference-frame theorem, a sham draw necessarily duplicates that frame.

Each of the six cells is fitted **twice consecutively on one physical GitHub-hosted CPU worker**: 12 fits. Torch deterministic algorithms, one CPU thread, pinned CPU Torch 2.5.1, disabled MKLDNN, and environment fingerprints accompany exact tensor, checkpoint-byte and optimizer-trace equality gates. All actual switched/novel/sham draws are counted. Cross-runner bitwise agreement is an outstanding separate issue and cannot be inferred from within-host duplication.

## Native outcome authority

Each trained model evaluates on the same **32 fresh calibration theorem roots** under the real pinned Lean 4.19 `lgym` toolchain, eight tactic attempts, no fine-tuning, and real kernel closure. All closed model proofs undergo independent standalone axiom-free compilation and a second model-free tactic replay. Frozen AB/BA and `auto8` policies are retained as controls, as is the native minimum-proof attainability ceiling.

A registered five-stream **uniform live public-menu** policy on those same roots (seeds 17/31/47/61/83) supplies the independent model-free proof baseline. Its proof scripts are also kernel-checked, independently compiled and replayed; all policy decisions depend only on public state/menu/seed.

**Only after these proof outcomes are frozen**, a separate Lean reference evaluator builds the calibration support corpus, independently certifying all 32 reference proofs and every recoverable alternate script. On these states, each checkpoint is scored by frozen AB argmax against native shortest-continuation labels. It reports:

- **off-reference discriminating fork accuracy**, root weighted (primary state-coverage endpoint);
- **off-reference all-state accuracy**, including forced singleton menus, separately;
- reference-path discriminating fork accuracy, distinct from offpath support;
- complete root-level proof closure and signed paired contrasts to reference/sham.

An eligible **off-reference discriminating fork** has a public menu with at least two tactics, with both optimal and nonoptimal verified continuations. Off-reference forced states must not inflate the primary endpoint.

## Registered decision gates

The primary development screen requires at least **16 independently certified off-reference fork frames spanning at least 8 independent calibration theorem roots**. If that coverage floor fails, the study reports `INSUFFICIENT_OFFPATH_FORK_COVERAGE` and does not claim a fork-action improvement, regardless of any whole-proof changes.

When eligible, success requires the offpath-trained arm's root-weighted off-reference fork accuracy to exceed **both reference and sham by at least +5 absolute percentage points**, with nonnegative mean whole-proof effects against both. This is a development screen; every root-level contrast is accompanied by a **descriptive 2,000-draw theorem-root bootstrap** and the precise denominator. Seeds are paired within roots and do not increase the number of independent theorems.

The separate **native competence floor** remains at least 32 independent roots, **≥40% standalone-verified full-proof closure**, and **≥+10 points over the finite native uniform-menu policy**. Only a passing competence preflight may justify registering fresh sealed confirmation and FSI-01, subject to external corpus-ancestry checks. It never directly authorizes architecture promotion or claims Mathlib transfer.

The experiment does not equate synthetic or deterministic controlled deviations with model-policy on-policy rollout. If recoverable off-reference states are scarce or forced choices dominate them, that is the scientific finding, and the next research mechanism may require explicit backtracking or richer verifier feedback.

## CPU GitHub Actions and provenance

The [LSA-02H CI workflow](../.github/workflows/lsa02h-offpath-cpu.yml) runs public/unit tests, builds the frozen teacher, collects and certifies recoverable train-only branches, trains same-host reference/offpath/sham cells, scores independent native random and neural proof episodes, and **then** collects calibration reference/off-reference states for read-only root-paired scoring.

It records the original native teacher hash, all alternative full-proof compiler certificates, exact lineage and public-state identities, budget adequacy, initial/final weights and source draws, full proof kernel receipts, independent replay receipts and random-menu baseline provenance. Any failed proof or lineage gate aborts the corresponding scientific result. GitHub Actions artifacts retain raw receipts for 14 days and a Markdown report should be committed after successful verification.

No GPU, local private machine, paid inference, or sealed evaluation is needed.
