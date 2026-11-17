# AF-29: Perfect matched-pair STOP ranking without any neural proof completion

**Registered scientific verdict: PAIRED_STOP_GENERALIZATION_FAIL (5/8 source-qualified checkpoints).**  
**Source-locked engineering and independent audit: PASS (19/19 jobs).**  
**Official successful scored source:** `8325f9a0f13c8fffdb05f2ae25cb727aa119c2d1`, November 17, 2026.

## Abstract

AF-29 asks whether a frozen 6,036-parameter recurrent Horn network and a small train-only fitted STOP gate can discriminate reachable and unreachable graph pairs whose root-centered *local* structure is identical. Each of 96 positive/negative pairs at two held-out depths differs by swapping two remote Horn-rule premises. The original root-known facts, host-legal first actions, marked node positions, per-node directed degree sequence and **directed, identity-preserving radius-two incidence neighborhoods** of seven anchors are fixed within each pair. Exact independent physical Horn proof search certifies the opposite global entailment labels.

Across eight unchanged AF-23 checkpoints, the prospectively trained six-feature CAL STOP gate ranks the unreachable member above the reachable member in **every one of 96/96 paired roots at both depths, in every seed**: paired AUC **1.00**, compared with the mathematically certified **0.50** limit for any deterministic classifier restricted to the registered radius-two local signature. However, original Direct, native VI-2 STOP with Direct rules, CAL, matched one-feature THR and forced continuation **all complete 0/768 positive proofs at both depths**. Public Horn closure and BFS16 each solve **768/768** and correctly STOP on every negative case.

The registered scientific criterion includes calibrated correct immediate STOP on at least **60/96 negative roots at both held depths in six of eight checkpoints**, alongside paired ordering and a source-relative positive-completion safeguard. CAL reaches that termination floor in **five of eight** seeds. It makes 768/768 correct negative STOP decisions at depth five but only **528/768** at depth six, losing 240 correct negative STOPs compared with the original Direct gate at depth six. The complete source-verified outcome is therefore **PAIRED_STOP_GENERALIZATION_FAIL**.

The experiment reveals a useful descriptive distinction between *within-pair discriminability*, *cross-pair threshold choice* and *closed-loop proof search*. The first property is present in the frozen-model-derived scores; the other two prevent a successful stopping-and-solving policy. This benchmark's **positive-proof safeguard is vacuous** because DD itself solves zero problems, a preregistration limitation that must be retained rather than retrospectively repaired.

## 1. Preregistration and immutable evidence

- **Protocol:** [AF-29 PROTOCOL.json](../../experiments/activation_funnel_af29/PROTOCOL.json) first committed [`65d0bb5`](https://github.com/MoralityLabAI/LoopedTransformers/commit/65d0bb59b5299b28fb0dece9c94c5ea892135965) *before* source graph generation or model scoring. Protocol SHA-256 `26c1db8b1d3f88deb79af69cf338e9a36a7c02261400953d9f35f601e4e2744f`. A dated **pre-model** structural repair removed a radius-two marked-subgoal leak from four-firing training/dev templates without weakening the matching gate.
- **Source-only structural preflight:** [CI #44016241887](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44016241887), PASS before any new model fitting. It certified both independent BFS reachability oracles, 384 matched pairs across all four data splits, physical degree/root-role parity and exact directed radius-two local incidence equality. Structural manifest SHA-256 `741e0a4199c9fb86abfa78bb7878d49b85c21813bea527eda713565a3a24edd1`.
- **Frozen original checkpoints:** [AF-23 Track B CI #43812708916](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43812708916), all eight original fixed-final 6,036-weight TypedLoop checkpoints, with per-seed original SHA-pinned ZIP/member evidence, Torch/NumPy parity and original training receipts. **Zero original-network optimization updates in AF-29**.
- **Official scored and independently audited experiment:** [GitHub Actions #44018612389](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44018612389), 19/19 jobs SUCCESS, immutable source-lock commit `8325f9a0f13c8fffdb05f2ae25cb727aa119c2d1`. Source lock, actual checkpoint smoke, eight independent train/dev logistic fits, eight held-out source-proof/model audits and eight-seed aggregate all succeeded. This is the official **strict `1e-4` logistic-refit numeric replay guard** run.
- **Official aggregate:** `af29-calibrated-stop-aggregate`, GitHub Actions artifact ID **14300630612**, ZIP SHA-256 `87611ac394b5bfe48473293fb0d917d85047fa3ecb87b95b3714c7d3862fe0f4`; member `af29_results.json` SHA-256 `2536a15ebcdccc4f795de855381269d2b2738edcd3d7306b7ca96fc853326f6d`.
- **Per-seed artifact references:** [AF29_EVIDENCE_MANIFEST.json](AF29_EVIDENCE_MANIFEST.json) pins both the source-paired train/dev calibrators and independently audited held-out artifacts for every seed, plus the original model identities, source-locked run SHA and integrity digests. CI artifact bytes remain subject to GitHub retention.

Source-only preflight and attempted scored-run failures remain visible. All original AF-22 through AF-28 scientific verdicts remain unchanged.

## 2. Matched global reachability challenge

There are four distinct source splits:

| Split | Matched pairs | Cases | Positive | Negative | Positive shortest proof |
|---|---:|---:|---:|---:|---|
| Train | 96 | 192 | 96 | 96 | 4 or 5 rule firings |
| Dev | 96 | 192 | 96 | 96 | 4 or 5 rule firings |
| Held depth 5 | **96** | **192** | **96** | **96** | Exactly 5 firings |
| Held depth 6 | **96** | **192** | **96** | **96** | Exactly 6 firings |

Two source-registered grammar families contribute 48 pairs apiece to each split. Every case has the identical original root-known logical facts `{0,1,2,3}`, and within a pair the same physical atom/rule permutations, legal initial rule slots, marked roles (`6,13,14`), directed in/out degrees and identity-preserving radius-two rule-premise/head incidence around all seven known/role anchors.

Two middle premise edges are exchanged between the derivable atom 5 and nonderivable atom 10 while rule heads, initial facts and all other premises remain fixed. In the **positive** member, the productive Horn chain consumes 5, permitting a proof of the registered depth. In the **negative** member, it consumes the unreachable 10, blocking every goal derivation. The alternative goal path remains blocked. Two separate host/BFS methods independently validate each label and proof length. The full public Horn graph remains visible to every model and non-oracular symbolic controller.

The model-visible `GraphBatch.targets` tensor is forcibly zeroed **for both pair members**, so private entailment labels cannot enter the original network through a legacy target field. Only registered TRAIN/DEV labels enter the small logistic fitting code. HELD truth is confined to scoring/audit and the proof budget.

### Exact local-observer null

The source-registered local view is the **seven-anchor, directed, physical-ID-preserving induced radius-two incidence graph**. It is identical between pair members, including initial-known/role markers. Any deterministic classifier restricted to that view receives the same input for both labels. On a sample containing both members of every pair it gets **50%** classification accuracy, and an identical scalar score receives **paired AUC 0.50**.

This is an exact local information ceiling, proved in [AF29_LOCAL_BLINDNESS_THEOREM.md](AF29_LOCAL_BLINDNESS_THEOREM.md), not a lower bound on any model that sees the entire public graph. The theorem also shows mathematically why **perfect within-pair AUC need not admit a single useful global STOP threshold** across different pairs.

## 3. Frozen model, fitted gate and registered criterion

CAL uses the original 6,036-weight recurrent checkpoint *without changing it*. Six source-only features describe the Direct native STOP margin, frozen reach/distance logits, VI-2 native STOP margin, public legal-rule count and known-fact count. A seven-parameter train-only logistic (`6` slopes + bias), fitted with **600 deterministic full-batch class-balanced BCE updates and L2 penalty**, converts those features into a STOP probability.

The one-feature THR logistic fits the original Direct STOP margin alone under identical train samples, optimizer budget and DEV threshold grid. Both select the threshold on distinct **96-positive/96-negative DEV roots** with an ≥87/96 provable continuation condition, then freeze it. Every neural policy retains frozen original **Direct legal-rule ranking**, with STOP determined by one of original Direct (DD), VI-2 (VD), fitted CAL, fitted THR, or forced continuation (ND). Original public closure, BFS16, BFS64 and exact shortest-proof oracle are scored as nonlearned controls.

### Registered primary scientific gate

For **one source checkpoint** at **both held depths**, CAL must:

- Correctly rank unreachable above reachable STOP probability on **at least 65 of 96 matched pairs**, awarding an exact tie half credit.
- Correctly initially STOP on **at least 60 of 96 unreachable roots**.
- Finish **at least DD's provable completion count minus five** among the 96 reachable roots, with zero false accepted Horn proofs.

At least **6 of 8 fixed checkpoint seeds** must satisfy every requirement at both held depths for `PAIRED_STOP_GENERALIZATION_PASS`. Source or audit failure yields `INCONCLUSIVE`; a complete source-verified study failing that bar yields `PAIRED_STOP_GENERALIZATION_FAIL`.

The **6/8 gate fails because only 5/8 qualify**. The calibrated score clears the paired-ordering criterion in every seed and depth. However, its absolute STOP floor fails at depth six for seeds 29, 43 and 73.

## 4. Full held-out proof and STOP results

The per-depth denominators are **96 unique positive and 96 unique negative graph cases × eight original neural seeds = 768** of each truth label. Symbolic methods repeat the exact same graphs, rather than discovering 768 distinct tasks.

| Policy | D5 completed proofs /768 | D5 correct immediate negative STOP /768 | D6 completed proofs /768 | D6 correct immediate negative STOP /768 |
|---|---:|---:|---:|---:|
| Original Direct, DD | **0** | 720 (93.75%) | **0** | **768 (100%)** |
| VI-2 STOP + Direct rules, VD | **0** | **768 (100%)** | **0** | **768 (100%)** |
| **Train-fitted six-feature CAL** | **0** | **768 (100%)** | **0** | **528 (68.75%)** |
| One-feature fitted THR | **0** | 624 (81.25%) | **0** | 624 (81.25%) |
| Never-STOP Direct, ND | **0** | 0 | **0** | 0 |
| Public Horn closure | **768 (100%)** | 768 (100%) | **768 (100%)** | 768 (100%) |
| BFS16/BFS64 | **768 (100%)** | 768 (100%) | **768 (100%)** | 768 (100%) |
| Exact shortest oracle | 768 | 768 | 768 | 768 |

CAL exceeds DD by **48** correct negative STOP decisions at depth five and trails DD by **240** at depth six. These are within-pair source-matched graph observations under eight fixed original model initializations, not independent random grammar samples.

The **zero proof-completion row across DD, VD, CAL, THR and even ND** is a central result. CAL only initially STOPs incorrectly on 48/768 reachable depth-five roots and 96/768 reachable depth-six roots. For the remaining reachable roots, the model can choose continuation and still never complete the required Horn derivation within the original shortest-plus-one firing budget. Since forcing the original Direct-ranked policy to continue cannot finish a single proof, this is fundamentally a **frozen legal-rule ranking/trajectory generalization failure** alongside the threshold-calibration failure.

The preregistered positive safeguard (`CAL completions ≥ DD completions − 5`) is **vacuous here because both counts are zero**. A future benchmark should include an absolute held-out proof-completion floor. The original preregistration is preserved, including that defect.

### Seed-level registered outcome

All completed proofs are zero for every seed and arm in this table. Correct negative initial STOP is out of 96, and CAL paired AUC is exactly 1.00 at **both** depths for every seed.

| Seed | D5 CAL negative STOP /96 | D6 CAL negative STOP /96 | CAL pair AUC D5 / D6 | Full registered two-depth gate |
|---:|---:|---:|---|---|
| 17 | 96 | 96 | 1.00 / 1.00 | PASS |
| 29 | 96 | 48 | 1.00 / 1.00 | FAIL |
| 43 | 96 | 0 | 1.00 / 1.00 | FAIL |
| 61 | 96 | 96 | 1.00 / 1.00 | PASS |
| 73 | 96 | 0 | 1.00 / 1.00 | FAIL |
| 89 | 96 | 96 | 1.00 / 1.00 | PASS |
| 103 | 96 | 96 | 1.00 / 1.00 | PASS |
| 131 | 96 | 96 | 1.00 / 1.00 | PASS |
| **Total** | **768/768** | **528/768** | **1.00 / 1.00** | **5/8 FAIL** |

The three failing seeds have a threshold that classifies some or all depth-six unreachable members as *continue* even though their STOP scores rank each pair correctly. This is the distinction between **within-pair discrimination and threshold transfer across pairs**.

### Matched-pair score discrimination against source controls

A native Direct/VI-2 STOP margin is compared **only to the margin from the same source policy** across two paired graphs. Raw Direct logits and learned VI-2 negative values are different scales and cannot be directly combined.

| STOP score source | Mean paired AUC, depth 5 | Mean paired AUC, depth 6 |
|---|---:|---:|
| **CAL** | **1.000** | **1.000** |
| THR, one-feature fitted margin | **1.000** | 0.938 |
| Original Direct native STOP margin | **1.000** | 0.938 |
| Original VI-2 native STOP margin | 0.875 | 0.813 |
| Registered radius-two-only observer | **0.500** | **0.500** |

The **original Direct STOP margin also has perfect depth-five pairwise ranking**, before any added logistic calibration. The one-feature THR produces identical depth-five paired ordering. Thus the AUC=1.00 observation does **not** isolate information newly learned by the seven-parameter CAL head, nor does it prove the recurrent model has generally solved global Horn reachability. These fixed template pairs permit detectable structural correlations, and the full public graph is directly available.

## 5. Probability calibration and native compute

### Held-out root probability metrics

Brier, log loss and expected calibration error (ECE, ten probability bins) are measured on the initial 192 held-out roots × eight checkpoints per depth.

| Metric, lower is better | Depth 5 CAL | Depth 5 THR | Depth 6 CAL | Depth 6 THR |
|---|---:|---:|---:|---:|
| Brier score | **0.07461** | 0.11245 | **0.11128** | 0.12035 |
| Binary log loss | **0.31219** | 0.39859 | **0.42066** | 0.43203 |
| 10-bin ECE | **0.12863** | 0.13873 | **0.14738** | 0.19337 |

CAL scores better than THR by these probability measures at both depths. The selected threshold still fails to produce adequate correct depth-six negative STOP in three checkpoint seeds. It has learned a useful within-template probabilistic ordering but not consistently useful complete-proof behavior.

### Native algorithm work, not matched FLOPs

| Work unit | Held depth 5 | Held depth 6 |
|---|---:|---:|
| DD original-model node evaluations | 16,416 | 16,320 |
| VD original-model node evaluations | 17,232 | 16,080 |
| CAL original-model node evaluations | 28,272 | 31,008 |
| THR original-model node evaluations | 24,528 | 24,816 |
| Forced continuation ND model-node evaluations | 35,808 | 38,112 |
| Public closure rule examinations | 198,880 | 264,192 |
| BFS16 public state expansions | 52,240 | 65,056 |

These units are specific to each algorithm. Equal host-state expansions do not imply equal runtime, memory or numerical FLOPs. Original source closure is exact and takes zero learned weights. Therefore this benchmark supplies no neural efficiency or proof-search capability advantage over public symbolic derivation.

## 6. Independent verification and honest engineering history

The successful [official source-locked run #44018612389](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44018612389) passed **all 19 jobs**: original source lock, genuine frozen model/calibration smoke, eight independently fitted and development-selected STOP gates, eight full held-out physical-proof/model/fit auditors, and an eight-seed aggregate bound to their original SHA-256 evidence.

| Verified evidence | Count |
|---|---:|
| Source-bound original frozen checkpoint seeds | **8** |
| Fully independently audited held-out proof episodes | **27,648** |
| Reconstructed original frozen-model node evaluations | **248,592** |
| Sampled actual frozen-checkpoint Torch forward replays | **160** |
| Independently replayed source TRAIN/DEV feature rows | **6,144** |
| Max train/dev original-feature absolute numerical difference | **6.676×10⁻⁶** |
| Max independently refitted logistic coefficient absolute difference | **7.564×10⁻⁶** |
| TRAIN/DEV source rows with difference above `1e-5` | **0** |
| DEV selected threshold changes | **0** |
| DEV STOP-decision changes | **0** |
| False accepted Horn proofs | **0** |
| Original recurrent network optimizer updates | **0** |

Every physical Horn rule firing was independently replayed from encoded model-visible incidence, with separate host bitset BFS truth and shortest cost validation, entire original node-ledger Direct/VI-2 score checks, calibrated probability/STOP decision reconstruction and sampled Torch forwards. The original saved train/dev feature rows, deterministic logistic optimizer and DEV-selected thresholds were independently reproduced. The **successful official run uses the inherited stricter numeric source-parameter audit bound of `1e-4`**.

### Historical source / CI amendments

- [Initial AF-29 source preflight #44016097245](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44016097245) exposed a true radius-two local information leak in the four-step training/development templates. Before model scoring, the productive intermediate was relocated to an unmarked atom and exact graph-pair local equality independently rechecked. All registered labels and thresholds were preserved.
- [First model smoke #44016807921](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44016807921) failed because an auditor local numeric margin variable shadowed the original checkpoint-loader module. The change was restricted to the independent auditor.
- [Early full attempt #44017031507](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44017031507) completed source training, then failed one seed's logistic independent numeric refit within the inherited `1e-4` bound. An exact-source retry failed identically. The separately registered [train/dev-only diagnostic #44017861619](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44017861619) reproduced the seed-89 features and logistic weights exactly with source-original deterministic Torch two-thread settings, with no heldout data access. The auditor adopted that exact Torch source runtime before loading the checkpoint.
- [An earlier rerun #44018136730](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44018136730) still encountered a source-feature replay deviation, and a static pre-aggregate review identified two copied AF-28 source-family references in the AF-29 aggregate. These were corrected to the source-registered `FAMILIES` registry **without touching the original graph generator, train/dev fit, model weights, actions, heldout scores or scientific thresholds**.
- The successful `#44018612389` source commit includes this registry correction and the **strict independent replay bound**. **Subsequently**, an engineering-only source branch amendment raised the acceptable independent logistic coefficient refit tolerance to `2e-4` after studying worker-to-worker FP32 variation. This later branch revision has a different source hash and is **not the scored source for this official report**. The earlier `1e-4` successful run is retained as the primary, stronger scientific evidence. No later engineering-only run changes this preregistered result.

The original branch remains draft and unmerged. The report is **post-scoring documentation** committed after the evidence was generated and does not retrospectively re-register any scientific criterion.

## 7. Interpretation and AF-30 recommendation

AF-29 establishes an exact locally-blind symbolic control construction and provides a striking empirical pattern: original frozen-network-derived STOP scores can identify the globally unreachable member of every matched pair at both held depths, while **none of the original neural rule-ranking policies can finish even one reachable proof** on the new family, including when forced never to stop. The original Direct margin's depth-five AUC=1.00 further limits any attribution of that discrimination to newly trained logistic coefficients.

A follow-up experiment should **separate global reachability discrimination from legal-action planning** with a source-frozen goal-derivation controller. One useful AF-30 design would compare a fixed exact-closure action policy with original/calibrated STOP gates on independently new matched-pair grammar families, while measuring actual rule examinations, source recurrent forward work and classification errors. An exact closure controller already determines reachability on these tiny Horn graphs, so any neural contribution must demonstrate a measurable compute or generalization advantage over that baseline.

AF-30 should also add an **absolute positive-proof completion floor** to its registered gate, ensuring that a zero-success Direct baseline cannot make a proof-retention criterion vacuous. New HELD source structural pairs should feature varied swap locations and root-topology motifs, and all target/fit split rules should be frozen prior to any outcome measurement.

**AF-29 official status:** PAIRED_STOP_GENERALIZATION_FAIL, 5/8; 19/19 engineering jobs PASS. Original AF-22 through AF-28 verdicts are unchanged. [AF-29 draft PR #137](https://github.com/MoralityLabAI/LoopedTransformers/pull/137) remains unmerged for independent scientific review.
