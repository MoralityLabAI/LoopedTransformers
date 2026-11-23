# LSA-02H: verified recoverable off-reference Lean states

**Date:** November 23, 2026. **Status:** native source/engineering **PASS**; primary off-reference fork screen **INELIGIBLE (zero discriminating forks)**; native proof competence **BLOCKED_INCOMPETENT_CHECKPOINT**. [PR #329](https://github.com/MoralityLabAI/LoopedTransformers/pull/329) merged at `880080923651c28bda3bf59368ab22c97381897a`. The two complete source/CPU/native campaigns are [run 45211312980](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45211312980) and [run 45211414102](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45211414102).

The experiment implemented [the preregistered LSA-02H protocol](../docs/lsa02h_offpath_native_cpu_protocol.md). Both full campaigns completed green on the implementation branch, including the previously installed PPT Gym, native Lean, and typed-state regression workflows. Neither ordinary-test nor OOD proof outcomes were opened. No paid inference or GPU workers were used.

## Motivation and native proof authority

Earlier [LSA-02G](lsa02g_native_fork_competence_20261123.md) upweighted shortest-reference-path training forks by fourfold, but its independently verified calibration fork-choice accuracy failed to exceed a shuffled source-weight control. The next hypothesis was that the model needed supervised proof decisions **after a legal deviation from a shortest proof**, rather than repeatedly emphasizing already-visited reference states.

LSA-02H froze a new Lean-CGRT Horn theorem suite under generator seed `20261201`, using actual Lean 4.19 `lgym`. It independently compiled **96 training reference proofs** and obtained **524 train-only reference-path frames**, with **83/96** shortest reference scripts fitting within eight tactic attempts. These 524 reference frames include **225 discriminating candidate-choice forks**.

The off-reference generator walked each reference script, considered legal alternative public-menu tactics, and asked Lean to confirm each tactic's successor state. It reconstructed a shortest finite continuation by filtering unprovable decoy edges from the native AND/OR proof graph. This filtering repaired an initial `None`-cost handling issue without changing the frozen proof graph or Lean verifier. The entire resulting *nonshortest root-to-terminal script*, not just its suffix, underwent native Lean terminal kernel verification and independent standalone axiom-free compilation. Only novel post-deviation public states were eligible for training.

### Source corpus

| Frozen training evidence | Count |
| --- | ---: |
| Independent theorem roots | 96 |
| Native-certified reference-path frames | 524 |
| Discriminating reference-path training forks | 225 |
| Accepted legal alternative tactics observed in the native graph | 355 |
| Alternatives with finite continuations | 80 |
| Alternatives whose complete proof fits eight tactic checks | 45 |
| Independently compiled recoverable alternate full proofs | **42** |
| Source roots yielding novel certified correction states | **31** |
| Novel certified off-reference correction states | **93** |
| Discriminating off-reference training forks | **0** |
| Unreachable or over-budget alternative candidates | 310 |

The training-support audit reads the actual frozen artifact `offreference/offreference_frames.jsonl`. Every one of its **93 correction states has exactly one public candidate tactic**. The 225 original-reference fork states provide meaningful action alternatives; the 93 novel recovery states expose goal patterns along longer but still solvable branches, with forced choices.

The controlled deviations are *not model-conditioned rollouts*. They are fixed public-menu alternatives whose complete recovery scripts were verified by Lean before use. The result must not be described as native on-policy self-correction.

**Source provenance:** theorem suite manifest SHA-256 `49261c0d1663f7feb675c23e8f5f75bc922f41a07114ea3be16bd4649d196533`; reference teacher manifest SHA-256 `b9a754d4f684949d6298876726fb81ba70be4456061cc31919f783056c9d1c09`; off-reference supplemental corpus manifest SHA-256 `a81d8d1f76d0116bf6d9e48aa50176137b566b4a7359f01ae2f2325a6798d2e8`. The [native source artifact](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45211414102/artifacts/14838230434) preserves the corpus, independent Lean certificates, candidate counts, hashes, and per-theorem source census.

## Matched, reproducible CPU training

The [off-reference trainer](../research/lean_cgrt_hrm400/ppt_gym/offpath_train.py) used the same Plan B 57,475-parameter architecture, width 32, two AB recurrent passes, four attention heads and `pp96` typed-plus-pretty input under a strict 384-byte ByteEncoder cap. Each arm received **720 FP32 AdamW updates**, using identical seeded initial model weights and the same theorem-root/reference-index/switch sampling stream within each seed.

| Training arm | Supplied frame on a matched switch opportunity |
| --- | --- |
| Reference-only | Already-certified reference-path frame |
| Off-reference | Novel Lean-certified recovery frame from the *same training theorem root* |
| Same-root sham | A different or repeated reference-path frame from the *same training theorem root* |

Model seeds were 157 and 163. For seed 157, off-reference training substituted **111** verified novel-state samples, and the sham substituted **111** reference-state samples at exactly the same opportunities. Seed 163 had **117** matching opportunities. Each of the six cells was fitted twice on a single CPU worker with deterministic Torch algorithms, one Torch thread and MKLDNN disabled. **All 12 within-worker fits passed bitwise tensor, checkpoint-byte and exact optimizer-trace reproduction** in both complete CI campaigns.

**Cross-worker limitation:** despite identical source identities, initial model weights and optimizer sample rosters, repeated training on different GitHub CPU hosts produced different final checkpoints and optimizer losses in **every** cell. Direct byte comparison of the archived optimizer traces showed equal theorem/frame exposure in both campaigns, but **645 to 703 of 720 per-cell loss values differed**. This is actual cross-worker numerical sensitivity, not merely checkpoint serialization. The two campaigns reuse the same 32 calibration roots and are technical repeats rather than independent replication cohorts.

## Native full-proof calibration

Every frozen checkpoint was evaluated on the **same 32 new calibration theorem roots**, permitting eight tactic attempts. All claimed model closures passed real Lean 4.19 kernel verification, independent axiom-free standalone compilation, and model-free action replay. The reference oracle independently proved **26 of the 32 roots attainable** under the public action menu and tactic budget. The public-order `auto8` heuristic closed **3/32**.

The five-stream independent **uniform public-menu** baseline, with fixed seeds 17, 31, 47, 61 and 83, generated 160 model-free native Lean episodes. It achieved **20/160 = 12.50%** certified proof completion. The baseline used no neural calls, and every terminal proof compiled independently.

### First complete campaign, run 45211312980

| Seed | Reference-only | Off-reference | Same-root sham |
| --- | ---: | ---: | ---: |
| 157 | 4/32 | 5/32 | 2/32 |
| 163 | 5/32 | 7/32 | 5/32 |
| Mean certified proof closure | **14.0625%** | **18.7500%** | **10.9375%** |

The theorem-root-paired effect of off-reference training was **+4.6875 percentage points versus reference-only**, with descriptive 95% root-bootstrap interval **[−4.6875, +15.625] points**. Versus the sham it was **+7.8125 points**, interval **[−1.5625, +18.750] points**.

### Second complete campaign, run 45211414102

| Seed | Reference-only | Off-reference | Same-root sham |
| --- | ---: | ---: | ---: |
| 157 | 3/32 | 4/32 | 4/32 |
| 163 | 5/32 | 6/32 | 6/32 |
| Mean certified proof closure | **12.5000%** | **15.6250%** | **15.6250%** |

Here off-reference training gained **+3.125 points versus reference-only**, with descriptive interval **[−7.8125, +14.0625] points**. It **tied the sham**, for a mean difference of 0 and interval **[−10.9375, +9.375] points**. Both campaigns used exactly the same verified roots and model sampling schedule, while cross-worker training numerical variability produced different final checkpoints. Pooling the two campaigns as independent theorem data would be invalid.

## Post-proof calibration support audit

**Only after all six native model proof runs completed**, the [independent calibration auditor](../research/lean_cgrt_hrm400/ppt_gym/offpath_eval.py) reran Lean on the frozen development roots and compiled every shortest reference proof plus all recoverable alternative full proofs. It found:

| Native calibration source | Certified states | Theorem roots | Discriminating tactic forks |
| --- | ---: | ---: | ---: |
| Reference-path eligible forks | **69** | **25** | 69 |
| Recoverable off-reference correction states | **31** | **10** | **0** |

The 31 off-reference states arose from **16 independently compiled recoverable alternative full scripts**. All 31 had singleton candidate menus, each with exactly one shortest-continuation label. Consequently every frozen model chose the correct action on these states, with **100% off-reference all-state accuracy** in both campaigns. This is *forced-choice correctness*, not a learned ability to distinguish alternative recoveries. The preregistered primary decision endpoint was accuracy at **discriminating off-reference forks**, which yielded **zero eligible roots and zero frames**.

The explicit coverage prerequisite required at least **16 discriminating off-reference fork states across eight independent theorem roots**. It failed on both runs, making the registered off-reference-action improvement screen **ineligible**. No statistically defensible claim of targeted recovery-decision learning follows from the positive full-proof point estimates.

Reference-path fork accuracy, reported separately on 25 development roots, changed by **+4.767 points** for off-reference versus reference-only training in the first campaign, with descriptive interval **[−3.733, +13.667] points**. In the repeat the effect was **−0.733 points**, interval **[−12.833, +10.333] points**. Both include zero and fail to establish stable choice improvements.

## Competence and scientific result

The maximum observed mean whole-proof completion was **18.75%** in the first campaign and **15.625%** in the repeat. Both are below the registered **40% native competence floor**. Against the 12.50% independent uniform-menu baseline, the off-reference margins were **+6.25** and **+3.125** points, both below the registered **+10-point** margin.

The final verdict is **`INSUFFICIENT_OFFPATH_FORK_COVERAGE`** for the primary state-coverage question and **`BLOCKED_INCOMPETENT_CHECKPOINT`** for model competence. The 93 certified novel training states establish useful new proof-data infrastructure. The data do not establish that added recoverable states improve fork decisions beyond a same-root sham.

A key structural finding is that this Horn generator's **recoverable departures from the shortest path lead into forced continuations**, while many other legal departures lead into unprovable or over-budget branches. Learning to take the sole action on a recoverable branch is a limited solution to a policy that frequently commits to a wrong branch in the first place.

## Next research mechanism, before a full gym

A prospective CPU preflight should investigate **verifier-grounded branch avoidance and bounded backtracking**, using the real Lean proof-state history. Source-only native labels can distinguish accepted legal transitions that lead to provable shortest continuations from accepted transitions whose subgoals are unreachable within the remaining budget. The evaluator would add an explicit backtrack/reconsider action with a registered tactic-check cost and source-custodied state stack, then compare it against existing deterministic public-order and model-free uniform-menu controls.

Every branch transition, backtrack, accepted script and final theorem proof should remain under native Lean replay and independent standalone compilation. The experiment would need fresh theorem root lineages, matched CPU training/shams, a preregistered cost metric, and an independently verified competence gate. No architecture promotion, Mathlib transfer claim or GPU scale-up is authorized by LSA-02H.

## Reproducibility and artifact links

- [Merged implementation PR #329](https://github.com/MoralityLabAI/LoopedTransformers/pull/329)
- [Native recoverable off-reference data collector](../research/lean_cgrt_hrm400/ppt_gym/offpath_source.py)
- [Same-host matched source/sham trainer](../research/lean_cgrt_hrm400/ppt_gym/offpath_train.py)
- [Native post-scoring calibration coverage auditor](../research/lean_cgrt_hrm400/ppt_gym/offpath_eval.py)
- [Protocol and test gating](../docs/lsa02h_offpath_native_cpu_protocol.md)
- [First complete CI run 45211312980](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45211312980)
- [Second complete CI run 45211414102](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45211414102)
- [Second-run native source artifact](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45211414102/artifacts/14838230434)
- [Second-run matched CPU checkpoint artifact](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45211414102/artifacts/14836179179)
- [Second-run post-scoring calibration and paired-root evidence](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45211414102/artifacts/14837013505)

GitHub Actions artifacts use **14-day retention**. Full root-level manifests, theorem/proof scripts, checkpoint SHA-256 files, model-free replay traces, and compiler receipts remain in those artifacts. This permanent Markdown report preserves key measured results and decision boundaries once artifacts expire.
