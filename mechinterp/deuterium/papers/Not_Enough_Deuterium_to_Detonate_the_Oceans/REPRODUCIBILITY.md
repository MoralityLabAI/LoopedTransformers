# Evidence and reproduction guide

**Companion to:** *Not Enough Deuterium to Detonate the Oceans* (20 November 2026). This guide separates reconstruction of the figures and recorded statistics from genuinely rerunning model training or held-out evaluation. The latter needs the frozen datasets, checkpoints, source dependencies, and original registration gates.

## 1. Verify the bibliography and rebuild the paper

The source bundle contains `Not_Enough_Deuterium_to_Detonate_the_Oceans.md`, `references.bib`, `preamble.tex`, `routing_by_root.pdf`, `routing_root_values.json`, `verify_routing.py`, and `make_figure.py`. From this directory:

```bash
python verify_routing.py
python make_figure.py
pandoc Not_Enough_Deuterium_to_Detonate_the_Oceans.md --citeproc --standalone \
  --pdf-engine=xelatex -H preamble.tex \
  -V mainfont='Noto Serif' -V sansfont='Inter' \
  -V papersize=a4 -V geometry:top=0.89in -V geometry:bottom=0.85in \
  -V geometry:left=0.90in -V geometry:right=0.90in \
  -V fontsize=11pt -V linestretch=1.08 -V colorlinks=true \
  -V linkcolor=linkblue -V urlcolor=linkblue -V citecolor=linkblue \
  -o Not_Enough_Deuterium_to_Detonate_the_Oceans.pdf
```

Every bibliography key used in the manuscript appears in `references.bib`. The author-year references link directly to scholarly publications or named, versioned research resources. Exact citations for the Trinity/ocean analogy are Los Alamos LA-602, Roger E. Batzel's signed *LLL Comments on the Ultimate Catastrophe* (UCRL-ID-125509; 23 December 1975; [OSTI record 443240](https://www.osti.gov/servlets/purl/443240)), and Weaver & Wood (1979), *Physical Review A* 20:316–328.

## 2. Versioned code and empirical evidence

| Evidence group | Exact sources | What can be independently checked |
| --- | --- | --- |
| Historical parameter-component transfers | [Recovered archive at Hugging Face](https://huggingface.co/datasets/AlephFunk/vpd-trm-original-records-202606), [archival audit](https://github.com/MoralityLabAI/LoopedTransformers/blob/39db0ab08a2c7c69b3b0bb04e74b13c7b59bdc7a/mechinterp/deuterium/experiments/deuterium/HISTORICAL_AUDIT.md) | Recount 1,305,171 total evaluation rows, 131,376 `vdp_or_class_row` candidate rows (110,731 stable), sixteen accepted SVD/head-row rows, and guarded random controls |
| Synthetic VPD experiment | [Source-frozen protocol](https://github.com/MoralityLabAI/LoopedTransformers/blob/39db0ab08a2c7c69b3b0bb04e74b13c7b59bdc7a/mechinterp/deuterium/experiments/deuterium/PROTOCOL.md), [machine configuration](https://github.com/MoralityLabAI/LoopedTransformers/blob/39db0ab08a2c7c69b3b0bb04e74b13c7b59bdc7a/mechinterp/deuterium/experiments/deuterium/protocol.json) | Check 32 learned and 32 random directions, both signs, 1% head-weight Frobenius norm, primary endpoint and positive-claim gate |
| Native recurrent-weight comparison | [Native protocol](https://github.com/MoralityLabAI/LoopedTransformers/blob/39db0ab08a2c7c69b3b0bb04e74b13c7b59bdc7a/mechinterp/deuterium/native_trm/PROTOCOL.md), [preserved source receipts](https://github.com/MoralityLabAI/LoopedTransformers/tree/39db0ab08a2c7c69b3b0bb04e74b13c7b59bdc7a/mechinterp/deuterium/papers/VPD_TRM_Feature_Steering/evidence/native01) | Verify 16 rank-one matrices per bank, both signs (32 signed candidates per arm), the 32/64 warmup/main-update fit, dose, 256-string test cohort and three-teacher summary |
| Weight-sharing diagnostic | [Frozen source report](https://github.com/MoralityLabAI/LoopedTransformers/blob/39db0ab08a2c7c69b3b0bb04e74b13c7b59bdc7a/mechinterp/deuterium/papers/VPD_TRM_Feature_Steering/evidence/cache_audit/old_train_source_only.json), [optimizer-status receipt](https://github.com/MoralityLabAI/LoopedTransformers/blob/39db0ab08a2c7c69b3b0bb04e74b13c7b59bdc7a/mechinterp/deuterium/native_trm/adapter02/results/source_schedule_20261119/schedule_receipt.json) | Check final-pass cache reuse on eight old training strings, and the pass-indexed 32/64 schedule comparison on one excluded untrained teacher |
| BPD state steering DT04 | [Frozen development protocol](https://github.com/MoralityLabAI/BPD/blob/8e02d09c1f67e82b6aed61199df6603a766758b9/mechinterp/bpd/experiments/depth_steering_gym/dt04/PROTOCOL_DT04.md), [original result](https://github.com/MoralityLabAI/BPD/blob/c24c676bad2f577cba94960093795c7dead5d218/mechinterp/bpd/experiments/depth_steering_gym/dt04/RESULTS_20261119.md) | Distinguish attempted 59 roots from 56 audit-clean, 16 fully eligible and 19 postselected competent |
| BPD DT05 transported basis | [Transport and EvalMath receipt](https://github.com/MoralityLabAI/BPD/blob/c24c676bad2f577cba94960093795c7dead5d218/mechinterp/bpd/experiments/depth_steering_gym/dt05/benchmarks/07_dynamic_transport_and_evalmath.json) | Recalculate the eight-root mean and inspect sham resolution and headroom |
| BPD DT05 online routing | [Eight-root JSON](https://github.com/MoralityLabAI/BPD/blob/c24c676bad2f577cba94960093795c7dead5d218/mechinterp/bpd/experiments/depth_steering_gym/dt05/benchmarks/08_autonomous_routing_study.json), [corrected diagnostic and erratum](https://github.com/MoralityLabAI/BPD/pull/136) | Recompute all-root and competent-subset exact boards, inspect invalid archived mask-overlap field |
| Sudoku9 first-valid stopping | [Depth-by-depth results and checkpoint hashes](https://github.com/MoralityLabAI/BPD/blob/c24c676bad2f577cba94960093795c7dead5d218/mechinterp/bpd/experiments/depth_steering_gym/dt05/benchmarks/09_first_valid_stopping_7m_and_dt04.json), [64-board source](https://github.com/MoralityLabAI/BPD/blob/c24c676bad2f577cba94960093795c7dead5d218/mechinterp/bpd/experiments/depth_steering_gym/dt05/benchmarks/raw_receipts/bench_9x9_64cases.json) | Check board validity, depth-16 results, stopping policy and reported BF16 batch-composition drift |

**VPD source identity.** The actual experiment checkout is the [`MoralityLabAI/param-decomp` fork at `2df0092d5be4d8e4748b3c66ef8325dedef17549`](https://github.com/MoralityLabAI/param-decomp/tree/2df0092d5be4d8e4748b3c66ef8325dedef17549), on `rust-accel-scoring`. The files [`param_decomp/run_param_decomp.py`](https://github.com/goodfire-ai/param-decomp/blob/1f531b3/param_decomp/run_param_decomp.py) and [`param_decomp/models/component_model.py`](https://github.com/goodfire-ai/param-decomp/blob/1f531b3/param_decomp/models/component_model.py) have the same Git blob SHA as Goodfire upstream commit [`1f531b3`](https://github.com/goodfire-ai/param-decomp/tree/1f531b3), dated 6 May 2026. SHA-1 for the first file: `d49882b081ababa263a5884c7f154996d17a2456`; for the second: `1becd0ba0e110f53a9c88462b44dcba8bf1a6d72`. Thus the **optimizer and component-model implementation are inherited without modification**, while the fork commit adds experiment wrappers and its toy model. The final-pass cache diagnostic is specifically about that upstream component-model code **as of 6 May 2026** when used on a four-pass weight-tied module. The Deuterium [PORTING.md](https://github.com/MoralityLabAI/LoopedTransformers/blob/39db0ab08a2c7c69b3b0bb04e74b13c7b59bdc7a/mechinterp/deuterium/PORTING.md) pins the experiment checkout. The earlier `head_svd`/`head_row` component-transfer search is a **separate method**, without an adversarial VPD fit.

**Recovered archive digest.** `a45db5ef5e603852efb4f247112655c5572eff5aa85f9ab94674a4ae9dc8f003` is the SHA-256 of the recovered historical ZIP, as recorded in the audit. This digest identifies the complete archive, not a new paper artifact.

## 3. Independent saved-evidence audit

With the matching `LoopedTransformers` repository checkout and the PDF requirements installed:

```bash
python -m pip install -r mechinterp/deuterium/experiments/deuterium/requirements-paper.txt
python -m mechinterp.deuterium.paper_closeout.render --verify-only
```

The existing repository checks 28 evidence receipts and independently derives the DT05 per-root aggregate arithmetic. A rebuilt PDF from its *original*, longer manuscript is not identical to this journal-style rewrite, which has its own source bundle and bibliography.

Synthetic-model reproduction requires the original unmodified source, locally extracted verified historical archive, and pinned dependency environment. See the run contract and exact invocation in [PORTING.md](https://github.com/MoralityLabAI/LoopedTransformers/blob/39db0ab08a2c7c69b3b0bb04e74b13c7b59bdc7a/mechinterp/deuterium/PORTING.md). The hosted replay repeats previously opened seeds; it is engineering replication, not a fresh prospective test. Native-model source and checkpoint provenance require their specific protocol and SHA checks. Do not reuse reserved cohorts to produce evidence for this revised paper.

## 4. Figure provenance

The paper figure is generated from `routing_root_values.json`, a readable transcription of the eight `units` in the DT05 JSON at immutable BPD commit `c24c676...`. The corresponding fractions come from each `arms[*].by_depth["8"].exact_board_accuracy` and the `dense_teacher.by_depth["8"].exact_board_accuracy` field. An additional `raw_depth8_exact_board` field differs on five roots, including root 2605904 (33/64 raw, 22/64 dense); the raw field's derivation is not established in the receipt. Using raw scores would change the competence-selected subset from three to four models. `python verify_routing.py` checks the transcription internally; `python verify_routing.py --source-json PATH` checks every root against the downloaded pinned source JSON. Its data are 64 cases per root. The 3-root competent subset contains dense 171/192, online 113/192, fixed 8/192, and teacher-informed 157/192 solved boards. The 5-root complement contains dense teacher 68/320, online 27/320, and fixed 0/320 successful boards; the competent group contains all eight fixed-route successful boards. The older reported top-k overlap is *ordered rank-position match fraction divided by 12* and cannot be reconstructed as actual unordered overlap without a fresh pilot replay.

The paper does not claim a one-command replication of training from scratch. Its recorded claims can be audited from the linked immutable source and JSON receipts; a genuine independent experiment must run the defined preregistration gates with fresh roots and protected inputs.

## 5. Supplemental scope and provenance notes

The exploratory arena's accepted components were nine `head_svd` and seven `head_row` directions; see `HISTORICAL_AUDIT.md`. None of those 16 acceptances constitutes a fitted VPD-component result. ARC Challenge's four held-out cards contained only **5, 4, 2 and 6** genuinely unseen rows; the selected rule changed correctness on **none** of those unseen rows.

The DT05 transport receipt reports `git_dirty: true` at head `8ccdcb2` and carries a per-file SHA-256 inventory. The eight-root `-0.0078125` effect comprises one +1/16 result, two -1/16 results, and five zero results. Its EvalMath outcome is a **planning preflight**, with source-sealed confirmatory roots unused.

The third-party model aliases point to [`yagizdevre/trm-sudoku-extreme-mlp`](https://huggingface.co/yagizdevre/trm-sudoku-extreme-mlp) and [`gaoxin492/TinyRecursiveModels-Sudoku-Extreme-mlp`](https://huggingface.co/gaoxin492/TinyRecursiveModels-Sudoku-Extreme-mlp). BPD's audit describes both as *caller supplied, no download*. The local model SHA-256 values in the receipt identify the evaluated checkpoint bytes. No exact Hugging Face repository revision is recorded for those local bytes.

## 6. November 20 review follow-up

The 19-root DT04 competence-selected group includes all three float32 gauge-audit outliers, roots `2604008`, `2604026` and `2604050`. The three-root DT05 routing cohort uses the explicitly named `dense_teacher.by_depth["8"]` field; the receipt's conflicting raw field has no established generation contract, and the competent-subset counts must not be silently redefined. The hosted VPD gate run is a replay of a preceding local score (local mean +0.00089040; hosted +0.00088999). The native teacher-211 paired-string interval is [−0.000268, −0.000022], on one fitted teacher.

Bibliography verification: Goodfire's May 5, 2026 VPD publication is at `goodfire.com` (the `goodfire.ai` URL redirects), with arXiv version `2610.20748` published later. The 1975 seawater calculation comes from Roger E. Batzel, *LLL Comments on the Ultimate Catastrophe*, UCRL-ID-125509, OSTI 443240, and states the factor of approximately 20,000. Tom Angsten's October 20, 2026 audit is publicly available on LessWrong. The PMLR record for Kaissis et al. confirms volume 306, pp. 55202–55235.
