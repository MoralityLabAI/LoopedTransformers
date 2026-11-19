# Deuterium: final paper and reproducible evidence

This directory is the versioned home of the recovered Deuterium paper, original
June 2026 evidence inventory and reproducible steering follow-up in
`MoralityLabAI/LoopedTransformers/mechinterp`.

- [Completed manuscript (Markdown, 19 November 2026)](papers/VPD_TRM_Feature_Steering/deuterium_complete_20261119.md)
- [BPD saved-data arithmetic audit](papers/VPD_TRM_Feature_Steering/cross_study_summary.json)
- [Completed manuscript (PDF)](papers/VPD_TRM_Feature_Steering/deuterium_complete_20261119.pdf)
- [Final scope and claim ledger](papers/VPD_TRM_Feature_Steering/PAPER_STATUS.md)
- [Byte-preserved hosted and native evidence](papers/VPD_TRM_Feature_Steering/evidence/)
- [Earlier retrospective manuscript](papers/VPD_TRM_Feature_Steering/vpd_trm_feature_steering_paper.md) and [PDF](papers/VPD_TRM_Feature_Steering/deuterium_retrospective.pdf)
- [Source and evidence audit](experiments/deuterium/HISTORICAL_AUDIT.md)
- [Frozen experimental protocol](experiments/deuterium/PROTOCOL.md)
- [Porting, hashes and commands](PORTING.md)
- [Per-seed result and uncertainty ledger](experiments/deuterium/results/followup/scored/steering_summary.json)
- [Independent row and intervention verifier](experiments/deuterium/verify_followup.py)
- [Recovered-record manifest](experiments/deuterium/results/archive_manifest.json)

## Outcome and interpretation

The final report includes the hosted synthetic replay, the first native
recurrent-weight experiment and related BPD Sudoku engineering observations. The native mean random-minus-VPD goal NLL is
-0.0000260810 nats, with a descriptive three-teacher 95% t interval of
[-0.0002873765, +0.0002352145]. Its recorded verdict is
`DEVELOPMENT_NO_REGISTERED_ADVANTAGE`. The fitted-checkpoint diagnostic documents
the stock adapter's reuse of a final-pass importance mask across four calls.
The prospective pass-indexed comparison remains unscored: a full source-only
optimizer trajectory exceeded its fixed numerical tolerance. New teacher roots
and their reserved sources remain unopened.

Section 9 integrates the DT04 and DT05 records. Across eight excluded DT05
engineering roots, online routing of 12 of 24 shared factors scores 27.34375%
exact boards, compared with 1.5625% for fixed routing and 38.4765625% for the
privileged teacher schedule. The invalid mask-overlap diagnostic and the
selected-subset limits remain explicit. First-valid stopping and numerical
fidelity are reported on their exposed checkpoint and training cohorts.

The earlier 22-page edition and companion note retain their exact bytes in
[preserved prior-edition files](papers/VPD_TRM_Feature_Steering/archive/closeout_before_integration_20261119/provenance.json).
The current paper contains the completed findings and a short future-work
discussion. Reserved model and input cohorts remain unopened.

The recovered archive matches its SHA-256
`a45db5ef5e603852efb4f247112655c5572eff5aa85f9ab94674a4ae9dc8f003`.
A six-seed replication contains 16 accepted grafts; the twelve-entry preview
truncated the list. All sixteen match the guarded random comparator on the
recorded task metrics and are accepted on a logit-movement tiebreak.
The earlier full-arena summary refers to a distinct missing sweep.

Historical ARC samples overlap substantially across train/evaluation cards.
The Wordle cached rule gains 18 correct decisions over 72 distinct observations,
while governance cleanup causes severe collateral failure.

The recovered checkpoint's frozen weight tensors agree 91/91, but original
forward/tokenizer code and reference logits are missing: **task replay is
blocked**.

The separately identified 1,763-parameter synthetic recurrent-gate experiment
used independent seeds 103, 211 and 307 with dose- and rank-matched controls.
The hosted random-minus-VPD target-loss effect is +0.000889987 nats across seeds,
95% t CI [-0.000160050, +0.001940024]. The first local values are retained
separately in Section 6.3. The frozen verdict is
**NO_REGISTERED_ADVANTAGE**. A task-gradient baseline has lower target loss in
all three seeds. See the per-seed records for uncertainty and collateral effects.

The reserved seeds are already public and scored. A new GitHub Actions run is
**engineering reproduction** of the prior result, rather than a new prospective
scientific experiment. Historical model artifacts remain on Hugging Face with
explicit hash verification; only derived records, small model receipts and paper
outputs are committed here.

## Entry points

The scientific programs are copied unchanged from the original review package
under `experiments/deuterium/`; this folder also contains its historical
`.github/workflows` copies. The active GitHub Actions workflows are at the
repository root and are named `deuterium-mechinterp-recovery.yml` and
`deuterium-mechinterp-followup.yml`. Relocated scoring runs through
`experiments.deuterium.relocated_runner`, which verifies original source hashes
and the pinned `param-decomp` implementation before opening held-out examples.
The relocated independent verifier uses the same hash gate to service two
historical Git lookups, leaving the saved-row arithmetic code unchanged.

For local replay commands and caveats see [PORTING.md](PORTING.md).

Build the completed PDF from the saved receipts at repository root:

```bash
python -m pip install -r mechinterp/deuterium/experiments/deuterium/requirements-paper.txt
python -m mechinterp.deuterium.paper_closeout.render
python -m mechinterp.deuterium.paper_closeout.render --verify-only
```

This entry point preserves the original historical manuscript, renderer, tables
and scored protocols. It checks 28 archived source files and the original
optimizer traces, recomputes DT05 per-root aggregates, and verifies the final
render manifest. Rendering reads recorded data without model execution or
evaluation-source generation.

The independently checked hosted replay, negative scientific verdict and native-model continuation gates are documented in [HOSTED_REPLAY_AUDIT_20261119.md](HOSTED_REPLAY_AUDIT_20261119.md).
