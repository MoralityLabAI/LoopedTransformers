# Deuterium: TRM/VPD retrospective and follow-up

This directory is the versioned home of the recovered Deuterium paper, original
June 2026 evidence inventory and reproducible steering follow-up in
`MoralityLabAI/LoopedTransformers/mechinterp`.

- [Manuscript (Markdown)](papers/VPD_TRM_Feature_Steering/vpd_trm_feature_steering_paper.md)
- [Rendered manuscript (PDF)](papers/VPD_TRM_Feature_Steering/deuterium_retrospective.pdf)
- [Source and evidence audit](experiments/deuterium/HISTORICAL_AUDIT.md)
- [Frozen experimental protocol](experiments/deuterium/PROTOCOL.md)
- [Porting, hashes and commands](PORTING.md)
- [Per-seed result and uncertainty ledger](experiments/deuterium/results/followup/scored/steering_summary.json)
- [Independent row and intervention verifier](experiments/deuterium/verify_followup.py)
- [Recovered-record manifest](experiments/deuterium/results/archive_manifest.json)

## Outcome and interpretation

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
Its random-minus-VPD target-loss effect is +0.0008904 nats across seeds,
95% t CI [-0.0001599, +0.0019407]. The frozen verdict is
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

For local replay commands and caveats see [PORTING.md](PORTING.md).
