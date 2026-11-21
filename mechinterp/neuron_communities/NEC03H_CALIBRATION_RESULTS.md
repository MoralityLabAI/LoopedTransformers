# NEC-03H opened-world calibration: 900 steps and source fallback

**2026-11-21. Completed development calibration.** The [GitHub Actions run 44877788211](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44877788211) passed its source-only engineering tests, then trained and independently checked 16 newly initialized 900-step checkpoints on the original, previously opened NEC-03G worlds dev-000 to dev-007. The [complete aggregate artifact](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44877788211) \`nec03h-opened-eight-world-450-vs-900-competence-and-selection\` has ID **14697106570**, ZIP SHA256 **b3483612533191eea3e8431bd7a8d036b86cd0a15c29ff28ec0989e9a26f8127**.

All eight original NEC-03G 450-step checkpoints and their raw-result receipts were restored by immutable Actions run ID \`44870543545\` for audit. The new 900-step models were **trained from their original initial seeds** with the same AdamW/data-index stream, including all optimizer moments over the complete 900 steps. Frozen new checkpoint bytes and source-only community/fallback selection receipts have distinct SHA256 values. Full 900-step runs admitted no new development roots, reserved contexts or confirmation-world models.

## Exact 48-context clean scores

Each model is tested on the same original 768 related histories, grouped as 48 source contexts. Those data were previously opened, so these results measure calibration feasibility only.

| Opened world | Composed 450 | Composed 900 | Composed native site | Additive 450 | Additive 900 | Additive native site |
|---|---:|---:|---|---:|---:|---|
| dev-000 | 642 | 735 | Graph | 768 | 768 | Graph |
| dev-001 | 768 | 768 | Graph | 688 | 768 | Fallback |
| dev-002 | 680 | 768 | Fallback | 768 | 768 | Graph |
| dev-003 | 768 | 768 | Graph | 723 | 768 | Fallback |
| dev-004 | 768 | 768 | Graph | 768 | 768 | Fallback |
| dev-005 | 768 | 768 | Graph | 768 | 768 | None |
| dev-006 | 672 | 665 | Graph | 768 | 768 | Graph |
| dev-007 | 695 | 768 | Graph | 736 | 768 | Graph |

**Clean competence:** 12/16 trained teachers exceeded the fixed 90% threshold at 450 updates and **15/16** at 900. Four of eight whole worlds had both native teachers competent at 450, rising to **7/8** at 900. The one remaining composed model below threshold is dev-006, whose correct predictions declined from 672 to 665.

**Source-site availability:** the source-only fallback supplied **four** otherwise missing candidate sites among the sixteen 900-step models. It used exactly the third and fourth focal chronological positions, two top-frequency non-hub native GELU channels per visit, and only the first eighty *training* context quartets. One additive model, dev-005, still had no eligible four-channel source fallback. For models with a qualifying native multi-visit graph community, the original Louvain source selection was retained.

## Limitations and what is frozen next

The competence gain supports a **fixed 900-step schedule** for a separately locked NEC-03H-v2 development experiment, using untouched dev-008 through dev-015. The new v2 procedure treats the composed teacher's learned head, source candidate and physically sufficient primary TopK-versus-active-alternative pair as root admission gates. Additive and native sham comparisons remain explicit falsification/control outcomes, with missing shams conservatively bounded rather than imputed as zero.

The first NEC-03G study remains **0/8 intention-to-treat** under its original 450-step and all-arm eligibility criteria. This calibration neither modifies its reported failures nor provides any independent replication. The 64 confirmation-world commitments and 48 reserved context quartets for each world remain unopened. Statistical confirmation requires new development root-level sham calibration and the pinned EvalMath FSI-01 power gate.
