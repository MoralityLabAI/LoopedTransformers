# OLR-06H: live frontier counterfactual results

**Status:** native verification PASS; source and fresh headroom gates FAIL. No model fitting.
**Registration:** [06H](OLR_06H_REGISTRATION.md).
**Executed commit:** 9a84196c7ff04321267034e5426d902b52465e6f.
**Workflow:** [45325483896](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45325483896), successful.
**Artifact:** [14889886106](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45325483896/artifacts/14889886106).

All three protocol tests passed. The unmodified demand continuation matched the original path and counter contract. Source: 64 TRAIN and 16 calibration systems, 261 legal component-action counterfactuals and 80 fixed restart trials. Evaluation: 32 new systems, 97 legal counterfactuals and 32 fixed restart trials. No model training.

| Cohort | Best control full | Hindsight full | Full-closure gap | Actionable | Differing full labels |
| --- | ---: | ---: | ---: | ---: | ---: |
| TRAIN | 52/64 | 53/64 | +1.56 pp | 52/64 | 2/64 |
| Calibration | 12/16 | 12/16 | 0 pp | 12/16 | 0/16 |
| Fresh | 26/32 | 26/32 | 0 pp | 23/32 | 0/32 |

At cap512, each of continue, balanced-block, demand-block, contact-block and largest-demand block closes 14/16 DEV and 12/16 longer/load roots. The fixed cache128->demand restart closes 13/16 and 8/16. The source-selected control is balanced-block in DEV and continue in transfer. The source and fresh closure gaps are below the registered 10-point threshold. Both best-control closure rates are 81.25%, above the allowed 80%. Reader fitting is explicitly ineligible.

| Fresh policy | DEV mean work | Transfer mean work | DEV mean utility | Transfer mean utility |
| --- | ---: | ---: | ---: | ---: |
| Continue | 232.75 | 308.00 | 1.11004 | .98148 |
| Balanced block | 230.56 | 315.13 | 1.11008 | .98135 |
| Demand block | 231.19 | 315.69 | 1.11007 | .98133 |
| Contact block | 233.31 | 316.63 | 1.11003 | .98132 |
| Largest demand | 225.81 | 311.13 | 1.11017 | .98142 |
| Hindsight | 213.94 | 294.81 | 1.11040 | .98174 |

Even the hindsight choice saves only 16 mean expansions relative to continue (approximately 5.9% of its mean work), with identical complete closure. Source 554 distinct native leaf replays and 156 conjunction declarations; fresh 236 and 68, respectively. Total 1,014 axiom-free declarations (790 leaves, 224 conjunctions). Multiple distinct proofs for a root are not independent samples.

## Custody

Downloaded archive SHA-256 1280825f3838443bac2700e4823b2ef2a00dcef511c791038b4bcf4702d0f302.
All 29 manifest-listed files independently checked.
- Registry: 3a4f0c2c8ca9b4e7f76b23349fc043e4218743bf2802851e69acc7174083c6fb
- Source: 004f0dca159e0e88b18669e7bdcb9c7b93fd2248615488f0136e2e6349f192f8
- Evaluation: 3f2788bdc64e1a8bf0351dfa247be381eebb5e1cc07988c986bf6931de0a9bf4

## Decision

Close the finite rewrite family for learned allocation. Keep the exact receipt ledger, path reuse, metered pattern abstractions, and source calibration as engineering outputs. Do not change the 06H block size or budget after seeing the result to manufacture headroom. OLR-07 moves to authentic library tactic states, with costly native diagnostic feedback and an explicit empirical adaptive-policy baseline. This is a new family and research question, not a promotion of the old cell.
