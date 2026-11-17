# LS-SAE07 source admission: fixed 54-hole cohort blocked

**Executed November 16, 2026; exact source-only receipt.** [LS-SAE07 prospective registration](../native_hard_task_v1/PROTOCOL.md) committed at 2641f05d5cc9a1d6c7359543c44693b151d3c3cf before generating new heldout model states. The study precommitted one original noncanonical full Sudoku solution per grid seed, with exactly 52 or 54 uniquely solvable removed clue cells required. Rejected source grids were explicitly to remain rejected without changing cohort denominator or replacing source seeds.

## Original source admission verdict

| Frozen unique Sudoku source cohort | Requested | Admitted | Scientific test |
|---|---:|---:|---|
| Pilot 52-hole | 2 | 2 | Engineering-only |
| Pilot 54-hole | 2 | 2 | Engineering-only |
| Full 52-hole | 8 | 8 | Unrun |
| **Full 54-hole** | **8** | **7** | **Blocked** |

The fixed 54-hole full-grid seed **202627040006** (dig permutation seed **202628040006**) reaches a maximum of **53** holes while maintaining a *unique* solution under the exact source two-solution oracle and source-preserving 81-cell removal-order algorithm. Requiring one more deletion would introduce multiple valid completions. The original exact-54-hole acceptance condition therefore fails for that registered puzzle. The full cohort remains 8 requested slots and **cannot produce a scientific model result** under the original protocol.

The original [source-only Actions CI #43850235698](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43850235698) passed all three source and Sudoku-propagation tests and independently archived the complete admission record, artifact **14217720117**, including the fixed seed, original failure text and all 19 admitted source fingerprints. The registered code checks original positive alternating rectangle invariant, exact 9×9 Sudoku validity, source unique-solution count and parity of a separate standard-library naked+hidden-single difficulty implementation with the existing LoopScope02 propagation trace.

**No trained neural checkpoint was run for LS-SAE07**, and there is no LS-SAE07 positive or negative scientific gate. The prior LS-SAE06 failure remains independently source-audited.

## Independently prospective continuation

[LS-SAE08 PR #132](https://github.com/MoralityLabAI/LoopedTransformers/pull/132) registered a new deterministic, wholly model-blind first-admissible source rule on **fresh full-grid seed bases** and retained the original LS-SAE07 test geometry, physical doses and source-registered flip gate. It searches up to 32 sequential source grid seeds per puzzle slot and admits the first whose exact unique dig reaches the requested 52 or 54 holes. No pretrained model correctness, hidden states or source feature scores enter that structural admission. LS-SAE08 must report its outcomes under its own separate preregistration and raw receipts.
