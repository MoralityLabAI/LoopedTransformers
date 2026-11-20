# NAN-GYM-05 streaming development report

Updated 2026-11-20T05:32:35.273497+00:00

Frozen source-only registration; one checkpoint, BF16/NF4 distinct conditions. Partial source-order counts are not final accuracy estimates.

| Backend | Status | Root groups complete |
|---|---|---:|
| BF16 | running | 61/89 |
| NF4 | not_loaded | 0/89 |

New pointer readout screen: BF16 pass. This only nominates a task for future causal design; original algebraic gates stay failed and no BPD transfer is enabled.

| Backend | Split | Suite | Condition | Correct | Format valid | Root groups |
|---|---|---|---|---:|---:|---:|
| BF16 | calibration | pointer | copy | 32/32 | 32/32 | 32 |
| BF16 | calibration | pointer | hop1 | 27/32 | 32/32 | 32 |
| BF16 | calibration | pointer | hop2 | 2/32 | 32/32 | 32 |
| BF16 | calibration | pointer | hop3 | 3/32 | 32/32 | 32 |
| BF16 | calibration | pointer | hop4 | 1/32 | 32/32 | 32 |
| BF16 | development | arc1 | copy | 3/3 | 3/3 | 3 |
| BF16 | development | arc1 | main | 1/3 | 3/3 | 3 |
| BF16 | development | arc2 | copy | 3/3 | 3/3 | 3 |
| BF16 | development | arc2 | main | 0/3 | 2/3 | 3 |
| BF16 | development | pointer | copy | 16/16 | 16/16 | 16 |
| BF16 | development | pointer | hop1 | 16/16 | 16/16 | 16 |
| BF16 | development | pointer | hop2 | 2/16 | 16/16 | 16 |
| BF16 | development | pointer | hop3 | 2/16 | 16/16 | 16 |
| BF16 | development | pointer | hop4 | 1/16 | 16/16 | 16 |
| BF16 | development | storyworld | copy | 4/4 | 4/4 | 4 |
| BF16 | development | storyworld | main | 0/4 | 4/4 | 4 |
| BF16 | development | sudoku | copy | 3/3 | 3/3 | 3 |
| BF16 | development | sudoku | main | 1/3 | 3/3 | 3 |

Routing is category classification, not Intellect/GSM8K/ARC QA solution accuracy. ARC-1/2 are local public grid microbenches. Sudoku primary is the upstream exact reference; alternate valid solutions are reported separately in JSON. Storyworld correctness executes the complete plan under the native rival.

SAE fitting uses only the 32 CAL pointer roots. Weight SVD is an algebraic baseline, not a trained VPD/BPD result. No model edits or BPD transfer run; the original 168-case test stays sealed.
