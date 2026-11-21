# NAN-GYM-05 streaming development report

Updated 2026-11-20T07:04:39.018022+00:00

Frozen source-only registration; one checkpoint, BF16/NF4 distinct conditions. Both native backends are complete; results remain descriptive exposed-development evidence.

| Backend | Status | Root groups complete |
|---|---|---:|
| BF16 | completed | 89/89 |
| NF4 | completed | 89/89 |

New pointer readout screen: BF16 pass, NF4 pass. This only nominates a task for future causal design; original algebraic gates stay failed and no BPD transfer is enabled.

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
| BF16 | development | routing | copy | 16/16 | 16/16 | 16 |
| BF16 | development | routing | main | 8/16 | 16/16 | 16 |
| BF16 | development | storyworld | copy | 16/16 | 16/16 | 16 |
| BF16 | development | storyworld | main | 0/16 | 16/16 | 16 |
| BF16 | development | sudoku | copy | 3/3 | 3/3 | 3 |
| BF16 | development | sudoku | main | 1/3 | 3/3 | 3 |
| NF4 | calibration | pointer | copy | 32/32 | 32/32 | 32 |
| NF4 | calibration | pointer | hop1 | 29/32 | 32/32 | 32 |
| NF4 | calibration | pointer | hop2 | 2/32 | 32/32 | 32 |
| NF4 | calibration | pointer | hop3 | 3/32 | 32/32 | 32 |
| NF4 | calibration | pointer | hop4 | 2/32 | 32/32 | 32 |
| NF4 | development | arc1 | copy | 3/3 | 3/3 | 3 |
| NF4 | development | arc1 | main | 1/3 | 3/3 | 3 |
| NF4 | development | arc2 | copy | 3/3 | 3/3 | 3 |
| NF4 | development | arc2 | main | 1/3 | 3/3 | 3 |
| NF4 | development | pointer | copy | 16/16 | 16/16 | 16 |
| NF4 | development | pointer | hop1 | 16/16 | 16/16 | 16 |
| NF4 | development | pointer | hop2 | 1/16 | 16/16 | 16 |
| NF4 | development | pointer | hop3 | 2/16 | 16/16 | 16 |
| NF4 | development | pointer | hop4 | 0/16 | 16/16 | 16 |
| NF4 | development | routing | copy | 16/16 | 16/16 | 16 |
| NF4 | development | routing | main | 4/16 | 14/16 | 16 |
| NF4 | development | storyworld | copy | 16/16 | 16/16 | 16 |
| NF4 | development | storyworld | main | 0/16 | 16/16 | 16 |
| NF4 | development | sudoku | copy | 3/3 | 3/3 | 3 |
| NF4 | development | sudoku | main | 1/3 | 3/3 | 3 |

Completed SAE towers use 256 CAL vectors per layer, shared across both loops. Each fitted and random initial dictionary is checked in physical residual coordinates. These are observational reconstruction diagnostics; normalized MSE is not task accuracy. Full suite/loop group statistics and independent audit counts are in the companion JSON.

| Backend | Layer (zero based) | CAL normalized MSE | CAL features ever active / 128 |
|---|---:|---:|---:|
| BF16 | 5 | 0.098648 | 27/128 |
| BF16 | 11 | 0.108568 | 35/128 |
| BF16 | 17 | 0.176215 | 52/128 |
| BF16 | 21 | 0.165952 | 43/128 |
| NF4 | 5 | 0.100781 | 31/128 |
| NF4 | 11 | 0.120473 | 30/128 |
| NF4 | 17 | 0.201052 | 52/128 |
| NF4 | 21 | 0.179816 | 45/128 |

Sparse-code indices are dictionary specific: BF16 and NF4 feature numbers cannot be directly equated. SVD/Haar direction geometry is a weight baseline; genuine trained VPD/BPD remains pending.

Routing is category classification, not Intellect/GSM8K/ARC QA solution accuracy. ARC-1/2 are local public grid microbenches. Sudoku primary is the upstream exact reference; alternate valid solutions are reported separately in JSON. Storyworld correctness executes the complete plan under the native rival.

SAE fitting uses only the 32 CAL pointer roots. Weight SVD is an algebraic baseline, not a trained VPD/BPD result. No model edits or BPD transfer run; the original 168-case test stays sealed.
