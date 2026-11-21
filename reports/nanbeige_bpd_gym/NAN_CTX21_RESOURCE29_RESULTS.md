# CTX21: separate audit after the BF16 RAM stop

At 4K, loop-2 development main accuracy is BF16 **90/96** and NF4 **92/96**. These are repeated worlds/positions from **16 roots**, not 96 independent puzzles.

This new, frozen CPU-only audit verifies all 378 native BF16 cases closed before the external host-RAM stop: 312 at 4K and 66 at 8K. It pairs all 312 shared 4K initial prefixes with the existing independently audited NF4 cache. NF4 produced no closed 8K cases. All remaining registered outcomes stay unknown.

**The original BF16 terminal audit remains failed because its normal receipt is absent.** No original receipt, successful terminal status, final weight hash or missing inference was synthesized. This report does not replace the canonical CTX21 terminal report or relax its gate. Final target hashes are unavailable for both stopped conditions; before-load parameter bindings and saved output/state integrity are checked.

The retrospective administrative cutoff and lightweight tokenizer wrapper were frozen and pushed before this separate grading pass. Original per-case scientific checks are AST-identical; the parser, tokenizer ID rules, six geometry formulas and root weighting are unchanged. Earlier NF4 outcomes and the BF16 execution counts were already exposed. This is an exposed development cache audit, not a fresh prospective competence test.

## Coverage and resource limits

| Backend | Audit provenance | Checked / registered | Missing | Vectors checked |
| --- | --- | --- | --- | --- |
| BF16 | separate closed-prefix audit; original terminal gate failed | 378 / 624 | 246 | 15120 |
| NF4 | original capacity-stop terminal audit | 312 / 624 | 312 | 12480 |

CPU audit private-memory maximum observed: **194.1 MiB**; minimum system available RAM: **6.467 GiB**. Admission required 5 GiB available, self-only watchdog reserved 4 GiB available and capped private memory at 0.75 GiB. One numerical thread, Below Normal priority, no Torch import, model load, GPU allocation or pretrained forward. Two-second samples are observations, not continuous peak measurements. NF4 RAM28 inference continued separately.

BF16 original inference used four resident decoder layers plus a stage and BF16 head on GPU; NF4 used 22 NF4 decoder layers and a BF16 head. Both used CPU embeddings, the pinned checkpoint, identical initial token IDs, two native loops, no KV cache, and a single free full-vocabulary greedy token per loop. The native head parity checks passed. Placement and memory policies differ, so this is not an isolated quantization-only performance experiment.

BF16 was stopped to protect host RAM, not on an answer-quality criterion. Missingness follows deterministic source order and resource stops; 8K BF16 results are partial and unpaired. No 8K precision comparison is available. No terminal BF16 runtime/memory peak is claimed.

## Every accuracy stratum and both loops

Missing outcomes are unknown. Bounds cover every registered world/root; complete-root success requires both worlds. Copy/erasure are semantic controls, not dose-matched causal interventions. All nulls and format-valid counts remain in the JSON/CSV.

| Backend | Split | Budget | Position | Arm | Loop | Correct / seen | Missing | Whole-roster correct bounds | Both-correct / complete roots | Whole-roster root bounds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bf16 | calibration | 4096 | middle | copy | 1 | 2 / 8 | 0 | [2, 2] / 8 | 0 / 4 | [0, 0] / 4 |
| bf16 | calibration | 4096 | middle | copy | 2 | 8 / 8 | 0 | [8, 8] / 8 | 4 / 4 | [4, 4] / 4 |
| bf16 | calibration | 4096 | middle | erased | 1 | 1 / 8 | 0 | [1, 1] / 8 | 0 / 4 | [0, 0] / 4 |
| bf16 | calibration | 4096 | middle | erased | 2 | 0 / 8 | 0 | [0, 0] / 8 | 0 / 4 | [0, 0] / 4 |
| bf16 | calibration | 4096 | middle | main | 1 | 2 / 8 | 0 | [2, 2] / 8 | 0 / 4 | [0, 0] / 4 |
| bf16 | calibration | 4096 | middle | main | 2 | 8 / 8 | 0 | [8, 8] / 8 | 4 / 4 | [4, 4] / 4 |
| bf16 | calibration | 8192 | middle | copy | 1 | 4 / 8 | 0 | [4, 4] / 8 | 1 / 4 | [1, 1] / 4 |
| bf16 | calibration | 8192 | middle | copy | 2 | 8 / 8 | 0 | [8, 8] / 8 | 4 / 4 | [4, 4] / 4 |
| bf16 | calibration | 8192 | middle | erased | 1 | 1 / 8 | 0 | [1, 1] / 8 | 0 / 4 | [0, 0] / 4 |
| bf16 | calibration | 8192 | middle | erased | 2 | 1 / 8 | 0 | [1, 1] / 8 | 0 / 4 | [0, 0] / 4 |
| bf16 | calibration | 8192 | middle | main | 1 | 1 / 8 | 0 | [1, 1] / 8 | 0 / 4 | [0, 0] / 4 |
| bf16 | calibration | 8192 | middle | main | 2 | 8 / 8 | 0 | [8, 8] / 8 | 4 / 4 | [4, 4] / 4 |
| bf16 | development | 4096 | early | copy | 1 | 4 / 32 | 0 | [4, 4] / 32 | 0 / 16 | [0, 0] / 16 |
| bf16 | development | 4096 | early | copy | 2 | 29 / 32 | 0 | [29, 29] / 32 | 14 / 16 | [14, 14] / 16 |
| bf16 | development | 4096 | early | erased | 1 | 2 / 32 | 0 | [2, 2] / 32 | 0 / 16 | [0, 0] / 16 |
| bf16 | development | 4096 | early | erased | 2 | 1 / 32 | 0 | [1, 1] / 32 | 0 / 16 | [0, 0] / 16 |
| bf16 | development | 4096 | early | main | 1 | 2 / 32 | 0 | [2, 2] / 32 | 0 / 16 | [0, 0] / 16 |
| bf16 | development | 4096 | early | main | 2 | 28 / 32 | 0 | [28, 28] / 32 | 13 / 16 | [13, 13] / 16 |
| bf16 | development | 4096 | late | copy | 1 | 6 / 32 | 0 | [6, 6] / 32 | 0 / 16 | [0, 0] / 16 |
| bf16 | development | 4096 | late | copy | 2 | 32 / 32 | 0 | [32, 32] / 32 | 16 / 16 | [16, 16] / 16 |
| bf16 | development | 4096 | late | erased | 1 | 2 / 32 | 0 | [2, 2] / 32 | 0 / 16 | [0, 0] / 16 |
| bf16 | development | 4096 | late | erased | 2 | 1 / 32 | 0 | [1, 1] / 32 | 0 / 16 | [0, 0] / 16 |
| bf16 | development | 4096 | late | main | 1 | 2 / 32 | 0 | [2, 2] / 32 | 0 / 16 | [0, 0] / 16 |
| bf16 | development | 4096 | late | main | 2 | 32 / 32 | 0 | [32, 32] / 32 | 16 / 16 | [16, 16] / 16 |
| bf16 | development | 4096 | middle | copy | 1 | 4 / 32 | 0 | [4, 4] / 32 | 0 / 16 | [0, 0] / 16 |
| bf16 | development | 4096 | middle | copy | 2 | 30 / 32 | 0 | [30, 30] / 32 | 15 / 16 | [15, 15] / 16 |
| bf16 | development | 4096 | middle | erased | 1 | 2 / 32 | 0 | [2, 2] / 32 | 0 / 16 | [0, 0] / 16 |
| bf16 | development | 4096 | middle | erased | 2 | 0 / 32 | 0 | [0, 0] / 32 | 0 / 16 | [0, 0] / 16 |
| bf16 | development | 4096 | middle | main | 1 | 2 / 32 | 0 | [2, 2] / 32 | 0 / 16 | [0, 0] / 16 |
| bf16 | development | 4096 | middle | main | 2 | 30 / 32 | 0 | [30, 30] / 32 | 15 / 16 | [15, 15] / 16 |
| bf16 | development | 8192 | early | copy | 1 | 3 / 6 | 26 | [3, 29] / 32 | 1 / 3 | [1, 14] / 16 |
| bf16 | development | 8192 | early | copy | 2 | 6 / 6 | 26 | [6, 32] / 32 | 3 / 3 | [3, 16] / 16 |
| bf16 | development | 8192 | early | erased | 1 | 1 / 6 | 26 | [1, 27] / 32 | 0 / 3 | [0, 13] / 16 |
| bf16 | development | 8192 | early | erased | 2 | 0 / 6 | 26 | [0, 26] / 32 | 0 / 3 | [0, 13] / 16 |
| bf16 | development | 8192 | early | main | 1 | 1 / 6 | 26 | [1, 27] / 32 | 0 / 3 | [0, 13] / 16 |
| bf16 | development | 8192 | early | main | 2 | 6 / 6 | 26 | [6, 32] / 32 | 3 / 3 | [3, 16] / 16 |
| bf16 | development | 8192 | late | copy | 1 | 2 / 4 | 28 | [2, 30] / 32 | 0 / 2 | [0, 14] / 16 |
| bf16 | development | 8192 | late | copy | 2 | 4 / 4 | 28 | [4, 32] / 32 | 2 / 2 | [2, 16] / 16 |
| bf16 | development | 8192 | late | erased | 1 | 0 / 4 | 28 | [0, 28] / 32 | 0 / 2 | [0, 14] / 16 |
| bf16 | development | 8192 | late | erased | 2 | 0 / 4 | 28 | [0, 28] / 32 | 0 / 2 | [0, 14] / 16 |
| bf16 | development | 8192 | late | main | 1 | 0 / 4 | 28 | [0, 28] / 32 | 0 / 2 | [0, 14] / 16 |
| bf16 | development | 8192 | late | main | 2 | 4 / 4 | 28 | [4, 32] / 32 | 2 / 2 | [2, 16] / 16 |
| bf16 | development | 8192 | middle | copy | 1 | 1 / 4 | 28 | [1, 29] / 32 | 0 / 2 | [0, 14] / 16 |
| bf16 | development | 8192 | middle | copy | 2 | 4 / 4 | 28 | [4, 32] / 32 | 2 / 2 | [2, 16] / 16 |
| bf16 | development | 8192 | middle | erased | 1 | 0 / 4 | 28 | [0, 28] / 32 | 0 / 2 | [0, 14] / 16 |
| bf16 | development | 8192 | middle | erased | 2 | 0 / 4 | 28 | [0, 28] / 32 | 0 / 2 | [0, 14] / 16 |
| bf16 | development | 8192 | middle | main | 1 | 1 / 4 | 28 | [1, 29] / 32 | 0 / 2 | [0, 14] / 16 |
| bf16 | development | 8192 | middle | main | 2 | 4 / 4 | 28 | [4, 32] / 32 | 2 / 2 | [2, 16] / 16 |
| nf4 | calibration | 4096 | middle | copy | 1 | 4 / 8 | 0 | [4, 4] / 8 | 1 / 4 | [1, 1] / 4 |
| nf4 | calibration | 4096 | middle | copy | 2 | 8 / 8 | 0 | [8, 8] / 8 | 4 / 4 | [4, 4] / 4 |
| nf4 | calibration | 4096 | middle | erased | 1 | 1 / 8 | 0 | [1, 1] / 8 | 0 / 4 | [0, 0] / 4 |
| nf4 | calibration | 4096 | middle | erased | 2 | 1 / 8 | 0 | [1, 1] / 8 | 0 / 4 | [0, 0] / 4 |
| nf4 | calibration | 4096 | middle | main | 1 | 1 / 8 | 0 | [1, 1] / 8 | 0 / 4 | [0, 0] / 4 |
| nf4 | calibration | 4096 | middle | main | 2 | 7 / 8 | 0 | [7, 7] / 8 | 3 / 4 | [3, 3] / 4 |
| nf4 | calibration | 8192 | middle | copy | 1 | — / 0 | 8 | [0, 8] / 8 | — / 0 | [0, 4] / 4 |
| nf4 | calibration | 8192 | middle | copy | 2 | — / 0 | 8 | [0, 8] / 8 | — / 0 | [0, 4] / 4 |
| nf4 | calibration | 8192 | middle | erased | 1 | — / 0 | 8 | [0, 8] / 8 | — / 0 | [0, 4] / 4 |
| nf4 | calibration | 8192 | middle | erased | 2 | — / 0 | 8 | [0, 8] / 8 | — / 0 | [0, 4] / 4 |
| nf4 | calibration | 8192 | middle | main | 1 | — / 0 | 8 | [0, 8] / 8 | — / 0 | [0, 4] / 4 |
| nf4 | calibration | 8192 | middle | main | 2 | — / 0 | 8 | [0, 8] / 8 | — / 0 | [0, 4] / 4 |
| nf4 | development | 4096 | early | copy | 1 | 8 / 32 | 0 | [8, 8] / 32 | 1 / 16 | [1, 1] / 16 |
| nf4 | development | 4096 | early | copy | 2 | 31 / 32 | 0 | [31, 31] / 32 | 15 / 16 | [15, 15] / 16 |
| nf4 | development | 4096 | early | erased | 1 | 2 / 32 | 0 | [2, 2] / 32 | 0 / 16 | [0, 0] / 16 |
| nf4 | development | 4096 | early | erased | 2 | 1 / 32 | 0 | [1, 1] / 32 | 0 / 16 | [0, 0] / 16 |
| nf4 | development | 4096 | early | main | 1 | 2 / 32 | 0 | [2, 2] / 32 | 0 / 16 | [0, 0] / 16 |
| nf4 | development | 4096 | early | main | 2 | 30 / 32 | 0 | [30, 30] / 32 | 15 / 16 | [15, 15] / 16 |
| nf4 | development | 4096 | late | copy | 1 | 9 / 32 | 0 | [9, 9] / 32 | 1 / 16 | [1, 1] / 16 |
| nf4 | development | 4096 | late | copy | 2 | 32 / 32 | 0 | [32, 32] / 32 | 16 / 16 | [16, 16] / 16 |
| nf4 | development | 4096 | late | erased | 1 | 2 / 32 | 0 | [2, 2] / 32 | 0 / 16 | [0, 0] / 16 |
| nf4 | development | 4096 | late | erased | 2 | 2 / 32 | 0 | [2, 2] / 32 | 0 / 16 | [0, 0] / 16 |
| nf4 | development | 4096 | late | main | 1 | 3 / 32 | 0 | [3, 3] / 32 | 0 / 16 | [0, 0] / 16 |
| nf4 | development | 4096 | late | main | 2 | 32 / 32 | 0 | [32, 32] / 32 | 16 / 16 | [16, 16] / 16 |
| nf4 | development | 4096 | middle | copy | 1 | 9 / 32 | 0 | [9, 9] / 32 | 2 / 16 | [2, 2] / 16 |
| nf4 | development | 4096 | middle | copy | 2 | 31 / 32 | 0 | [31, 31] / 32 | 15 / 16 | [15, 15] / 16 |
| nf4 | development | 4096 | middle | erased | 1 | 2 / 32 | 0 | [2, 2] / 32 | 0 / 16 | [0, 0] / 16 |
| nf4 | development | 4096 | middle | erased | 2 | 1 / 32 | 0 | [1, 1] / 32 | 0 / 16 | [0, 0] / 16 |
| nf4 | development | 4096 | middle | main | 1 | 3 / 32 | 0 | [3, 3] / 32 | 0 / 16 | [0, 0] / 16 |
| nf4 | development | 4096 | middle | main | 2 | 30 / 32 | 0 | [30, 30] / 32 | 15 / 16 | [15, 15] / 16 |
| nf4 | development | 8192 | early | copy | 1 | — / 0 | 32 | [0, 32] / 32 | — / 0 | [0, 16] / 16 |
| nf4 | development | 8192 | early | copy | 2 | — / 0 | 32 | [0, 32] / 32 | — / 0 | [0, 16] / 16 |
| nf4 | development | 8192 | early | erased | 1 | — / 0 | 32 | [0, 32] / 32 | — / 0 | [0, 16] / 16 |
| nf4 | development | 8192 | early | erased | 2 | — / 0 | 32 | [0, 32] / 32 | — / 0 | [0, 16] / 16 |
| nf4 | development | 8192 | early | main | 1 | — / 0 | 32 | [0, 32] / 32 | — / 0 | [0, 16] / 16 |
| nf4 | development | 8192 | early | main | 2 | — / 0 | 32 | [0, 32] / 32 | — / 0 | [0, 16] / 16 |
| nf4 | development | 8192 | late | copy | 1 | — / 0 | 32 | [0, 32] / 32 | — / 0 | [0, 16] / 16 |
| nf4 | development | 8192 | late | copy | 2 | — / 0 | 32 | [0, 32] / 32 | — / 0 | [0, 16] / 16 |
| nf4 | development | 8192 | late | erased | 1 | — / 0 | 32 | [0, 32] / 32 | — / 0 | [0, 16] / 16 |
| nf4 | development | 8192 | late | erased | 2 | — / 0 | 32 | [0, 32] / 32 | — / 0 | [0, 16] / 16 |
| nf4 | development | 8192 | late | main | 1 | — / 0 | 32 | [0, 32] / 32 | — / 0 | [0, 16] / 16 |
| nf4 | development | 8192 | late | main | 2 | — / 0 | 32 | [0, 32] / 32 | — / 0 | [0, 16] / 16 |
| nf4 | development | 8192 | middle | copy | 1 | — / 0 | 32 | [0, 32] / 32 | — / 0 | [0, 16] / 16 |
| nf4 | development | 8192 | middle | copy | 2 | — / 0 | 32 | [0, 32] / 32 | — / 0 | [0, 16] / 16 |
| nf4 | development | 8192 | middle | erased | 1 | — / 0 | 32 | [0, 32] / 32 | — / 0 | [0, 16] / 16 |
| nf4 | development | 8192 | middle | erased | 2 | — / 0 | 32 | [0, 32] / 32 | — / 0 | [0, 16] / 16 |
| nf4 | development | 8192 | middle | main | 1 | — / 0 | 32 | [0, 32] / 32 | — / 0 | [0, 16] / 16 |
| nf4 | development | 8192 | middle | main | 2 | — / 0 | 32 | [0, 32] / 32 | — / 0 | [0, 16] / 16 |

## Every matched correctness stratum

These paired counts share prompts and worlds. They do not supply independent root-level formal comparisons.

| Split | Budget | Position | Arm | Loop | Both correct | BF16 only | NF4 only | Neither | Matched | Missing pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calibration | 4096 | middle | copy | 1 | 2 | 0 | 2 | 4 | 8 | 0 |
| calibration | 4096 | middle | copy | 2 | 8 | 0 | 0 | 0 | 8 | 0 |
| calibration | 4096 | middle | erased | 1 | 1 | 0 | 0 | 7 | 8 | 0 |
| calibration | 4096 | middle | erased | 2 | 0 | 0 | 1 | 7 | 8 | 0 |
| calibration | 4096 | middle | main | 1 | 1 | 1 | 0 | 6 | 8 | 0 |
| calibration | 4096 | middle | main | 2 | 7 | 1 | 0 | 0 | 8 | 0 |
| calibration | 8192 | middle | copy | 1 | 0 | 0 | 0 | 0 | 0 | 8 |
| calibration | 8192 | middle | copy | 2 | 0 | 0 | 0 | 0 | 0 | 8 |
| calibration | 8192 | middle | erased | 1 | 0 | 0 | 0 | 0 | 0 | 8 |
| calibration | 8192 | middle | erased | 2 | 0 | 0 | 0 | 0 | 0 | 8 |
| calibration | 8192 | middle | main | 1 | 0 | 0 | 0 | 0 | 0 | 8 |
| calibration | 8192 | middle | main | 2 | 0 | 0 | 0 | 0 | 0 | 8 |
| development | 4096 | early | copy | 1 | 4 | 0 | 4 | 24 | 32 | 0 |
| development | 4096 | early | copy | 2 | 28 | 1 | 3 | 0 | 32 | 0 |
| development | 4096 | early | erased | 1 | 2 | 0 | 0 | 30 | 32 | 0 |
| development | 4096 | early | erased | 2 | 0 | 1 | 1 | 30 | 32 | 0 |
| development | 4096 | early | main | 1 | 0 | 2 | 2 | 28 | 32 | 0 |
| development | 4096 | early | main | 2 | 27 | 1 | 3 | 1 | 32 | 0 |
| development | 4096 | late | copy | 1 | 6 | 0 | 3 | 23 | 32 | 0 |
| development | 4096 | late | copy | 2 | 32 | 0 | 0 | 0 | 32 | 0 |
| development | 4096 | late | erased | 1 | 2 | 0 | 0 | 30 | 32 | 0 |
| development | 4096 | late | erased | 2 | 0 | 1 | 2 | 29 | 32 | 0 |
| development | 4096 | late | main | 1 | 2 | 0 | 1 | 29 | 32 | 0 |
| development | 4096 | late | main | 2 | 32 | 0 | 0 | 0 | 32 | 0 |
| development | 4096 | middle | copy | 1 | 4 | 0 | 5 | 23 | 32 | 0 |
| development | 4096 | middle | copy | 2 | 29 | 1 | 2 | 0 | 32 | 0 |
| development | 4096 | middle | erased | 1 | 1 | 1 | 1 | 29 | 32 | 0 |
| development | 4096 | middle | erased | 2 | 0 | 0 | 1 | 31 | 32 | 0 |
| development | 4096 | middle | main | 1 | 2 | 0 | 1 | 29 | 32 | 0 |
| development | 4096 | middle | main | 2 | 30 | 0 | 0 | 2 | 32 | 0 |
| development | 8192 | early | copy | 1 | 0 | 0 | 0 | 0 | 0 | 32 |
| development | 8192 | early | copy | 2 | 0 | 0 | 0 | 0 | 0 | 32 |
| development | 8192 | early | erased | 1 | 0 | 0 | 0 | 0 | 0 | 32 |
| development | 8192 | early | erased | 2 | 0 | 0 | 0 | 0 | 0 | 32 |
| development | 8192 | early | main | 1 | 0 | 0 | 0 | 0 | 0 | 32 |
| development | 8192 | early | main | 2 | 0 | 0 | 0 | 0 | 0 | 32 |
| development | 8192 | late | copy | 1 | 0 | 0 | 0 | 0 | 0 | 32 |
| development | 8192 | late | copy | 2 | 0 | 0 | 0 | 0 | 0 | 32 |
| development | 8192 | late | erased | 1 | 0 | 0 | 0 | 0 | 0 | 32 |
| development | 8192 | late | erased | 2 | 0 | 0 | 0 | 0 | 0 | 32 |
| development | 8192 | late | main | 1 | 0 | 0 | 0 | 0 | 0 | 32 |
| development | 8192 | late | main | 2 | 0 | 0 | 0 | 0 | 0 | 32 |
| development | 8192 | middle | copy | 1 | 0 | 0 | 0 | 0 | 0 | 32 |
| development | 8192 | middle | copy | 2 | 0 | 0 | 0 | 0 | 0 | 32 |
| development | 8192 | middle | erased | 1 | 0 | 0 | 0 | 0 | 0 | 32 |
| development | 8192 | middle | erased | 2 | 0 | 0 | 0 | 0 | 0 | 32 |
| development | 8192 | middle | main | 1 | 0 | 0 | 0 | 0 | 0 | 32 |
| development | 8192 | middle | main | 2 | 0 | 0 | 0 | 0 | 0 | 32 |

## Every physical geometry stratum

Physical residuals before final loop RMSNorm, at zero-based layers 5/11/17/21. Displacement Δ = h₂ − h₁; relative L2 uses the BF16 vector norm. Available worlds are averaged within each root, then roots equally. Zero-norm metrics are undefined. No coordinate alignment or independently fitted SAE-index subtraction is used. The scalar CSV includes defined roots and undefined-context counts separately for every metric.

| Split | Budget | Position | Arm | Layer | Matched / registered | Observed / registered roots | Partial roots | h1 relL2 | h1 cosine | h2 relL2 | h2 cosine | Δ relL2 | Δ cosine |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calibration | 4096 | middle | copy | 5 | 8 / 8 | 4 / 4 | 0 | 0.153985779 | 0.988182064 | 0.179015014 | 0.984040309 | 0.294000501 | 0.957035377 |
| calibration | 4096 | middle | copy | 11 | 8 / 8 | 4 / 4 | 0 | 0.209373905 | 0.978254822 | 0.21170759 | 0.977309757 | 0.374049571 | 0.938178701 |
| calibration | 4096 | middle | copy | 17 | 8 / 8 | 4 / 4 | 0 | 0.231447824 | 0.972872561 | 0.249079989 | 0.968320662 | 0.387359839 | 0.926651648 |
| calibration | 4096 | middle | copy | 21 | 8 / 8 | 4 / 4 | 0 | 0.0503630908 | 0.999027878 | 0.15074706 | 0.988920447 | 0.0739177923 | 0.99772076 |
| calibration | 4096 | middle | erased | 5 | 8 / 8 | 4 / 4 | 0 | 0.161373151 | 0.986999881 | 0.168566147 | 0.985711303 | 0.274578313 | 0.96211567 |
| calibration | 4096 | middle | erased | 11 | 8 / 8 | 4 / 4 | 0 | 0.21736066 | 0.976541551 | 0.217273707 | 0.976431036 | 0.354918969 | 0.941229066 |
| calibration | 4096 | middle | erased | 17 | 8 / 8 | 4 / 4 | 0 | 0.230971422 | 0.973031707 | 0.356213755 | 0.933101121 | 0.423743998 | 0.912082738 |
| calibration | 4096 | middle | erased | 21 | 8 / 8 | 4 / 4 | 0 | 0.0665813822 | 0.999044541 | 0.246833334 | 0.972627533 | 0.103222082 | 0.996868338 |
| calibration | 4096 | middle | main | 5 | 8 / 8 | 4 / 4 | 0 | 0.161853574 | 0.986915422 | 0.166160319 | 0.986122961 | 0.27177421 | 0.96278768 |
| calibration | 4096 | middle | main | 11 | 8 / 8 | 4 / 4 | 0 | 0.221329368 | 0.97565743 | 0.192872935 | 0.981316758 | 0.33907272 | 0.945409783 |
| calibration | 4096 | middle | main | 17 | 8 / 8 | 4 / 4 | 0 | 0.233781149 | 0.972376182 | 0.260915474 | 0.965520577 | 0.364318941 | 0.933699194 |
| calibration | 4096 | middle | main | 21 | 8 / 8 | 4 / 4 | 0 | 0.0709444919 | 0.999057298 | 0.152912639 | 0.98875864 | 0.090319479 | 0.997778121 |
| calibration | 8192 | middle | copy | 5 | 0 / 8 | 0 / 4 | 0 | — | — | — | — | — | — |
| calibration | 8192 | middle | copy | 11 | 0 / 8 | 0 / 4 | 0 | — | — | — | — | — | — |
| calibration | 8192 | middle | copy | 17 | 0 / 8 | 0 / 4 | 0 | — | — | — | — | — | — |
| calibration | 8192 | middle | copy | 21 | 0 / 8 | 0 / 4 | 0 | — | — | — | — | — | — |
| calibration | 8192 | middle | erased | 5 | 0 / 8 | 0 / 4 | 0 | — | — | — | — | — | — |
| calibration | 8192 | middle | erased | 11 | 0 / 8 | 0 / 4 | 0 | — | — | — | — | — | — |
| calibration | 8192 | middle | erased | 17 | 0 / 8 | 0 / 4 | 0 | — | — | — | — | — | — |
| calibration | 8192 | middle | erased | 21 | 0 / 8 | 0 / 4 | 0 | — | — | — | — | — | — |
| calibration | 8192 | middle | main | 5 | 0 / 8 | 0 / 4 | 0 | — | — | — | — | — | — |
| calibration | 8192 | middle | main | 11 | 0 / 8 | 0 / 4 | 0 | — | — | — | — | — | — |
| calibration | 8192 | middle | main | 17 | 0 / 8 | 0 / 4 | 0 | — | — | — | — | — | — |
| calibration | 8192 | middle | main | 21 | 0 / 8 | 0 / 4 | 0 | — | — | — | — | — | — |
| development | 4096 | early | copy | 5 | 32 / 32 | 16 / 16 | 0 | 0.153396113 | 0.988275842 | 0.180994603 | 0.983826781 | 0.297388017 | 0.955881493 |
| development | 4096 | early | copy | 11 | 32 / 32 | 16 / 16 | 0 | 0.210852674 | 0.977953123 | 0.215742154 | 0.976385047 | 0.383026589 | 0.9355626 |
| development | 4096 | early | copy | 17 | 32 / 32 | 16 / 16 | 0 | 0.232137138 | 0.972710013 | 0.243761596 | 0.968769268 | 0.383154401 | 0.927182104 |
| development | 4096 | early | copy | 21 | 32 / 32 | 16 / 16 | 0 | 0.0518038232 | 0.999023245 | 0.149491051 | 0.989162523 | 0.069457066 | 0.997842405 |
| development | 4096 | early | erased | 5 | 32 / 32 | 16 / 16 | 0 | 0.161447192 | 0.986985183 | 0.164282358 | 0.98647123 | 0.271358237 | 0.962928463 |
| development | 4096 | early | erased | 11 | 32 / 32 | 16 / 16 | 0 | 0.219333892 | 0.976039584 | 0.189662336 | 0.98189707 | 0.341248418 | 0.945270779 |
| development | 4096 | early | erased | 17 | 32 / 32 | 16 / 16 | 0 | 0.231265365 | 0.972926087 | 0.307893056 | 0.950422589 | 0.406611121 | 0.919065029 |
| development | 4096 | early | erased | 21 | 32 / 32 | 16 / 16 | 0 | 0.060776144 | 0.999044221 | 0.19068457 | 0.981992583 | 0.0885345446 | 0.997416568 |
| development | 4096 | early | main | 5 | 32 / 32 | 16 / 16 | 0 | 0.16170397 | 0.98694242 | 0.169284993 | 0.985638894 | 0.278627483 | 0.960987804 |
| development | 4096 | early | main | 11 | 32 / 32 | 16 / 16 | 0 | 0.222807656 | 0.975279235 | 0.192367161 | 0.981439233 | 0.345791074 | 0.944164528 |
| development | 4096 | early | main | 17 | 32 / 32 | 16 / 16 | 0 | 0.240818901 | 0.97061089 | 0.245974691 | 0.968291382 | 0.367243615 | 0.932794295 |
| development | 4096 | early | main | 21 | 32 / 32 | 16 / 16 | 0 | 0.0621757127 | 0.999010562 | 0.152485883 | 0.989047825 | 0.0773732263 | 0.998046532 |
| development | 4096 | late | copy | 5 | 32 / 32 | 16 / 16 | 0 | 0.153539946 | 0.98825695 | 0.17795229 | 0.984247837 | 0.291363457 | 0.957704558 |
| development | 4096 | late | copy | 11 | 32 / 32 | 16 / 16 | 0 | 0.210647167 | 0.978053878 | 0.205866469 | 0.978562982 | 0.370328334 | 0.938920921 |
| development | 4096 | late | copy | 17 | 32 / 32 | 16 / 16 | 0 | 0.232393768 | 0.972651038 | 0.244237954 | 0.969576408 | 0.383920626 | 0.927065039 |
| development | 4096 | late | copy | 21 | 32 / 32 | 16 / 16 | 0 | 0.0506785611 | 0.999026428 | 0.144658654 | 0.989880678 | 0.0698530189 | 0.997864958 |
| development | 4096 | late | erased | 5 | 32 / 32 | 16 / 16 | 0 | 0.155926189 | 0.987909691 | 0.163956534 | 0.986557189 | 0.267833593 | 0.963806479 |
| development | 4096 | late | erased | 11 | 32 / 32 | 16 / 16 | 0 | 0.213602294 | 0.977414733 | 0.187658071 | 0.982269806 | 0.329875886 | 0.948679161 |
| development | 4096 | late | erased | 17 | 32 / 32 | 16 / 16 | 0 | 0.227041263 | 0.973917684 | 0.304935677 | 0.951531837 | 0.384563884 | 0.926366478 |
| development | 4096 | late | erased | 21 | 32 / 32 | 16 / 16 | 0 | 0.0576591761 | 0.999094739 | 0.198971248 | 0.981797637 | 0.088274164 | 0.997482941 |
| development | 4096 | late | main | 5 | 32 / 32 | 16 / 16 | 0 | 0.161772795 | 0.986925921 | 0.166884891 | 0.986031236 | 0.272398053 | 0.962616803 |
| development | 4096 | late | main | 11 | 32 / 32 | 16 / 16 | 0 | 0.221736158 | 0.975658295 | 0.191311731 | 0.981550893 | 0.341547539 | 0.945047032 |
| development | 4096 | late | main | 17 | 32 / 32 | 16 / 16 | 0 | 0.236662325 | 0.971655439 | 0.26276951 | 0.964719937 | 0.368244247 | 0.932224272 |
| development | 4096 | late | main | 21 | 32 / 32 | 16 / 16 | 0 | 0.0670341767 | 0.999047277 | 0.153180515 | 0.988762026 | 0.0822445096 | 0.997786837 |
| development | 4096 | middle | copy | 5 | 32 / 32 | 16 / 16 | 0 | 0.153801769 | 0.988215208 | 0.180554191 | 0.983790509 | 0.294253925 | 0.956959279 |
| development | 4096 | middle | copy | 11 | 32 / 32 | 16 / 16 | 0 | 0.210609778 | 0.978021791 | 0.212654828 | 0.977092915 | 0.374272949 | 0.938294033 |
| development | 4096 | middle | copy | 17 | 32 / 32 | 16 / 16 | 0 | 0.234431109 | 0.972171925 | 0.255408981 | 0.966647538 | 0.384679762 | 0.9279177 |
| development | 4096 | middle | copy | 21 | 32 / 32 | 16 / 16 | 0 | 0.0529786471 | 0.999018727 | 0.149923127 | 0.98942738 | 0.0728019245 | 0.997824125 |
| development | 4096 | middle | erased | 5 | 32 / 32 | 16 / 16 | 0 | 0.161537724 | 0.986963419 | 0.168788279 | 0.985661558 | 0.27604075 | 0.961697216 |
| development | 4096 | middle | erased | 11 | 32 / 32 | 16 / 16 | 0 | 0.21874829 | 0.976291417 | 0.207579809 | 0.978409095 | 0.348409638 | 0.943060351 |
| development | 4096 | middle | erased | 17 | 32 / 32 | 16 / 16 | 0 | 0.231636705 | 0.972875377 | 0.338568739 | 0.938282872 | 0.414233347 | 0.915918575 |
| development | 4096 | middle | erased | 21 | 32 / 32 | 16 / 16 | 0 | 0.0599087715 | 0.99905174 | 0.232274878 | 0.975049755 | 0.0955646961 | 0.996936403 |
| development | 4096 | middle | main | 5 | 32 / 32 | 16 / 16 | 0 | 0.161661314 | 0.986949365 | 0.167350783 | 0.985944652 | 0.273944305 | 0.962165976 |
| development | 4096 | middle | main | 11 | 32 / 32 | 16 / 16 | 0 | 0.221727037 | 0.975603373 | 0.192537113 | 0.981338273 | 0.339500188 | 0.945431022 |
| development | 4096 | middle | main | 17 | 32 / 32 | 16 / 16 | 0 | 0.236290498 | 0.971768407 | 0.258449934 | 0.965947855 | 0.363132948 | 0.934746471 |
| development | 4096 | middle | main | 21 | 32 / 32 | 16 / 16 | 0 | 0.0676101014 | 0.999053723 | 0.147446431 | 0.989427395 | 0.0822910168 | 0.997930208 |
| development | 8192 | early | copy | 5 | 0 / 32 | 0 / 16 | 0 | — | — | — | — | — | — |
| development | 8192 | early | copy | 11 | 0 / 32 | 0 / 16 | 0 | — | — | — | — | — | — |
| development | 8192 | early | copy | 17 | 0 / 32 | 0 / 16 | 0 | — | — | — | — | — | — |
| development | 8192 | early | copy | 21 | 0 / 32 | 0 / 16 | 0 | — | — | — | — | — | — |
| development | 8192 | early | erased | 5 | 0 / 32 | 0 / 16 | 0 | — | — | — | — | — | — |
| development | 8192 | early | erased | 11 | 0 / 32 | 0 / 16 | 0 | — | — | — | — | — | — |
| development | 8192 | early | erased | 17 | 0 / 32 | 0 / 16 | 0 | — | — | — | — | — | — |
| development | 8192 | early | erased | 21 | 0 / 32 | 0 / 16 | 0 | — | — | — | — | — | — |
| development | 8192 | early | main | 5 | 0 / 32 | 0 / 16 | 0 | — | — | — | — | — | — |
| development | 8192 | early | main | 11 | 0 / 32 | 0 / 16 | 0 | — | — | — | — | — | — |
| development | 8192 | early | main | 17 | 0 / 32 | 0 / 16 | 0 | — | — | — | — | — | — |
| development | 8192 | early | main | 21 | 0 / 32 | 0 / 16 | 0 | — | — | — | — | — | — |
| development | 8192 | late | copy | 5 | 0 / 32 | 0 / 16 | 0 | — | — | — | — | — | — |
| development | 8192 | late | copy | 11 | 0 / 32 | 0 / 16 | 0 | — | — | — | — | — | — |
| development | 8192 | late | copy | 17 | 0 / 32 | 0 / 16 | 0 | — | — | — | — | — | — |
| development | 8192 | late | copy | 21 | 0 / 32 | 0 / 16 | 0 | — | — | — | — | — | — |
| development | 8192 | late | erased | 5 | 0 / 32 | 0 / 16 | 0 | — | — | — | — | — | — |
| development | 8192 | late | erased | 11 | 0 / 32 | 0 / 16 | 0 | — | — | — | — | — | — |
| development | 8192 | late | erased | 17 | 0 / 32 | 0 / 16 | 0 | — | — | — | — | — | — |
| development | 8192 | late | erased | 21 | 0 / 32 | 0 / 16 | 0 | — | — | — | — | — | — |
| development | 8192 | late | main | 5 | 0 / 32 | 0 / 16 | 0 | — | — | — | — | — | — |
| development | 8192 | late | main | 11 | 0 / 32 | 0 / 16 | 0 | — | — | — | — | — | — |
| development | 8192 | late | main | 17 | 0 / 32 | 0 / 16 | 0 | — | — | — | — | — | — |
| development | 8192 | late | main | 21 | 0 / 32 | 0 / 16 | 0 | — | — | — | — | — | — |
| development | 8192 | middle | copy | 5 | 0 / 32 | 0 / 16 | 0 | — | — | — | — | — | — |
| development | 8192 | middle | copy | 11 | 0 / 32 | 0 / 16 | 0 | — | — | — | — | — | — |
| development | 8192 | middle | copy | 17 | 0 / 32 | 0 / 16 | 0 | — | — | — | — | — | — |
| development | 8192 | middle | copy | 21 | 0 / 32 | 0 / 16 | 0 | — | — | — | — | — | — |
| development | 8192 | middle | erased | 5 | 0 / 32 | 0 / 16 | 0 | — | — | — | — | — | — |
| development | 8192 | middle | erased | 11 | 0 / 32 | 0 / 16 | 0 | — | — | — | — | — | — |
| development | 8192 | middle | erased | 17 | 0 / 32 | 0 / 16 | 0 | — | — | — | — | — | — |
| development | 8192 | middle | erased | 21 | 0 / 32 | 0 / 16 | 0 | — | — | — | — | — | — |
| development | 8192 | middle | main | 5 | 0 / 32 | 0 / 16 | 0 | — | — | — | — | — | — |
| development | 8192 | middle | main | 11 | 0 / 32 | 0 / 16 | 0 | — | — | — | — | — | — |
| development | 8192 | middle | main | 17 | 0 / 32 | 0 / 16 | 0 | — | — | — | — | — | — |
| development | 8192 | middle | main | 21 | 0 / 32 | 0 / 16 | 0 | — | — | — | — | — | — |

## Interpretation and provenance

Accuracy, state similarity and displacement differences are descriptive. High cosine is not equivalence, mediation or faithful reasoning evidence. Effect/sham floors and causal power remain uncalibrated; planned/actual formal comparisons are zero, with resolution **REDESIGN_OR_PILOT_ONLY**. No BPD donor transfer or SAE analysis is claimed for this cache. Earlier failed competence gates remain unchanged. The original 168-case test stays sealed; the continuous research goal remains active.

Original source seal: `d3d0587416d0bf92e7b5d044f527d1fc6bbb630c6d3f074e5fcf0d1e03b6f4bc`. Separate audit seal: `f0c03a11cf5383aa1d66072d8b903f25d4cc884a3bef0a891181448eda6c115b`.

BF16 closed-prefix audit: `da7f55aa40f6d9bbe263734c45854b62c9ede411dac81068484798ee5c9249fd`. Original NF4 audit: `5d37567808dfcac1b54bb70e9e6e6280ec8cd23912a015b7edf4b29b12b1ac72`. Stop attestation: `a28c7615c91c480955000246f46e47946062105137e96c80fea15317ed32d3d0`.

[Protocol](../../experiments/nan_ctx21_resource29/PROTOCOL.md) · [Registration](../../experiments/nan_ctx21_resource29/registration.json) · [All aggregate data](nan_ctx21_resource29_results.json) · [Accuracy CSV](nan_ctx21_resource29_results_accuracy.csv) · [Paired CSV](nan_ctx21_resource29_results_paired.csv) · [576 geometry scalars](nan_ctx21_resource29_results_geometry.csv) · [Original terminal report](NAN_CTX21_PROGRESS.md)
