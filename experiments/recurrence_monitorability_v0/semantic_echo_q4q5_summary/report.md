# SEM-ECHO-01 on the trained Q4/Q5 checkpoints

**Date:** 2026-11-11
**Evidence stage:** retrospective development. The 24 checkpoints and their Q9 extrapolation labels were known before this run. No decision rule is fitted here.
**Code:** branch `rmp-v0-phase2` at `d7ff997`, plus [`semantic_echo_summary.py`](../../../research_gym/scripts/semantic_echo_summary.py) for collation.
**Full tables:** [summary.md](summary.md), [summary.json](summary.json). Per-checkpoint receipts: [`semantic_echo_q4/`](../semantic_echo_q4), [`semantic_echo_q5/`](../semantic_echo_q5).

## Run

```bash
python -m research_gym.scripts.loopscope_semantic_echo \
  --checkpoint-dir experiments/recurrence_monitorability_v0/loopscope_toy/ckpt \
  --out experiments/recurrence_monitorability_v0/semantic_echo_q4 \
  --items 64 --max-loops 32 --horizon 4 --max-pairs 16 \
  --seed 940117 --device cuda --dtype float32
# same with loopscope_toy_q5/ckpt -> semantic_echo_q5

python -m research_gym.scripts.semantic_echo_summary \
  --runs experiments/recurrence_monitorability_v0/semantic_echo_q4 experiments/recurrence_monitorability_v0/semantic_echo_q5 \
  --q9-dir experiments/recurrence_monitorability_v0/loopscope_toy_q9 \
  --out experiments/recurrence_monitorability_v0/semantic_echo_q4q5_summary
```

Environment: Python 3.12.10, PyTorch 2.11.0+cu128, NumPy 2.5.3, RTX 5080 Laptop GPU, float32. The 35 semantic-echo unit tests passed in this environment before the run. Each checkpoint hash matches the hash in its Q9 record.

## Execution checks

- Native trace parity passed on all 24 checkpoints (2048 state and 2048 logit comparisons each).
- Full-donor replay passed on 384/384 capped pairs. The largest relative state L2 deviation was 7.9e-7 (float32, CUDA).
- The per-token LayerNorm shift changed no future label on any checkpoint.

## Findings

**1. Every Q9 extrapolation failure shows endpoint-gated continuation disagreement.** The seven arms with Q9 held-out accuracy below 0.9 have item-mean disagreement between 0.11 and 0.65. Thirteen of the 17 remaining arms show no disagreement on any gated pair.

**2. Two Q9 passes fail beyond depth 16, and the 32-loop trace detects them.** On this run's own-head accuracy, `lam0_s8` (Q9 held-out 0.981) scores 0.967 over loops 9–16 and 0.476 over loops 17–32. `lam0.01_s4` (Q9 0.990) scores 0.994, then 0.556. Their primary disagreement is 0.379 and 0.236. The disagreement occurs only in pairs whose receiver lies past loop 8; within loops 1–8 the step mismatch rate is 0.000 for both. Q9's held-out window ends at depth 16, so it cannot see this degradation. Two further passes, `lam0.01_s5` and `lam0_s9`, each have one disagreeing item, which gives an item mean of 0.0003.

**3. On this set the primary score tracks deep own-head inaccuracy.** Across the 24 arms:
- Spearman(primary, 1 − accuracy over loops 9–32) = 0.922.
- Spearman(primary, Q9 held-out) = −0.891.
- Spearman(primary, F1/F8) = 0.814.

These correlations use average ranks, and the arms with zero disagreement are tied. With the own head as observer, the primary score separates these arms only about as well as deep accuracy over the same trace does. Its distinct contribution is item-level: it shows which equal-label states fail to determine the future.

**4. The disagreement comes from items with few gated pairs.** In failing arms, disagreeing items have a median of 1.5–19.5 gated pairs. These are typically isolated visits whose endpoints happen to be correct. In five of the failing arms, agreeing items have a median of 378 = C(28,2) gated pairs; most of those are fixed-point starts (cycle length 1) that the model holds. Dropping every fixed-point item leaves the arm ordering intact: Spearman with F1/F8 = 0.805. Equal item weighting gives an item with one chance-correct pair the same weight as an item with 378 pairs, so pooled pair rates (0.002–0.18) sit far below the item means.

**5. Agreement with a shared error occurs in trained models.** Gated pairs whose four-step futures agree while both are wrong number 246 in `lam0_s3`, 201 in `lam0_s6`, 181 in `lam0_s1` and 133 in `lam0_s7`. The oracle companion metric therefore matters on trained weights as well as on the untrained smoke.

**6. In the capped interventions, edge swaps carry the donor effect and cursor swaps do not.** This is exploratory: 16 pairs per checkpoint, giving 64 future steps per intervention kind. The figures below are the step disagreement of the patched future with the receiver's own future.

| arm | full donor | edge swap | cursor swap | matched random, edges |
|---|---:|---:|---:|---:|
| lam0_s5 | 0.14 | 0.14 | 0.03 | 0.08 |
| lam0.01_s4 | 0.12 | 0.12 | 0.00 | 0.02 |
| lam0_s6 | 0.06 | 0.06 | 0.02 | 0.00 |
| lam0_s3 | 0.06 | 0.08 | 0.02 | 0.00 |

Where two equal-cursor states have different futures, the difference sits in the edge-token states, which hold the graph memory, rather than in the cursor token. A norm-matched random edge perturbation moves the future less in each of these arms.

## Scope

These are measurements on fixed checkpoints and one graph seed (64 items, 32 loops). They describe only the own-head candidate quotient. All 24 models shaped the protocol, so these numbers carry no prospective weight. A prospective test needs fresh training seeds and a rule frozen before training. Finding 2 suggests that any such rule should read a trace extending past depth 16.

## Files

- `semantic_echo_q{4,5}/<run>/metrics.json`, `report.md`, `pairs.jsonl.gz`: per-checkpoint receipt, report and all pair records. Read `pairs.jsonl.gz` with gzip; [`semantic_echo_summary.py`](../../../research_gym/scripts/semantic_echo_summary.py) reads either form.
- `full_trace.npz` (34 MB per checkpoint) is gitignored. The run command regenerates it deterministically, and its SHA-256 is in each `metrics.json` receipt.
