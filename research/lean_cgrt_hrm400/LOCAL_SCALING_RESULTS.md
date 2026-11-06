# Local scaling results: campaign c1 (and c2, in progress)

Machine: MSI Stealth A18 AI+, RTX 5080 Laptop 16 GB, 31 GB RAM, Lean 4.19.0. Runs 2026-11-05 20:56 to 2026-11-06 09:52.

**Metric.** Lean-certified proof completion: the tactic script found by Lean-backed search is compiled as a standalone file against the dataset statement, with a per-declaration axiom audit ([`lcgrt/certify.py`](lcgrt/certify.py)). The budget is node expansions (Lean calls):
- `b1` = shortest proof + 1, the brief's "proof-length-plus-one";
- `b2` = 2× shortest.

**Receipts.**

| what | file |
|---|---|
| every number below (per-theorem rows, certificate source sha256, Lean version, checkpoint sha256) | [`results/c1/evals/<job>.json.gz`](results/c1/evals) |
| all tables, 482 paired comparisons | [`results/c1/tables.md`](results/c1/tables.md), [`results/c1/summary.json`](results/c1/summary.json) |
| identity hash, checkpoint sha256, throughput, peak VRAM, pauses per training run | [`results/c1/train_receipts.json`](results/c1/train_receipts.json) |
| data manifests | [`results/data/`](results/data) |

**Matched across all models.** The same 163,124 training states (lcgrt-v1), 3,000 steps at batch 256 (768k samples), and the same deterministic sample order. The final checkpoint is evaluated; nothing is selected on validation or test. LR-01 and the v1 smoke runs are pilot evidence and are not part of these results.

## 1. The original holdouts saturate; the mirror holdout does not

Certified success % at b1, best-first search, seed 17 unless noted. Columns 2–5 are the original holdouts, columns 6–7 the two calibration splits.

| model | test_id | test_depth (7–12) | test_family | test_dep (3-premise) | test_reason | **test_mirror** |
|---|---:|---:|---:|---:|---:|---:|
| random DFS | 23.7 | 4.7 | 21.0 | 29.7 | 6.0 | 4.4 |
| BFS | 7.0 | 0.0 | 6.0 | 7.0 | 0.0 | 0.3 |
| HRM-S | 100 | 100 | 100 | 93.3 | 99.7 | 50.8 |
| flat-S | 100 | 100 | 100 | 93.3 | 99.3 | 58.1 |
| HRM-M | 100 | 100 | 100 | 93.3 | 99.3 | 78.9 |
| flat-M | 100 | 100 | 100 | 93.3 | 99.3 | 46.4 |
| HRM-L | 100 | 100 | 100 | 93.3 | 99.3 | 29.2 |
| flat-L | 100 | 100 | 100 | 94.3 | 99.3 | 63.9 |

(The random-DFS and BFS rows are model-free; the HRM-S and flat-S `test_id` cells are from their main evaluation jobs.)

Every learned model solves the brief's depth holdout (proof depth 7–12, trained on 2–6), the held-out families and the unseen 3-premise dependencies at the tightest budget. They also solve `test_reason`, whose decoy chains are 6–12 steps long against 2–4 in training. A network with 8 propagation hops cannot follow a 12-step chain, so these splits are solved through **local structural cues**, not deep reasoning. That is recorded as a calibration finding in [`LEAN_DATA_PIPELINE.md`](LEAN_DATA_PIPELINE.md) and the commit log.

`test_mirror` removes the cues: each decoy copies the real derivation rule for rule, with dead leaves. It is the only split on which methods differ.

## 2. HRM vs flat: no reliable difference; seed variance dominates

`test_mirror`, best-first:

| model | seed 17 | seed 29 | seed 43 | mean ± sd (b1) | mean ± sd (b2) |
|---|---:|---:|---:|---:|---:|
| HRM-S (5.19M) | 50.8 | 26.9 | 61.1 | 46.3 ± 17.5 | 85.6 ± 10.6 |
| flat-S (5.36M) | 58.1 | 51.7 | 86.1 | 65.3 ± 18.3 | 87.1 ± 10.3 |

- Flat-S is nominally ahead at b1 by 19 points, and the two are level at b2. Three seeds per arm do not separate them: the seed sd is about 18 points.
- Paired theorem-bootstrap intervals are narrow but conditional on one seed. Within a seed, HRM−flat is −7.2 [−11.7, −2.8] at S, **+32.5** [+26.9, +37.8] at M and **−34.7** [−39.7, −29.7] at L.
- The sign flips with size, which is what seed noise of this magnitude produces.
- There is **no size trend**: HRM gives 46 (S, 3-seed mean), 79 (M) and 29 (L); flat gives 65, 46 and 64.
- HRM-S at T = 8 uses 3.8× the FLOPs of flat-S per decision. The FLOP-matched comparison, HRM-S vs flat-M (1.75 vs 1.35 GFLOP/decision), is +4.4 [+0.3, +8.6] for HRM-S at seed 17.

## 3. Recurrent depth: needed in training; extra test-time loops help only the single-clock loop

HRM-S trained at T fast steps (seed 17), certified success % at b1, best-first:

| trained T | test_depth | test_reason | test_mirror | val top-1 on choice states |
|---:|---:|---:|---:|---:|
| 1 | 18.6 | 5.3 | 8.9 | 49.8 |
| 2 | 81.1 | 78.7 | 35.6 | 97.3 |
| 4 | 95.0 | 94.7 | 31.1 | 99.4 |
| 8 | 100 | 99.7 | 50.8 | 99.9 |

On `test_reason`, T8 vs T1 is +94.3 [+91.7, +97.0] and T8 vs T4 is +5.0 [+2.7, +7.7]. Recurrent depth is what makes the shortcut-driven splits solvable at all.

The same weights run longer at test time (`test_mirror`, b1, best-first):

| model | T = 8 (trained) | T = 16 | T = 32 |
|---|---:|---:|---:|
| HRM-S (two clocks) | 50.8 | 41.1 (−9.7 [−13.1, −6.7]) | 39.4 (−11.4) |
| HRM-M | 78.9 | 80.6 (+1.7, n.s.) | 80.6 |
| HRM-L | 29.2 | 31.4 (+2.2 [+0.6, +4.2]) | 31.4 |
| **S-loop (single clock, tied, no slots)** | 48.6 | **60.6** (+11.9 [+7.5, +16.7]) | **65.0** (+16.4 [+11.7, +21.4]) |

The single-clock tied loop improves with every extra iteration on mirrors (b2: 80.3 → 92.5 → 93.9). The two-clock HRM does not. On `test_reason`, though, S-loop falls with more iterations (99.3 → 85.3 → 89.3). Test-time extrapolation is real for the plain loop on the shortcut-free split and absent for the two-clock design. One seed per configuration.

## 4. Other ablations (seed 17)

| ablation | result |
|---|---|
| **Proof-graph edges removed** (full attention, one edge type) | collapses: `test_mirror` 0.8%, `test_reason` 0.3%, `test_depth` 0.0% at b1 (HRM-S with graph: +50.0 [+44.7, +55.0] on mirror). Incidence structure is the signal |
| Two-clock HRM-S vs single-clock S-loop at T = 8 | +2.2 [−2.8, +7.2] on mirror: no difference at the trained depth |
| **Search history hidden** (dead-goal features zeroed) | no change for HRM-S or flat-S on any split: 2 of 360 mirror theorems change expansion count, none changes success. The control layer already discards children with known-dead goals, so the history features carry nothing extra. Compressing search history into a carried slow state therefore has nothing to compress here, and the carry-over experiment was not run |
| **Adaptive halting** (τ = 0.5) | halts at the first slow tick (mean 2.0 fast steps instead of 8, about 4× less compute), but loses accuracy: HRM-S mirror 29.4 vs 50.8, reason 79.3 vs 99.7 (b1). The halting head learned in-distribution confidence that does not transfer |

## 5. CGRT search: learned branch scores matter, cross-context branching does not help

Same Lean-call budget, `test_mirror`, certified success % b1 / b2, with model compute (GFLOP per theorem):

| checkpoint | best-first, 1 context | CGRT, 3 contexts, learned scorer | CGRT, shuffled branch scores |
|---|---|---|---|
| HRM-M | **78.9** / 93.3 (134) | 69.2 / 93.6 (563) | 2.8 / 24.7 (1,060) |
| HRM-S seed 29 | **26.9** / 73.6 (65) | 25.0 / 76.4 (256) | 3.3 / 24.7 (348) |
| HRM-S seed 43 | **61.1** / 93.9 (50) | 46.4 / 93.6 (218) | 3.6 / 26.4 (351) |

- Shuffling the branch scores among siblings, with the same contexts and budget, collapses search to 3% at b1, so the scores carry the search.
- Branching across contexts (recurrence depth T and 2T; failure history hidden) does **not** beat the single-context policy at matched Lean calls. It is worse at b1 in 3 of 3 runs, level at b2, and costs about 4× the model compute.
- The learned scorer had nothing to learn: on held-aside training theorems, learned weights, prior score and plain log-probability all reach AUC 1.000 (`results/c1/branch/`).
- CGRT runs for HRM-S seed 17 and HRM-L failed on a mistyped branch-weights path (`branch_S_t8_s17` for `branch_S_t8_s17b`). The paths are fixed and the runs are re-queued.

## 6. Measured resources (RTX 5080 Laptop, shared machine)

| run | params | steps/s (median) | peak VRAM MiB | pauses (thermal 85 °C / power cap / RAM) |
|---|---:|---:|---:|---:|
| HRM-S | 5,186,387 | 7.8 | 4,979 | 1–2 |
| flat-S | 5,358,682 | 10.6 | 3,749 | 1 |
| HRM-M | 15,849,875 | 3.0 | 9,777 | 5 |
| flat-M | 15,891,442 | 7.0 | 5,332 | 2 |
| HRM-L (grad ckpt) | 30,202,771 | 1.6 | 4,532 | 10 across two jobs |
| flat-L | 30,236,442 | 4.9 | 6,845 | 5 |

Batch 256 states, with a padded-token budget of 20,480. Under sustained load the GPU reaches 85–87 °C in about 5 minutes; the trainer then checkpoints, exits 75, and vramorama restarts it when the card is clear. That gives a duty cycle of about 55% for M/L and about 80% for S. Every resumed run continued from a hash-verified checkpoint; resume is bitwise identical (`tests/test_lcgrt.py`). Pauses across all c1 and c2 training jobs: 45 thermal, 7 for low host RAM (0.05–3 GB free while other agents' jobs ran), 3 for the GPU power cap.

## 7. Answer to the success criterion

> Does the hierarchical recurrent architecture plus learned CGRT search improve Lean-certified proof completion on unseen problems at a fixed compute budget?

**Not on this evidence.** At matched data and Lean-call budget:
- The two-clock HRM does not reliably beat a parameter-matched flat graph transformer. It is behind at S over three seeds (46 vs 65 at b1), level at b2, and has no consistent size trend.
- CGRT branching is no better than single-context best-first while costing 4× the compute.
- What did matter: the proof-graph incidence structure; training-time recurrent depth (T1 → T8); and, for the single-clock tied loop only, test-time iteration on shortcut-free problems.

## c2 (in progress at the stop for the RAM upgrade)

c2 asks whether training on mirror decoys at depths 2–5, with randomized recurrence depth (T ∈ [4, 12]), gives an HRM that extrapolates to mirror depths 6–9 better than flat. Dataset `lcgrt-mirror`: 13,000 theorems, all certified ([manifest](results/data/lcgrt-mirror_manifest.json)).

At the stop:
- `c2_S_t8_s17` was checkpointed at step 2,850/3,000;
- `c2_Srd_s17` was checkpointed at step 2,450/3,000;
- the other c2 jobs had not started.

Both checkpoints have verified sha256 sidecars and resume when the queue is relaunched. No c2 number is reported here.
