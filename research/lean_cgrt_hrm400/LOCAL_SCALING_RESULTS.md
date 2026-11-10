# Local scaling results: campaigns c1, c2 and c3

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

After the RAM upgrade (2026-11-06 17:25), c2 and Gate 0 resumed in two lanes: GPU training, and CPU evaluation with the model in RAM. By 2026-11-07 00:10, c2 training and Gate 0 seeds 17–61 had finished; their evaluations are queued on the CPU lane.

### c2 and Gate 0 result (2026-11-09)

**Gate 0 fails.**
- The rule ([handoff](SILICO_400M_HANDOFF.md#gates)): a recurrent arm must beat flat-S on `test_mirror_deep` at b1 by at least 10 points, with the seed-level Welch 95% CI excluding 0.
- Ten seeds per arm (17–113). Seeds 17 and 29 of flat-S and HRM-S-rd are the c2 runs.
- Best-first policy shown; greedy gives the same picture ([summary](results/c1/summary.json), key `gate0`).

| arm | mean b1 % | sd | worst seed | vs flat-S, points (95% CI) |
|---|---:|---:|---:|---:|
| flat-S | 99.5 | 0.3 | 99.2 | – |
| HRM-S rd, T = 8 | 99.8 | 0.1 | 99.8 | +0.3 (+0.1, +0.5) |
| HRM-S rd, T = 32 | 99.4 | 1.0 | 96.7 | −0.1 (−0.9, +0.6) |
| S-loop, T = 8 | 99.8 | 0.0 | 99.8 | +0.2 (+0.0, +0.4) |
| S-loop, T = 32 | 89.9 | 14.5 | 65.8 | −9.7 (−20.0, +0.7) |

**What this shows:**
- **The benchmark is saturated.** With mirror decoys at depths 2–5 in training (`lcgrt-mirror`), flat-S solves 99.5% of mirror depths 6–9. No arm can clear +10 points, so Gate 0 cannot pass on this data.
- **c2 gets the same answer.** The question was whether randomised-depth HRM extrapolates to deeper mirrors better than flat. Every arm extrapolates, and the gaps are under one point.
- **Test-time iteration at T = 32 breaks the tied loop on 3 of 10 seeds** (71, 83, 97: 70.4%, 65.8%, 71.1%). Seed 71 was evaluated on the CPU, seeds 83 and 97 on the GPU. HRM-S rd at T = 32 is at worst 96.7%.
- **The 400M decision stands.** The remaining local route to a 400M job is c3 on Mathlib.

## c3: Mathlib (registered before results)

**Question.** Does the c1 answer carry over to real Lean? With the same parameter budget, does a two-clock or single-clock recurrent encoder beat a flat encoder at completing unseen Mathlib goals? And does any learned model add to Lean's own automation at a fixed number of Lean calls?

**Data.** Snapshot `mathlib-snap1` of the full trace. The first training job cuts it from the files traced by then: the 1,184 alphabetical files plus a hashed-order sample of the rest. Its manifest records which files went in. Training uses only certified declarations, from the train split only.

**Runs.** `tlm-flat-S`, `tlm-hrm-S` and `tlm-loop-S` × seeds 17, 29 and 43, one GPU job at a time, after Gate 0 training ([architecture](HRM_CGRT_ARCHITECTURE.md#mathlib-tactic-model-c3)).

**Evaluation** ([`lcgrt/mathlib_eval.py`](lcgrt/mathlib_eval.py)):
- **States:** 200 terminal states from each held-out split (`test_file`, `test_long`, `test_family` = NumberTheory), at most 4 per file. The same states are used for every model; each file is elaborated once.
- **Completion:** at each state's own source position, a tactic completes the state when Lean accepts it, no goals remain, the kernel accepts the closed proof as a standalone theorem, and its axioms are within the standard three.
- **Metric:** close@k with k Lean calls, k ∈ {1, 2, 4, 8}.
- **Policies:**
  - `auto`: the 12 automation tactics, ordered by how often each closed a training state;
  - each model's beam;
  - each model interleaved with `auto`;
  - `human`: the source tactic, as a pipeline check.

**Decision rules** (fixed now; implemented in [`lcgrt/report_mathlib.py`](lcgrt/report_mathlib.py)):
- **Validity:** the source tactic must complete at least 95% of reached states, or the evaluation is reported as broken, not as a result.
- **H1 (architecture):** an encoder beats flat if the Welch 95% CI over the three seeds of its close@8 minus flat's close@8 lies above 0. Model-only proposals; all held-out states pooled.
- **H2 (learning adds to automation):** model+auto close@8 minus auto close@8 must have a paired-bootstrap 95% CI above 0, with success averaged over the arm's seeds for each state.
- **Bearing on the 400M decision:** if H1 fails for both recurrent encoders, real Lean agrees with c1, and the no-go in [SILICO_400M_HANDOFF.md](SILICO_400M_HANDOFF.md) stands. If H1 holds, that file's Gate 2 job is re-specified on this task in place of `lcgrt-mirror`.

### c3 amendment (2026-11-07, before any completion was measured)

The first attempt failed in two ways. Both are training-side; no close@k was ever computed.

1. **The snapshot was too small.** It was cut when the first job started (05:03), at 1,573 traced files, giving 43,583 usable training rows. `tlm_flatS_s17` then trained for 35 epochs and memorised them: training loss fell to 0.026 nats/byte. Its monitor, which was the `test_file` split, reached its minimum of 1.88 bits/byte at step 2,000 and climbed to 4.51 by step 12,000.
   - Disclosure: that monitor curve, on one arm, is the only test-split quantity anyone saw. It informed the step budget below.
2. **The recurrent encoders exceeded VRAM.** At batch 128, with T up to 12 and without activation checkpointing, they need more than the card: both run out of memory under a 12 GB cap. On Windows the CUDA allocator spills past VRAM into system RAM instead of failing. `tlm_hrmS_s17` therefore ran at 0.03 steps/s for 12 h, and vramorama held every other lease on the machine back while the spill lasted.

Changes:
- **Data:** `mathlib-c3`, assembled from the **complete** trace. The jobs wait for its manifest (`"requires"`).
- **New `val` split:** whole files with sha1(path) % 33 == 1, in-distribution. It replaces `test_file` as the training monitor.
- **Training:** 6,000 steps, about 4 epochs. Each run is evaluated at its best-on-`val` checkpoint (bits/byte, checked every 500 steps). That checkpoint is a function of `val` only; no test split is used.
- **Memory:** activation checkpointing for all arms, which is the same math. Measured peaks at T = 12: flat 0.85 GiB, HRM 3.6 GiB, loop 3.9 GiB. A CUDA memory cap of 0.55 of the card turns any future overflow into an immediate OOM.
- **Unchanged:** presets, seeds, evaluation, policies and the decision rules above.
- **Compute is matched by parameters, not FLOPs:** measured per step at batch 128, flat takes 0.11 s; HRM 1.35 s and loop 1.40 s at T = 12.

Runs: `tlm3_{flatS,hrmS,loopS}_s{17,29,43}`, then `evalM_tlm3_s17_29_43`. The superseded `tlm_*` jobs stay in the queue as `skip`. `tlm_flatS_s17` stays as a receipt and is not evaluated.

Status (2026-11-08):
- The full trace is done: 6,305 files.
- `mathlib-c3` is published, sha256-verified ([receipt](results/data/mathlib-c3_hf_upload.json)).
- `tlm3_flatS_s17` and `tlm3_hrmS_s17` are complete, with best-on-`val` scores of 1.436 and 1.473 bits/byte.
- The other 7 runs wait for the GPU, which is in use for other work.
- No close@k has been measured at scale.

Disclosure (2026-11-08):
- A smoke check of the CPU evaluation path, run under the 9 GiB cap, scored `tlm3_flatS_s17` on 6 `test_file` states from 3 files. These states belong to the c3 sample.
- That printout is the only test-split model number anyone has seen. It informs nothing: the c3 and c3-cpu rules were fixed before it.
- It did expose a harness defect, now fixed ([pipeline](LEAN_DATA_PIPELINE.md)). The source tactic failed 1 of the 6 states, because universe metavariables blocked certification. Uncorrected, this would have failed the validity rule.

### c3-cpu: the c3 question without a GPU (registered 2026-11-08, before any c3-cpu run)

**Why.** The GPU and most of the RAM are in use for other work, leaving 9 GB and no GPU. On this CPU, beside that work, flat-S takes 16.4 s per step at batch 128, against 0.11 s on the GPU. The 7 remaining c3 runs would take weeks. c3-cpu asks the same question at a scale the CPU can train. It is run by `lcgrt.cpugym` under a 9 GiB hard cap at Idle priority ([README](README.md)).

**Changes from c3:**
- **Model:** XS presets with about 1.48M parameters, matched within 1% ([architecture](HRM_CGRT_ARCHITECTURE.md#mathlib-tactic-model-c3)).
- **Training budget:** batch 32 and 3,000 steps, which is 96k examples, about 0.4 epoch.
- **Precision and memory:** fp32 on the CPU, without activation checkpointing.
- **Unchanged:** data `mathlib-c3`, splits, the `val` monitor and the best-on-`val` checkpoint, the lr schedule, seeds 17/29/43, randomised depth [4, 12] in training and T = 8 at test, the evaluation, the policies and the decision rules (validity, H1, H2).

**Measured before the runs.** Batch 32, T = 8, 4 threads, with the CPU 40–90% busy with the other work:

| arm | s/step | peak commit |
|---|---:|---:|
| HRM-XS | 3.3 | 3.2 GiB |
| loop-XS | 4.8 | 3.8 GiB |

Under the cap, two measurements set the plan:
- **Training:** loop-XS through the trainer at T = 12, including validation, peaked at 5.85 GiB.
- **Mathlib evaluation:** 2 Lean workers peaked at 5.83 GiB.

The Mathlib evaluation needs every c3-cpu checkpoint, so it never runs beside training. Training plus a Gate 0 evaluation (about 1.6 GiB) fits.

**Runs.** `tlmcpu_{flatXS,hrmXS,loopXS}_s{17,29,43}`, then `evalM_tlmcpu_s17_29_43`. That eval uses the same states as c3: the sampling is seeded and does not depend on the models.

**Reading.**
- c3-cpu is a smaller-scale replicate. It does not replace c3.
- Where the two disagree, c3 decides H1 for the 400M question, and c3-cpu is reported beside it.
- Results at XS are labelled as XS results.

### GPU share amendment (2026-11-09, before any c3-cpu or c3 evaluation)

The GPU became partly free on 2026-11-09, while the other work kept most of the CPU. From 06:20 the gym runs under `lcgrt.cpugym --ram-cap 9G --gpu-vram 9G --gpu-duty 0.6` ([README](README.md)). Nothing in a recipe, split, seed or decision rule changes; only devices do.

**c3-cpu.**
- Already done on the CPU: `flatXS` and `hrmXS` at seeds 17 and 29, and `loopXS_s17`.
- `loopXS_s29` ran steps 0–2000 on the CPU, then resumed from its step-2000 checkpoint on the GPU. Seed 43 runs on the GPU throughout.
- On the GPU these runs use fp32 without autocast or TF32, as on the CPU.
- Data order and the depth schedule are functions of (seed, step), so a device changes rounding only.
- Each training segment's device and precision are in the run's `train.log` start events.

**Gate 0 T32 evaluations.**
- The 8 still open run on the GPU. The finished ones ran on the CPU.
- Check before switching: `evalT32_g0_Srd_s43`, re-run on the GPU, matched the CPU result in every field except timing (b1/b2/b4, mean expansions, certified and submitted counts, every split and policy). It took 5.5 minutes, against 135 on the CPU.

**c3.** The 7 open runs train on the GPU with their registered recipe (bf16 autocast, activation checkpointing, batch 128). Each run's VRAM lease and allocator cap now match its measured peak at T = 12:

| run | VRAM in use | lease |
|---|---:|---:|
| flat-S | 3.3 GB | 5G |
| hrm-S | 7.0 GB | 8G |
| loop-S | 8.0 GB | 9G |
| loop-XS fp32 | 4.7 GB | 6G |
| Gate 0 evaluation, T = 32 | 0.7 GB | 3G |

**Speed.**
- At a 60% duty cap: loop-S 0.54 steps/s, loop-XS 3.6 steps/s.
- The card already sits near 80 °C from another job, so the thermal limit usually holds the duty at about 30%.
- **Order:** c3-cpu training first, then the Gate 0 evaluations, then c3. The Mathlib evaluations stay on the CPU.

### c3-cpu evaluation v1 is invalid; harness fixed, re-run registered (2026-11-09, before the c3 evaluation)

**v1 fails the validity rule.**
- `evalM_tlmcpu_s17_29_43` finished at 10:58. The source tactic completed 567 of 600 states (94.5%), under the 95% rule.
- So v1 is reported as broken, not as a result ([summary](results/c3cpu/v1_invalid/summary.json)).

**Causes.** Of the 33 failures:
- 19 are the per-tactic budget, which is kept;
- 5 are steps that rely on goals or instances outside the main goal, also kept;
- 9 are harness defects, now fixed: data goals and goals with open metavariables scored as if certifiable (7), explicit universe names out of scope (1), and the source tactic's layout collapsed (1).

Details: [pipeline](LEAN_DATA_PIPELINE.md). The causes were read from the source tactic's results alone.

**Disclosure.** v1's model numbers were printed before the cause was diagnosed:
- every model arm scored 21–26% close@8, against 52% for automation;
- no H1 or H2 held.

The fixes apply to every policy alike. Uncertifiable states leave every denominator; nothing else a policy sees changes.

**Re-run:** `evalM_tlmcpu_s17_29_43_v2`.
- Same checkpoints, states, policies, budgets and decision rules, on the fixed harness.
- The validity rule is unchanged: at least 95%, now over the scored (certifiable) states.
- The report lists every uncertifiable state.
- c3's `evalM_tlm3_s17_29_43` uses the same fixed harness. v2's result is in the next section.

### c3-cpu result (v2, 2026-11-10): no architecture effect, and no model adds to automation

**The evaluation is valid.** `evalM_tlmcpu_s17_29_43_v2` reached all 600 states.
- 12 are uncertifiable and listed in the [summary](results/c3cpu/summary.json); 588 are scored.
- The source tactic completes 564 of the 588 (95.9%), above the 95% rule.

**close@8 (%), held-out splits pooled.** XS encoders trained on the CPU; `+ auto` interleaves the model with automation inside the same 8 Lean calls.

| policy | s17 | s29 | s43 | mean |
|---|---:|---:|---:|---:|
| flat-XS | 25.0 | 23.6 | 25.0 | 24.5 |
| HRM-XS | 26.0 | 23.1 | 21.8 | 23.6 |
| loop-XS | 24.8 | 24.7 | 22.3 | 23.9 |
| flat-XS + auto | 51.2 | 51.4 | 51.0 | 51.2 |
| HRM-XS + auto | 51.0 | 51.0 | 51.0 | 51.0 |
| loop-XS + auto | 51.4 | 51.2 | 51.0 | 51.2 |
| auto | – | – | – | 52.2 |

Automation alone reaches 39.1 / 49.0 / 51.0 at close@1 / @2 / @4.

**H1 fails for both recurrent encoders** (Welch 95% CI over seeds):
- HRM-XS − flat-XS: −0.9 points (−5.7, +3.8);
- loop-XS − flat-XS: −0.6 points (−3.6, +2.3).

**H2 fails for every arm.** model + auto minus auto at close@8 (paired bootstrap):
- flat-XS: −1.0 (−2.0, −0.2);
- HRM-XS: −1.2 (−2.2, −0.3);
- loop-XS: −1.0 (−2.0, −0.2).

Interleaving hands the model 4 of the 8 calls. Automation's own tactics 5–8 add 1.2 points; the model's 4 tactics add 0.0–0.2.

**Exploratory (computed after the result, not registered).** The models close what automation already closes.
- Each run closes 128–153 scored states, of which 0–2 are outside the 307 that automation's 8 tactics close.
- Across all nine runs, the models close 176 distinct states, and 3 of them are outside automation's 307.

**Bearing.** At XS scale, real Lean agrees with c1: neither recurrent encoder beats flat. c3 (S scale, GPU-trained) decides H1 for the 400M question; its result is in the next section.

### c3 result (2026-11-10): no recurrent advantage on Mathlib; the 400M no-go stands

**The evaluation is valid.** `evalM_tlm3_s17_29_43` used the same 600 states and fixed harness as c3-cpu v2: 12 uncertifiable, 588 scored, and the source tactic completes 564 of the 588 (95.9%).

**close@8 (%), held-out splits pooled.** S encoders, GPU-trained at the 0.6 duty share.

| policy | s17 | s29 | s43 | mean |
|---|---:|---:|---:|---:|
| flat-S | 27.2 | 26.5 | 29.9 | 27.9 |
| HRM-S | 26.0 | 25.3 | 24.3 | 25.2 |
| loop-S | 27.0 | 26.7 | 27.4 | 27.0 |
| flat-S + auto | 54.2 | 53.9 | 54.2 | 54.1 |
| HRM-S + auto | 54.2 | 53.2 | 52.9 | 53.5 |
| loop-S + auto | 54.4 | 55.1 | 54.4 | 54.6 |
| auto | – | – | – | 52.2 |

At close@1: flat-S 21.0, HRM-S 19.4, loop-S 20.0, automation 39.1.

**H1 fails for both recurrent encoders** (Welch 95% CI over seeds):
- HRM-S − flat-S: −2.7 points (−6.4, +1.1);
- loop-S − flat-S: −0.9 points (−5.1, +3.4).

**H2 holds for flat-S and loop-S, not for HRM-S** (model + auto minus auto at close@8, paired bootstrap):
- flat-S: +1.9 (+0.4, +3.5);
- HRM-S: +1.3 (−0.2, +2.6);
- loop-S: +2.4 (+0.9, +4.1).

**Decision.** H1 failed for both recurrent encoders. By the registered rule, real Lean agrees with c1, and the no-go in [SILICO_400M_HANDOFF.md](SILICO_400M_HANDOFF.md) stands. Gate 2 is not re-specified.

**XS against S, on the same states.**
- The S models score 1.6–3.4 points above the XS models at close@8: flat 24.5 → 27.9, HRM 23.6 → 25.2, loop 23.9 → 27.0.
- H2 goes from failing for every arm at XS (−1.0 to −1.2) to holding for two arms at S.
- Flat scores highest at both sizes.

**Exploratory (computed after the result, not registered).** At S, the models close states that automation's first 8 tactics do not.
- Each run closes 143–176 scored states, 11–24 of them outside automation's 307. At XS it was 0–2.
- Across all nine runs, the models close 232 distinct states, 37 of them outside automation's 307. At XS it was 3.
- The tactics that close those states are ones automation does not reach within 8 calls: `omega` (its 10th), `infer_instance`, `ring`, `linarith`, `fun_prop`, and `simp only [...]` with lemmas.

## c4: a TRM arm on Mathlib (registered 2026-11-10, before any c4 run)

**Question.** Can a 4M Tiny Recursive Model stand in for scale on real Lean? Concretely: does an encoder trained with the TRM recipe, at 4.0M parameters in all, complete as many unseen Mathlib goals as c3's 5.6M flat encoder, and more than c3's single-clock loop?

**Why c3's loop arm is not a TRM.** loop-S iterates one block with input injection, at a random depth in [4, 12], with gradient through every iteration. TRM ([Jolicoeur-Martineau 2025, arXiv:2510.04871](https://arxiv.org/abs/2510.04871)) differs in four ways:
- separate answer and latent states, y and z, updated on a fixed schedule;
- deep supervision: up to 16 supervision steps per sample, with (y, z) carried and detached between steps, and the first T − 1 recursions of each step run without gradient;
- halting learned from whether the current answer is right;
- EMA weights are what gets evaluated.

**Arm: `tlm-trm-4M`** ([`lcgrt/tactic_lm.py`](lcgrt/tactic_lm.py), [architecture](HRM_CGRT_ARCHITECTURE.md#mathlib-tactic-model-c3)).
- **Network:** 2 post-norm transformer layers at width 256 with 4 heads (RMS norm without weights). The encoder has 1.74M parameters, 4.01M in all with c3's decoder (flat-S 5.59M, loop-S 5.67M).
- **Recursion:** n = 6 updates z ← net(x + y + z), then y ← net(y + z). T = 3 recursions make a supervision step; the first two run without gradient. The decoder reads y.
- **Training:** TRM's carry scheme, one supervision step per optimizer step. A slot that has halted takes a new sample.
  - Loss: tactic cross-entropy + 0.5 × BCE(halting logit, teacher-forced exact match).
  - A slot halts after 16 steps or when its halting logit is above 0. With probability 0.1 it may not halt before a random step in [2, 16].
  - EMA 0.999 of the weights. The EMA is what is validated, kept as best-on-val and evaluated.
- **Test:** all 16 supervision steps (672 layer applications per goal), then c3's beam search (k = 8).
- **Kept from c3:**
  - data `mathlib-c3` (192,767 training rows), the `val` monitor and the best-on-val checkpoint;
  - 6,000 optimizer steps at batch 128;
  - AdamW at lr 5e-4 (300 warmup steps, cosine to 10%), weight decay 0.05, clipping at 1.0;
  - bf16 on the GPU at the 0.6 duty share, activation checkpointing;
  - seeds 17, 29 and 43.
- **Not taken from TRM:** its puzzle-tuned optimizer (lr 1e-4, weight decay 0.1–1.0, batch 768), its data augmentation and its SwiGLU layers. c3's settings are kept so that the arms differ only in the encoder and how it is trained. TRM reads its answer off y position by position; a tactic needs a decoder, so c3's decoder cross-attends to y.
- **Budget, stated before any run.** At the same 6,000 updates, each TRM sample is trained for up to 16 supervision steps. So the TRM arm starts fewer distinct samples than the c3 arms' 768,000 (about 4 epochs). The count is logged (`fresh`) and reported.
- **Checks before registration** (no validation or test numbers were produced):
  - unit tests;
  - a 16-step GPU run: peak 1.6 GiB allocated, about 0.9 steps/s at the duty share, so the lease is 3G.
  - A tiny TRM collapsed to an input-independent encoder under plain Adam at 3e-3 without clipping. Under the trainer's recipe it learned (`test_trm_tiny_learns_and_beam_recovers`).

**Evaluation:** `evalM_tlm4_trm_s17_29_43`.
- c3's evaluation, unchanged: the same 600 states, harness, budgets and policies.
- The three TRM checkpoints are scored in the same job as c3's flat-S and loop-S checkpoints, which are re-scored.

**Decision rules** (fixed now; `python -m lcgrt.report_mathlib --pairs trm:loopS --replicate results/c3/summary.json`):
- **Validity:** as in c3.
- **Replication:** flat-S and loop-S should reproduce their c3 close@8 for every seed. Any difference is reported, and the c4 tests use this job's numbers.
- **C4-1 (primary), TRM vs flat-S:** the Welch 95% CI over the three seeds of the close@8 difference, model-only, all held-out states pooled.
  - CI above 0: TRM beats flat-S.
  - CI below 0: TRM is worse.
  - Otherwise: no difference detected, with the CI reported.
- **C4-2, TRM vs loop-S:** the same test.
- **C4-3:** TRM + auto vs auto, by c3's H2 rule.
- **Training failure:** a TRM run whose best val bits/byte is above 2.21 (loop-XS seed 17, the weakest model scored on Mathlib) is declared a training failure before the Mathlib evaluation. All three seeds are then rerun once at TRM's lr 1e-4, everything else unchanged, and both results are reported.
- **Bearing on the 400M decision:**
  - If C4-1 holds, a 4M recursive model beats a larger flat one on real Lean, which c1 to c3 never showed. [SILICO_400M_HANDOFF.md](SILICO_400M_HANDOFF.md) then records that any scale-up starts from the TRM recipe, with a local seed sweep at 4M and 15M before anything at 400M.
  - Otherwise the no-go stands, and TRM joins the recurrent designs that did not beat flat on real Lean at this scale.

**Disclosure.** c3's test numbers for flat-S and loop-S were known when this was written. The TRM settings are the paper's defaults (n = 6, T = 3, 16 steps, halting exploration 0.1, EMA 0.999), and nothing was tuned on `val` or test.
