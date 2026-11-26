# OLR-08B results: adaptive proof frontier allocation, BPD/MMS audit, and high-powered calibration headroom closure

2026-11-26. Branch `research/olr-obligation-ledger-20261123`, PR #318 (draft), extending pinned head `8931f9d9cf0b064cea5cb3ad0760fe2dcd5197fa`.
Pre-outcome registration: [OLR_08B_REGISTRATION.md](OLR_08B_REGISTRATION.md) (committed at `9612b6fa` before any fresh outcome collection).
Code and CPU launch package: [olr08b/](olr08b/) and [olr08b/RUN_08B.md](olr08b/RUN_08B.md).
OLR-08A's failed headroom verdict ([OLR_08A_RESULTS.md](OLR_08A_RESULTS.md), 18/24 menu-hard CAL roots, 5.6 pp vs 10 pp required) and all original thresholds are preserved unchanged.

## Executive verdict

1. **Gate 0 (Native validity) & Gate 1A (Calibration cohort feasibility): PASS.** The expanded uncontaminated `src/Init/` calibration campaign yielded **214 menu-hard calibration roots across all 26 independent file clusters** (`16` non-TRAIN families; `98.64%` CAL and `97.55%` TRAIN human-replay validity; `0` spent, `0` OLR-08A cal, `0` protected eval, `0` `SInt`, `0` TRAIN-family files), clearing the preregistered `≥ 120` roots and `≥ 12` file-cluster feasibility requirements.
2. **Gate 1B (Offline headroom `≥ 10.0 pp` over strongest 300-step fixed control, including portfolios): DECISIVE FAIL.** On all 214 menu-hard calibration roots, the strongest 300-step fixed control (`s2`, hand prior) closed **89/214 (41.59%)**, while the 1,200-step offline reference (`cal_s3_long`) closed **101/214 (47.20%)** — a measured headroom of **+5.61 pp** (95% 26-file cluster bootstrap CI **[+3.14 pp, +8.56 pp]**), replicating OLR-08A's `5.56 pp` to two significant digits and ruling out the `10.0 pp` threshold even at the upper 95% confidence bound. Even the hindsight 1,200-step union oracle (`104/214 = 48.60%`) reaches only **+7.01 pp** over `s2`.
3. **Fail-closed enforcement:** Per [OLR_08B_REGISTRATION.md](OLR_08B_REGISTRATION.md), **0 neural models were fitted under the preregistered calibration gate** (`models_fitted = 0`), the original **275-root evaluation registry remains untouched** (`274` pristine confirmation-eligible with smoke root `BitVec.shiftLeftZeroExtend`, id `45ce790b…`, explicitly excluded), **`SInt` remains strictly sealed (`0` files touched)**, **`architecture_promotion_eligible = false`**, and PR #318 stays in draft.
4. **Objective 1 root-cause diagnosis (spent cohorts):** Low closure and bounded `< 10 pp` step-budget headroom under the fixed depth-3 grammar arise primarily from **(a) premise retrieval** (Lean's top-24/6 index achieves only `28.0%` lemma recall; adding just 16 same-file recent declarations doubled `S3@300` closure from `3/24` to `6/24`, `+12.5 pp`, beating `S3@1200`'s `5/24` at 1/4 budget), **(b) action coverage** (`multi_lemma_simp_rw`, `<;>` combinators, `term_exact_refine`), **(c) depth-3 restriction** (`depth5_s2_300` newly solved `Array.finRange_succ`), and **(d) eager-expansion waste** (`58.6%`–`97.1%` of expanded states never popped, spending `22%`–`30%` of raw heartbeats on `premises`).
5. **Objectives 2 & 3 exploratory spent-cohort findings (`exploratory_spent_only = true`):** All 18 matched 3-seed neural models (`128,647` matched parameters; `recurrent`, `recurrent_bpd`, `direct_mlp`, `matched_compute_sham`, `one_way`, plus `2,075`-param `linear`) were trained on `66` replay-certified TRAIN frames and audited on the 18 spent OLR-08A CAL roots. All real trained arms tied `s2` and `portfolio_rr2` at `4/18` (`22.22%`, `std = 0.0`) with zero recurrence-specific advantage over `direct_mlp` or `linear`. **BPD component features did NOT beat RMS-matched Haar orthogonal rotation nulls** on branch utility (`0.6236` vs `0.6797` AUROC), tactic applicability (`0.6944` vs `0.7042`), or recovery after failure (`0.6677` vs `0.7310`). **MMS memory budgets (`0, 64, 256, 1024, 4096 MiB`) saturated at `< 90 KiB`** per explicit-memory episode (`< 390 KiB` for neural arms), yielding zero hit or closure scaling beyond `64 MiB` while maintaining `100%` SHA-256-verified immutable verifier receipt chains.

---

## 1. Preregistered gates (OLR-08B full calibration campaign)

| Gate | Criterion | Threshold | Measured | Result |
| --- | --- | --- | --- | --- |
| 0A | Native scope sentinel + 07A/07B custody | PASS | Sentinel PASS; 07A (`b614ffab…`) & 07B (`fc7bb50e…`) verified | **pass** |
| 0B | TRAIN `T`-human replay validity | ≥95%, ≥20 audits | 159/163 = 97.55% (187 scored, 106 hard) | **pass** |
| 0B | CAL `T`-human replay validity | ≥95%, ≥10 audits | 290/294 = 98.64% (389 scored, 214 hard) | **pass** |
| 1A | Menu-hard TRAIN roots | ≥40 | 106 | **pass** |
| 1A | Menu-hard CAL roots | ≥120 | **214** | **pass** |
| 1A | Independent CAL file clusters | ≥12 | **26** (across 16 non-TRAIN families) | **pass** |
| 1B | Strongest 300-step fixed control on CAL | ≤80% | `s2` 89/214 = 41.59% (strongest portfolio `portfolio_rr2` 85/214 = 39.72%) | **pass** |
| 1B | `cal_s3_long` (1200) minus strongest 300-step control | ≥10.0 pp | 101/214 − 89/214 = **+5.61 pp** (95% CI `[+3.14, +8.56]` pp) | **fail** |
| 2 | Preregistered neural reader fitting | Gate 1 pass & ≥64 frames | Not triggered (`models_fitted = 0`) | — |
| Seal | Protected 275-root eval / `SInt` / 500K–5M promotion | Protected / sealed / blocked | `protected_eval_opened = false` (274 pristine); `sealed_SInt_opened = false`; `promotion = false` | **preserved** |

Cross-campaign determinism on reused TRAIN (`106` menu-hard roots): `source_s2` closed **33/106** (identical to OLR-08A), and the fitted `prior.json` matched OLR-08A's `weight` and `rank` tables **bit-for-bit** (`prior_weight_exact_match = true`, `prior_rank_exact_match = true`).

---

## 2. Objective 1: Spent-cohort search-limitation diagnostics

Source artifact: [results/olr08b/obj1_diagnostics.json](results/olr08b/obj1_diagnostics.json). Strictly restricted to spent cohorts (`18` OLR-08A spent CAL hard roots + `12` deterministic OLR-08A spent TRAIN hard roots = `30` roots across `14` spent files, plus offline audit of all `482` valid hard roots in the authenticated referee archive `23225d1d…`).

### 2.1 Paired search headroom across budgets (75, 150, 300, 600, 1,200 steps)

| Arm / Control (`n = 30` spent roots, `DEPTH = 3`) | @75 | @150 | @300 | @600 | @1200 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `s1` (uniform) | 2 (6.7%) | 3 (10.0%) | 3 (10.0%) | 5 (16.7%) | 5 (16.7%) |
| `s2` (hand prior) | 3 (10.0%) | 4 (13.3%) | **7 (23.3%)** | 7 (23.3%) | **8 (26.7%)** |
| `s3` (counted prior) | 4 (13.3%) | 5 (16.7%) | 5 (16.7%) | 6 (20.0%) | 7 (23.3%) |
| `s3c` (cost-aware prior) | 4 (13.3%) | 4 (13.3%) | 4 (13.3%) | 5 (16.7%) | 5 (16.7%) |
| `portfolio_rr_s2_s3` (fixed portfolio) | 4 (13.3%) | 4 (13.3%) | 6 (20.0%) | 7 (23.3%) | 8 (26.7%) |
| `portfolio_rr_all4` (fixed portfolio) | 3 (10.0%) | 4 (13.3%) | 4 (13.3%) | 6 (20.0%) | 7 (23.3%) |
| `oracle_best_arm_at_budget` (retrospective oracle) | 4 (13.3%) | 6 (20.0%) | 7 (23.3%) | 8 (26.7%) | **9 (30.0%)** |

On the `482` valid hard roots of the authenticated referee archive (`92` CAL / `390` EVAL at 300 steps): `S3` (`38.0%` CAL, `59.7%` EVAL) and `S2` (`34.8%` CAL, `56.7%` EVAL) beat equal-split `RR(S1,S2,S3,S3c)` (`28.3%` CAL, `48.5%` EVAL) because equal 4-way budget splitting starves the stronger priors; the retrospective 4-arm oracle@300 adds only `+2.2 pp` (`40.2%` CAL) and `+2.1 pp` (`61.8%` EVAL) above `S3`.

### 2.2 Controlled mechanism probes (`n = 24` spent roots: 18 OLR-08A CAL + 6 unsolved TRAIN)

| Hypothesis / Mechanism Probe | Baseline Closure | Probe Closure | Delta | Newly Solved Theorems / Root Cause |
| --- | ---: | ---: | ---: | --- |
| **Premise retrieval:** add last-16 same-file decls (`recency16_s3_300` vs `s3@300`) | 3/24 (12.5%) | **6/24 (25.0%)** | **+12.5 pp** | `Int.OfNat.Expr.denoteAsInt_nonneg`, `Int.OfNat.Expr.le`, `List.countP_eq_length` (beats `s3@1200`'s 5/24 at 1/4 steps!) |
| **Depth-3 restriction:** `DEPTH=5` vs `DEPTH=3` (`depth5_s2_300` vs `s2@300`) | 4/24 (16.7%) | **5/24 (20.8%)** | **+4.2 pp** | `Array.finRange_succ` (`split <;> · simp; omega`, requires 5 single-goal steps) |
| **Depth-3 under breadth prior:** `depth5_s3_600` vs `s3@600` | 4/24 (16.7%) | 4/24 (16.7%) | 0.0 pp | Breadth explosion in `S3` prevents deeper expansion from paying off |
| **Eager vs lazy expansion:** `lazy_s3_300` vs eager `s3` | 3/24 (12.5%) | 3/24 (12.5%) | 0.0 pp | Cuts `premises` hb share from `23.4%–29.7%` down to `18.8%` |

### 2.3 Quantified bottleneck attribution (action coverage, premise recall, verifier waste, censoring)

- **Premise retrieval recall (`n = 13` library-lemma-naming roots in spent 30):** Lean's top-24/6 index (`in24`) achieves `28.0%` lemma recall (`3/13` all-in); last-16 same-file declarations alone achieve `40.0%`; union `in24 ∪ recent16` lifts recall to **`52.0%`** (approaching `in200 = 56.0%` without 200-candidate blowup). Across the 344 referee `premise_recall.json` roots, `in24` lemma recall is `27.3%` (CAL) / `20.2%` (EVAL).
- **Action coverage (referee EVAL `n = 390` / spent 30):** `single_lemma_simp_rw` closes at `82.7%` (`105/127`) / `60.0%` (`3/5`), while `multi_lemma_simp_rw` drops to `54.7%` / `22.2%`, `multi_step_seq` to `38.5%` / `16.7%`, and `term_exact_refine` to `16.7%` / `0.0%`.
- **Verifier cost & eager expansion waste (spent 30 @ 1,200 steps):** Eager expansion called `premises` on states that were **never popped** at rates of **97.1%** (`s1`: `15,353/15,812`), **84.4%** (`s3`: `9,417/11,158`), **83.8%** (`s3c`: `9,457/11,287`), and **58.6%** (`s2`: `4,517/7,711`), consuming `22.3%–29.7%` of all raw heartbeats (`0.04%–0.08%` on `cert`, `70.3%–77.6%` on `step`).
- **Censoring vs proven unreachable states:** On the 30 spent roots @ 1,200 steps, **100% of unsolved roots** (`25/25` `s1`, `22/22` `s2`, `23/23` `s3`, `25/25` `s3c`) hit `censored_steps` (`0` RPC/HB cap failures, `0` finite-space exhaustions).

---

## 3. Objective 4: OLR-08B full calibration campaign (214 menu-hard roots, 26 file clusters)

Source artifacts: [results/olr08b/full/summary.json](results/olr08b/full/summary.json), [results/olr08b/cal214_breakdown.json](results/olr08b/cal214_breakdown.json).

### 3.1 Main arm table (`n = 214` menu-hard calibration roots across `26` files)

| Arm | Budget | @75 | @150 | @300 / Final | Rate | vs Best Fixed (`s2`) & 95% File CI | Fates (`solved / cens_steps / exhausted`) | Total Raw HB | Native s | Open s | Certified / Chain |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |
| `s1` (uniform) | 300 | 37 | 47 | 56 | 26.17% | −15.42 pp `[-22.12, -9.25]` | 56 / 157 / 1 | 4,413,608,910 | 512.3 | 357.0 | 56/56 · valid |
| **`s2` (hand prior)** | 300 | 52 | 74 | **89** | **41.59%** | baseline (best fixed) | 89 / 124 / 1 | **3,509,147,504** | 545.2 | 543.2 | 89/89 · valid |
| `s3` (counted prior) | 300 | 49 | 61 | 87 | 40.65% | −0.93 pp `[-5.63, +3.33]` | 87 / 126 / 1 | 4,772,006,707 | 641.8 | 492.2 | 87/87 · valid |
| `s3c` (cost-aware) | 300 | 39 | 56 | 67 | 31.31% | −10.28 pp `[-16.85, -4.31]` | 67 / 146 / 1 | 4,055,076,133 | 546.4 | 432.0 | 67/67 · valid |
| `portfolio_rr2` (`s2,s3`) | 300 | 42 | 67 | 85 | 39.72% | −1.87 pp `[-5.38, +1.57]` | 85 / 129 / 0 | 3,755,026,274 | 516.4 | 493.2 | 85/85 · valid |
| `portfolio_rr4` (`all 4`) | 300 | 24 | 45 | 67 | 31.31% | −10.28 pp `[-17.62, -4.23]` | 67 / 147 / 0 | 3,979,832,310 | 739.0 | 662.5 | 67/67 · valid |
| `adaptive_ucb` (online) | 300 | 33 | 48 | 66 | 30.84% | −10.75 pp `[-17.65, -4.60]` | 66 / 148 / 0 | 4,210,869,932 | 653.2 | 516.3 | 66/66 · valid |
| `explicit_memory_64m` | 300 | 36 | 50 | 62 | 28.97% | −12.62 pp `[-22.51, -4.60]` | 62 / 152 / 0 | 4,708,309,776 | 620.8 | 463.5 | 62/62 · valid |
| **`cal_s3_long` (offline ref)** | 1200 | 49 | 61 | **101** | **47.20%** | **+5.61 pp `[+3.14, +8.56]`** | **101 / 109 / 4** | 14,552,100,252 | 2230.4 | 513.4 | 101/101 · valid |
| Retrospective oracle@300 | 300 | — | — | 100 | 46.73% | +5.14 pp (hindsight) | — | — | — | — | — |
| Retrospective union@1200 | 1200 | — | — | 104 | 48.60% | +7.01 pp (hindsight) | — | — | — | — | — |

**Proven unreachable states within the finite depth-3 grammar (`4` roots in `cal_s3_long`):**
- `map_congr` (`src/Init/Control/Lawful/Basic.lean`, exhausted at 224 steps; human: `simp [funext h]`)
- `Lean.Order.admissible_pprod_snd` (`src/Init/Internal/Order/Basic.lean`, exhausted at 748 steps; human: `apply h ⟨x,y⟩ hxy`)
- `PSigma.revLexAccessible` (`src/Init/WF.lean`, exhausted at 917 steps; human: nested `induction`)
- `emptyWf` (`src/Init/WF.lean`, exhausted at 1,195 steps; human: `apply WellFounded.intro`)

All remaining `109` unsolved roots in `cal_s3_long` hit `censored_steps` (censored search failures, not proofs of unreachability).

### 3.2 By-family breakdown (`16` non-TRAIN families, `26` files, `214` hard roots)

| Family | Files | Hard Roots | `s1` | `s2` | `s3` | `s3c` | `portfolio_rr2` | `adaptive_ucb` | `explicit_mem` | `s3_long@1200` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `AC` | 1 | 10 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| `Array` | 2 | 19 | 5 | 7 | 7 | 6 | 6 | 6 | 5 | 9 |
| `Bool` | 1 | 4 | 1 | 3 | 1 | 0 | 3 | 0 | 0 | 3 |
| `Init.Control` | 2 | 16 | 5 | 10 | 10 | 8 | 10 | 6 | 9 | 10 |
| `Init.GetElem` | 1 | 9 | 1 | 4 | 3 | 2 | 3 | 3 | 1 | 5 |
| `Init.Grind` | 3 | 8 | 8 | 7 | 8 | 7 | 7 | 7 | 1 | 8 |
| `Init.Internal` | 2 | 25 | 3 | 5 | 7 | 7 | 7 | 7 | 7 | 7 |
| `Init.Omega` | 4 | 40 | 7 | 13 | 14 | 11 | 13 | 12 | 13 | 15 |
| `Init.PropLemmas` | 1 | 14 | 4 | 8 | 5 | 4 | 8 | 4 | 4 | 8 |
| `Init.SimpLemmas` | 1 | 8 | 2 | 6 | 5 | 2 | 4 | 2 | 2 | 7 |
| `Init.SizeOfLemmas` | 1 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 |
| `Init.WF` | 1 | 7 | 1 | 2 | 2 | 1 | 2 | 1 | 2 | 2 |
| `Int` | 3 | 27 | 7 | 11 | 12 | 7 | 9 | 6 | 6 | 14 |
| `Ord` | 1 | 10 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| `Range` | 1 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `Vector` | 1 | 2 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 1 |
| **Total** | **26** | **214** | **56** | **89** | **87** | **67** | **85** | **66** | **62** | **101** |

---

## 4. Objectives 2 & 3: Utility predictor controls, BPD decomposition, and MMS scaling

Because Gate 1B failed on the fresh 214-root calibration cohort (`models_fitted = 0` under the preregistered gate), neural fitting, BPD predictiveness, and full MMS eviction-stress tests were executed exclusively on **spent OLR-08A cohorts** (`exploratory_spent_only = true`, [results/olr08b/spent_bpd_mms_report.json](results/olr08b/spent_bpd_mms_report.json)), while the configurable `0, 64, 256, 1024, 4096 MiB` MMS sweep for `explicit_memory` was measured on **both** the 24-root OLR-08B CAL slice and the 18-root spent cohort.

### 4.1 Three-seed neural controls (`128,647` matched parameters; seeds `17, 29, 43`; `66` TRAIN frames -> `18` spent CAL roots)

Frame extraction replay-certified `32/33` solved TRAIN paths (`66` frames; `1` path, `List.eraseIdx_ne_nil_iff` depth-2 `omega`, hit the 200k-hb boundary after graph queries and was excluded rather than relaxed). Across every seed, all five matched neural arms (`recurrent`, `recurrent_bpd`, `direct_mlp`, `matched_compute_sham`, `one_way`) verified identical `(parameters = 128,647, initial_sha256, sampler_sha256)`.

| Arm (Spent-Cohort Exploratory Only) | Params | Seeds `(17, 29, 43)` Solved / 18 | Mean Rate ± Std | Paired vs `s2` (`4/18`) | Paired vs `s3` (`3/18`) | Mean Feature s | Mean Inference s | Mean Total HB |
| --- | ---: | --- | ---: | --- | --- | ---: | ---: | ---: |
| `s2` / `portfolio_rr2` (fixed controls) | 0 | 4 / 18 | 22.22% | baseline | +5.56 pp | 0.0 | 0.0 | 415.8M / 452.8M |
| `s3` / `s1` / `s3c` / `adaptive_ucb` / `explicit_mem` | 0 | 3 / 18 | 16.67% | −5.56 pp | 0.00 pp | 0.0 | 0.0 | 413.9M–541.4M |
| `recurrent` (3-visit bidir + frontier $z_t$) | 128,647 | 4, 4, 4 | 22.22% ± 0.00% | 0.00 pp `[0, 0]` | +5.56 pp `[0, +33.3]` | 37.2 | 10.9 | 467.7M |
| `recurrent_bpd` (`recurrent` + learned BPD) | 128,647 | 4, 4, 4 | 22.22% ± 0.00% | 0.00 pp `[0, 0]` | +5.56 pp `[0, +33.3]` | 40.5 | 17.2 | 473.0M |
| `direct_mlp` (0 graph msg, 0 recurrence) | 128,647 | 4, 4, 4 | 22.22% ± 0.00% | 0.00 pp `[0, 0]` | +5.56 pp `[0, +33.3]` | 37.5 | 12.6 | 455.3M |
| `one_way` (reverse messages masked) | 128,647 | 4, 4, 4 | 22.22% ± 0.00% | 0.00 pp `[0, 0]` | +5.56 pp `[0, +33.3]` | 36.4 | 11.2 | 462.0M |
| `linear` (bag + cond linear) | 2,075 | 4, 4, 4 | 22.22% ± 0.00% | 0.00 pp `[0, 0]` | +5.56 pp `[0, +33.3]` | 37.5 | 0.48 | 500.1M |
| `matched_compute_sham` (shuffled targets) | 128,647 | 3, 3, 4 | 18.52% ± 2.62% | −3.70 pp | +1.85 pp | 47.4 | 15.1 | 430.4M |

**Verdict on neural architecture specificity:** Zero closure separation between `recurrent`, `recurrent_bpd`, `direct_mlp`, `one_way`, and `linear` (`4/18` on all 15 checkpoints). Adaptive routing recovers `s2`'s closure relative to `s3` and `matched_compute_sham`, at the cost of `37–41 s` Python feature extraction + `0.5–17 s` inference overhead and `~10–14%` higher raw heartbeats than `s2`.

### 4.2 Optional BPD decomposition predictiveness vs RMS-matched Haar rotation null

Rank-8 BPD decomposition was fitted on `recurrent_bpd` tied recurrent sites (`up`, `down` tagged `learned_bpd` with reconstruction cosines `0.666–0.760`; `cell_out`, `frontier_cell` tagged `svd_seed_proxy` with cosines `0.630–0.720`) and evaluated out-of-sample from `1,173` spent TRAIN events (`520` failure-recovery events) to `2,223` spent CAL events (`823` failure-recovery events with `fail_streak >= 2`):

| Target (`n_train -> n_eval`, `pos_eval`) | `bpd_components` AUROC (95% CI) | `haar_null_components` AUROC (95% CI) | `prior_plus_receipts` AUROC (95% CI) | `bpd_plus_prior_receipts` AUROC (95% CI) | BPD > Haar Null? |
| --- | --- | --- | --- | --- | --- |
| **Branch utility** (`on_path`, 7/2,223) | 0.6236 `[0.409, 0.776]` | **0.6797** `[0.475, 0.789]` | 0.7824 `[0.572, 0.947]` | **0.8183** `[0.652, 0.944]` | **No** (`−0.0561`) |
| **Tactic applicability** (`ok`, 653/2,223) | 0.6944 `[0.662, 0.715]` | **0.7042** `[0.670, 0.751]` | 0.6478 `[0.494, 0.701]` | 0.6901 `[0.561, 0.737]` | **No** (`−0.0098`) |
| **Recovery after failure** (`streak ≥ 2`, 238/823) | 0.6677 `[0.650, 0.697]` | **0.7310** `[0.704, 0.748]` | 0.6855 `[0.508, 0.758]` | 0.6780 `[0.554, 0.731]` | **No** (`−0.0633`) |

**Verdict on BPD predictiveness:** Learned proof-state representations carry genuine predictive information for tactic applicability (beating prior+receipts alone), **but BPD component alignment does not outperform an RMS-matched random orthogonal Haar rotation of equal rank** on any of the three targets. Preserving this negative specificity test prevents overclaiming mechanistic component utility.

### 4.3 Configurable MMS memory budgets (`0, 64, 256, 1024, 4096 MiB` + `0.02 MiB` eviction stress)

| Cohort & Budget | Solved | Peak Allocated (B) | Hits | Misses | Evictions | Cache Time (s) | Total Raw HB | Immutable Receipt Chain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CAL-24 · `0 MiB` | 8/24 | 0 | 0 | 74,381 | 0 | 0.0474 | 538,929,169 | 24/24 valid |
| CAL-24 · `64 MiB` | 8/24 | 89,327 | 8,943 | 64,845 | 0 | 0.2299 | 534,295,958 | 24/24 valid |
| CAL-24 · `256 MiB` | 8/24 | 89,327 | 8,943 | 64,845 | 0 | 0.2265 | 534,295,958 | 24/24 valid |
| CAL-24 · `1024 MiB` | 8/24 | 89,327 | 8,943 | 64,845 | 0 | 0.2352 | 534,295,972 | 24/24 valid |
| CAL-24 · `4096 MiB` | 8/24 | 89,327 | 8,943 | 64,845 | 0 | 0.3536 | 534,295,950 | 24/24 valid |
| Spent-18 · `0 MiB` | 3/18 | 0 | 0 | 70,378 | 0 | 0.0453 | 523,095,213 | 18/18 valid |
| Spent-18 · `0.02 MiB` (stress) | 3/18 | 20,972 | 6,280 | 64,608 | **3,705** | 0.6501 | 523,004,887 | 18/18 valid |
| Spent-18 · `64–4096 MiB` | 3/18 | 87,896 | 6,280 | 64,608 | 0 | 0.188–0.293 | 523,004,791–870 | 18/18 valid |

**Verdict on MMS scaling:** Mirroring DataGoblin's `MMS-01–05` saturation finding, hard byte caps enforce exact eviction under tight budgets (`0.02 MiB` caps peak at `20,972 B` with `3,705` evictions), while realistic per-episode Lean proof caches saturate at `< 0.1 MiB` (`< 0.4 MiB` with cached lemma graphs in neural arms). All budgets `64–4096 MiB` therefore yield identical hit counts and zero closure scaling, with lazy allocation keeping process commit strictly bounded.

---

## 5. Objective 5: Infrastructure audit, resource telemetry, and artifacts

- **Host & resource contract (`results/olr08b/full/resource.json`):** Windows 11 (`10.0.26200`), Python `3.12.10`, `torch==2.5.1+cpu`, `CPU-only`, **`8` threads, `2` native workers**, enforced **`12.0 GiB` Windows Job Object hard process-tree commit limit**.
- **Full OLR-08B calibration telemetry:** Peak Job commit **`3,235,307,520 B` (`3.01 GiB`, 25.1% of 12 GiB cap)**; peak sampled process-tree RSS **`2,892,800,000 B` (`2.69 GiB`)** across peak `5` processes; wall time **`8,335.9 s`** (shared host); OS-accounted Job CPU time **`16,359.7 s`** (`272.7 CPU-min`).
- **Regression tests:** `16/16` passing (`7/7` `olr08b/test_olr08b.py`, `4/4` `olr08/test_gym.py`, `3/3` `tests/test_olr07_reissue_custody.py`, `2/2` `tests/test_olr08_spent_cohorts.py`).
- **Authenticated archives (100% SHA-256 verified against `audit_manifest.json` `e0cc4bac…`):**
  - Full audit archive: [results/olr08b/olr08b-full_AUDIT.zip](results/olr08b/olr08b-full_AUDIT.zip) (`34,685,531 B` / `33.08 MiB`, `5,586/5,586` entries verified, SHA-256 `e7f9045366ac39faaabf02d28efac2086783f65823ceb4f69e7061dbda1aebd9`).
  - Compact referee archive: [results/olr08b/olr08b-full_FABLE.zip](results/olr08b/olr08b-full_FABLE.zip) (`770,864 B` / `753 KiB`, `52/52` entries verified, SHA-256 `30e178ee11af24ca168820fcf612b65fb392f3db639b00fca18644650ac8a91f`).
  - Commit-pinned canonical URLs & hashes: [results/olr08b/PINNED.md](results/olr08b/PINNED.md).

## 6. Scientific conclusion and next-rung implication

Scaling calibration 11.9× from 18 to **214 uncontaminated menu-hard roots across 26 independent file clusters** establishes with tight cluster confidence intervals (`+5.61 pp`, 95% CI `[+3.14, +8.56] pp`) that **the OLR-08 depth-3 fixed-enumerator action space lacks 10 pp of pure search-ordering headroom** — even under 4× step budgets or hindsight multi-arm oracles (`+7.01 pp`). Objective 1 shows the concrete engineering lever that DOES unlock `> 10 pp` headroom at fixed budget: **same-file recency premise retrieval (`+12.5 pp` at 300 steps), multi-lemma/combinator action coverage, depth `> 3` expansion, and lazy premise enumeration**. No 500K or 5M model promotion is warranted until the action/premise enumerator itself passes a newly registered `≥ 10 pp` headroom gate; the protected 275-root evaluation cohort (`274` pristine) and sealed `SInt` cohort remain intact for that next rung.
