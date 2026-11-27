# OLR-08B pre-outcome registration and experimental contract

2026-11-26. Branch `research/olr-obligation-ledger-20261123`, PR #318 (draft), extending pinned head `8931f9d9cf0b064cea5cb3ad0760fe2dcd5197fa`.
Code: [olr08b/run08b.py](olr08b/run08b.py), [olr08b/allocator.py](olr08b/allocator.py), [olr08b/models.py](olr08b/models.py), [olr08b/bpd_frontier.py](olr08b/bpd_frontier.py), [olr08b/mms_cache.py](olr08b/mms_cache.py), [olr08b/roster.py](olr08b/roster.py), [olr08b/launch08b.py](olr08b/launch08b.py).

Written and registered **before** collecting any fresh OLR-08B calibration menu, human-replay, or search outcomes. OLR-08A's failed headroom gate (`OLR_08A_RESULTS.md`, 18/24 menu-hard CAL roots, 5.6 pp vs 10 pp required) and all original OLR-08A thresholds are preserved unchanged.

## 1. Prerequisites and preserved infrastructure invariants

1. **Pinned Lean source:** `leanprover/lean4` `v4.19.0`, commit `6caaee842e9495688c1567e78c0e68dbb96942aa`, clean tracked tree.
2. **Native binary:** standalone `lgym.exe` built from `olr08/lean/` (`SHA-256 61923c657261feb458c60c7049f83cc0f2c121ce8dd3dcad09b039f8d3c5dede`).
3. **Corrected 07A/07B reissues:** custody-authenticated manifests in `results/olr07_reissue/` (`07A` manifest `b614ffab…`, 207 selected; `07B` manifest `fc7bb50e…`, 1,441 selected).
4. **Scope & certification checks:** `scope_version = pre_command_session_v1`, `env = before`, `decl_in_env = false`, transitive auxiliary-recursion rejection, native scope sentinel pass before any cohort scoring.
5. **Preserved repairs (`olr08/REPAIRS.md`):**
   - Repair 0001: uncharged, logged `exact?` library-search cache warmup (`hb = 20000`) before replaying an `exact?` step under the 500 cap in a fresh pre-command process.
   - Repair 0002: premise-enumerator heartbeat timeout (`ok = false`, heartbeat message, no `rw`/`apply`) treated as empty premise proposals while staying charged in the meter RPC journal and immutable receipt ledger.
6. **Immutable verifier receipts:** every native RPC (`open`, `premises`, `step`, `cert`, `graph`, `lemma_graph`) is appended to a SHA-256-chained `ImmutableReceiptLedger`; caches never alter, fabricate, or bypass native certification.
7. **CPU-first resource contract:** `threads = 8`, `workers = 2`, `CUDA_VISIBLE_DEVICES = ""`, enforced `12 GiB` Windows Job Object hard commit cap (or Linux cgroup `memory.max = 12G`), OS-accounted Job CPU seconds + peak job commit + peak sampled process-tree RSS logged in `resource.json`.

## 2. Frozen pre-outcome cohort selection and protection rules

### Exclusions (frozen before outcomes)
- **Spent files:** all 69 files in `results/olr08_spent/spent_cohorts.json` (07A, 07B, referee pilot) + all 5 calibration files spent in OLR-08A (`Int/Cooper`, `Int/OfNat`, `Array/BinSearch`, `Array/FinRange`, `Fin/Iterate`).
- **Protected evaluation registry:** all 27 files declared in OLR-08A `FRESH['evaluation']` protecting the original **275-root evaluation registry** (`results/olr08a/preoutcome/registry.json`, SHA-256 `d61bf979…`). The single smoke-observed evaluation root `BitVec.shiftLeftZeroExtend` (`id = 45ce790b9a0f67f50253ba1b41c778473b4b7631d3a9eef91f6ff0e5642e9a1f`) is explicitly flagged `pristine_confirmation_eligible = false`, leaving **274 pristine evaluation roots**. OLR-08B calibration stops before opening this evaluation cohort.
- **Sealed family:** `SInt` (`src/Init/Data/SInt.lean` and `src/Init/Data/SInt/*.lean`) remains strictly sealed (`0` files touched).
- **Family-disjoint calibration:** all `TRAIN` families (`Nat`, `List`, `Option`, `Char`, `String`) are excluded from calibration file selection.
- **Spent goals & declarations:** all spent exact-goal SHA-256 hashes and declaration names from 07A, 07B, referee, and OLR-08A TRAIN/CAL are excluded from fresh root selection.

### Deterministic calibration file selection rule (`olr08b/roster.py`)
Select every Lean file under `src/Init/` (both `src/Init/Data/` and `src/Init/` core libraries) that:
1. is not in the exclusion set above,
2. is not in `TRAIN_FAMILIES` or `SInt`,
3. is not a foundational macro/prelude definition file (`src/Init/Core.lean`, `src/Init/Tactics.lean`), and
4. contains at least `MIN_BY_BLOCKS_RULE = 14` syntactic tactic-proof blocks (`:= by` or ` by\n`).

This rule uniquely and deterministically selects **26 uncontaminated Lean files across 16 families** (`AC`, `Array`, `Bool`, `Int`, `Ord`, `Range`, `Vector`, `Init.Control`, `Init.GetElem`, `Init.Grind`, `Init.Internal`, `Init.Omega`, `Init.PropLemmas`, `Init.SimpLemmas`, `Init.SizeOfLemmas`, `Init.WF`), verified automatically (`rule_exact_match = true`, `0` overlap with spent/cal08a/eval08a/SInt/TRAIN families).

Per file, select up to `12` terminal `T` roots and `6` first-multistep `R` roots (matching OLR-08A's exact per-file selection cap; `SEED_08B = 20270211`, at most one root per declaration, deterministic hash order, zero menu/search outcome peeking). TRAIN uses the exact same deterministic 192-root subset of `olr08/train_registry.json` (`SEED = 20270129`).

## 3. Registered gates (fail-closed; no threshold relaxation)

| Gate | Criterion | Threshold | Failure Action |
| --- | --- | --- | --- |
| **Gate 0A** | Native scope sentinel + 07A/07B custody | PASS | Abort before scoring |
| **Gate 0B** | Native validity (TRAIN & CAL `T`-human replay) | ≥95% in both; ≥20 TRAIN, ≥10 CAL audits | Stop; report `gate0_source_validity_failed` |
| **Gate 1A** | **Calibration cohort feasibility** | `train_hard ≥ 40`, **`cal_hard ≥ 120`**, **`cal_hard_file_clusters ≥ 12`** | Report **`cohort_feasibility_failed`**; no promotion |
| **Gate 1B** | **Offline headroom vs strongest fixed-budget control** | Best 300-step fixed control (over `s1, s2, s3, s3c` **and fixed-budget portfolios** `portfolio_rr2`, `portfolio_rr4`) `≤ 80%`, AND 1,200-step offline reference minus best 300-step control **`≥ 10.0 pp`** | Stop before neural fitting; no promotion; protected EVAL untouched |
| **Gate 2** | Replay-certified TRAIN action frames | `≥ 64` source frames | No neural fit if `< 64` |
| **Promotion** | 500K / 5M architecture promotion | Blocked until all preregistered gates + independent sealed confirmation pass | `architecture_promotion_eligible = false`; keep PR #318 in draft |

## 4. Architecture, controls, BPD, and MMS specification

- **Search caps:** `DEPTH = 3`, `STEP_CAP = 300`, `RPC_CAP = 1200`, `HB_CAP = 512,000,000` raw heartbeats per root/policy (4× caps for the privileged 1,200-step offline headroom reference).
- **Base priors & portfolios:** `s1` (uniform), `s2` (hand), `s3` (counted TRAIN prior), `s3c` (cost-aware heartbeat prior), fixed-budget portfolios `portfolio_rr2 (s2,s3)`, `portfolio_weighted (s3,s2,s3,s3c)`, `portfolio_rr4 (s1,s2,s3,s3c)`, plus online non-neural `adaptive_ucb`.
- **Extended utility predictor (`olr08b/models.py`):** Minimal `+5,766`-parameter extension of OLR-08A's `122,881`-parameter architecture to **`128,647` parameters** (`< 146K`, never 500K/5M), adding 24-dim conditioning (remaining budget, live verifier receipts, 4-prior costs, MMS memory readouts), recurrent frontier state $z_t \in \mathbb{R}^{40}$, applicability/cost auxiliary heads, and 4-prior adaptive allocation gates.
- **Matched controls (3 seeds `17, 29, 43`, 400 CPU AdamW updates, exact matched parameter count + init SHA + sampler SHA across all 5 neural arms):**
  - `recurrent` (3-visit bidirectional graph + recurrent frontier state)
  - `recurrent_bpd` (`recurrent` + BPD component features)
  - `direct_mlp` (0 graph messages, 0 frontier recurrence; matched 128,647-param compute schedule)
  - `matched_compute_sham` (identical `recurrent` compute trained on within-menu shuffled targets)
  - `one_way` (reverse graph messages masked)
  - `linear` (415-param bag + conditioning linear baseline)
  - `explicit_memory` (0 learned parameters; exact empirical-Bayes template/head cache backed by `ProofMemoryCache`)
- **BPD decomposition (`olr08b/bpd_frontier.py`):** Decomposes shared recurrent linear sites (`up`, `down`, `cell_out`, `frontier_cell`) into rank-8 `(V_m, U_m)` factors, explicitly tagging `svd_seed_proxy` vs `learned_bpd`, and tests whether component-level features predict (a) branch utility, (b) tactic applicability, and (c) recovery following ≥2 consecutive failures vs RMS-matched Haar rotation nulls.
- **MMS memory budgets (`olr08b/mms_cache.py`):** Configurable budgets **`0, 64, 256, 1024, 4096 MiB`** with strict lazy byte-budget enforcement (`allocated_bytes <= budget_bytes`), priority/LRU eviction, and explicit accounting of `build_seconds`, `lookup_seconds`, `maintenance_seconds`, hits, misses, evictions, and peak bytes.

## 5. Frozen pre-outcome registry (audited before any outcome collection)

Frozen via `olr08b/launch08b.py --stop-after-freeze` under the 12 GiB Windows Job Object hard commit cap (`threads = 8`, `workers = 2`, peak job commit `3.01 GiB`, peak sampled RSS `2.68 GiB`, wall `86.4 s`). Copies are committed under [results/olr08b/preoutcome/](results/olr08b/preoutcome/).

- **Registry SHA-256:** `818de04c7ec847d4f07ef156c0a9b35fdf0a7414e66411659b735f09c4100217`
- **Native scope sentinel:** PASS (`results/olr08b/preoutcome/sentinel_PASS.json`)
- **Files traced cleanly:** `26 / 26` uncontaminated `src/Init/` files across `16` non-TRAIN families (`rule_exact_match = true`)
- **Contamination overlaps:** `0` spent files, `0` OLR-08A cal files, `0` OLR-08A eval files, `0` `SInt` files, `0` TRAIN families
- **Frozen root counts:** `192` TRAIN (reused referee source), `397` fresh calibration (`301 T` + `96 R` across `26` independent file clusters), `275` protected evaluation (`274` pristine confirmation-eligible; smoke-observed `BitVec.shiftLeftZeroExtend`, id `45ce790b…`, explicitly excluded from pristine confirmation).

