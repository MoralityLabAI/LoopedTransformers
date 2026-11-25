# LBF-19 Confirmation Registration: Expensive-Verifier Proof Search & Scaled Lengths ($L \in \{16, 20, 24\}$)

**Status:** FROZEN BEFORE DRAND BEACON ROUND `34258635` IS PUBLISHED AND BEFORE ANY CONFIRMATION ROOT IS GENERATED OR SCORED  
**Timestamp (UTC+8):** `2026-11-25T12:00:00+08:00`  
**Base Commit:** Merged LBF-18 PR #349 (`f2ffca191359fa2b3790e784a9827d18240d87c6`, head `0f73bc883ee891a513dd6fda5f5a73c73f2f57dc`)  
**Companion LBF-19 Tracks (Registered & Archived Separately):**
- Track T (Tree-Automata Abstractions): Registration `11581588`, Results `43cf326f` ([`LBF_19_TREE_AUTOMATA_RESULTS.md`](LBF_19_TREE_AUTOMATA_RESULTS.md))
- Track R (Learned Recurrent Transfer): Registration `ab09c870`, Results `ceb9fd03` ([`LBF_19_LEARNED_TRANSFER_RESULTS.md`](LBF_19_LEARNED_TRANSFER_RESULTS.md))

---

## 1. Public Randomness Beacon Commit–Reveal Protocol (Fable §11)

Following Fable's independent referee protocol ([`REFEREE_REPORT.md` §11](review/independent_referee_20261124/REFEREE_REPORT.md)), the confirmation cohort seed is committed to a **future** `drand quicknet` beacon round published **after** this registration commit is pushed to GitHub:

- **Beacon Network:** League of Entropy `drand` `quicknet` (`chain_hash = 52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971`, period `3 s`)
- **Latest Observed Round at Freeze:** `34258610`
- **Registered Future Target Round:** **`34258635`** (`+25` rounds / `~75 s` in the future)
- **Seed Derivation:** `seed_hex = sha256(quicknet_signature_hex_of_round_34258635)` (equivalently the first 16 hex digits of the round signature passed as `--seed-hex` to `lbf19_run.py gen`), with per-length sub-seeds `sha256(f"{seed_int}:{L}:confirm")[:12]`.

---

## 2. Frozen Code & Checkpoint Hashes (SHA-256)

All 18 frozen checkpoints from LBF-17/18 (`seeds 107, 109, 113`) are reused **without retraining** (exact SHA-256 verified at load time against [`results/lbf17/manifest.json`](results/lbf17/manifest.json) and `results/lbf18/{107,109,113}_hybrid_sym_rec_opened/manifest.json`).

| Frozen LBF-19 Module | SHA-256 | Role |
|---|---|---|
| [`lbf19_scaled_domain.py`](lbf19_scaled_domain.py) | `cdff2dd85715f5b8385a8294f4fc75b106c3ac3761af843bc133591044bc57c2` | Lengths $L \in \{16, 20, 24\}$ generator & exact classical `bidir` |
| [`lbf19_native_verifier.py`](lbf19_native_verifier.py) | `faa10b73bb8ef6e5901b6edbeab6823abba89e2067201ba3362a00d063394206` | Live Lean 4.19.0 (`lgym`) tactic execution, correspondence check, zero-axiom certificates |
| [`lbf19_fast_controller.py`](lbf19_fast_controller.py) | `dfe53c2ee507492cb12d94d8d273ff98e879f6913892c2f36b3db0a7b0a9f89a` | Dense pair-store vectorization of `IncrementalSymController` (exact path/work equivalence) |
| [`lbf19_harness.py`](lbf19_harness.py) | `79e33b6a11730002cc29f1013cb540ffbd5d0912559f333e91071e020fb761e2` | Unified verifier-in-the-loop search harness (`count` & `lean` modes) |
| [`lbf19_run.py`](lbf19_run.py) | `90778f858c23155f375047714743eed346130628184fda1d1e44cbb20cab1e53` | Cohort generation, parallel scoring, & certificate batching CLI |
| [`tests/test_lbf19_expensive_verifier.py`](tests/test_lbf19_expensive_verifier.py) | `41764684b026f32d1c45b39f84c17116f721166e8d36283859da28c7fb5beaa3` | Correspondence, dedup, replay, & live zero-axiom Lean cert tests |

---

## 3. Cohort Design, Arms, & Execution Protocol

### 3.1 Scaled Confirmation Cohort ($R = 24$ Independent Roots)
- **Lengths:** $L \in \{16, 20, 24\}$, with `8` independent roots per length (`24` roots total).
- **Rule Distribution:** Identical to pinned LSA-02K (`8` length-2 + `4` length-3 equations over `{a..f}`, all sides distinct).
- **Hardness Calibration (Classical Comparator Only):** Rejection-sampled so uncapped classical bidirectional BFS (`bidir_uncapped`, depth cap `24`) requires **`10,000` to `1,000,000` expansions** (walk lengths `70..110` for $L=16$, `45..70` for $L=20$, `28..50` for $L=24$). No learned or abstract arm is consulted during generation.
- **Lineage Disjointness:** Strictly disjoint from all `144` historical LBF-17 lineages and all `12` LBF-19 dev-pilot lineages (`dev_cohort`).

### 3.2 Registered Arms & Modes
- **Primary Controller:** `sym_prov_readout` (symbolic guarded `prov4_adj2` `u64` Boolean support envelope + frozen learned cross-frontier readout, 3 frozen seeds `107, 109, 113`).
- **Recurrent Comparators:** `rel_prov_guarded` (3-pass learned relational envelope + recurrent axial block, 3 seeds) and `hybrid_sym_rec_opened` (3-pass axial recurrence over progressive symbolic bitsets, 3 seeds).
- **Learned Readouts Without Relational Envelopes:** `lbf16_pos1_opened` (positionwise $K=3$, 3 seeds) and `compute_nonrecurrent` (single-pass cross-fiber, 3 seeds).
- **Exact 3-Hop Lookahead:** `exact_mm` (exact concrete radius-3 ball oracle; every speculative successor is charged to the verifier layer).
- **Symbolic & Classical Baselines:** `absint_mm` (0-parameter 3-hop guarded `prov4_adj2`), `geom_mm` (Hamming MM), `graph_bridge` (k-mer conductance), `bidir_uncapped` (classical bidirectional BFS to closure).
- **Budgets & Execution:**
  1. **Count Mode (Exact Deduplicated Verifier Calls, `B = 512`, depth cap `24`):** All `10` arms across all `24` confirmation roots (plus `B = 1024` for `sym_prov_readout`, `geom_mm`, `graph_bridge`, `exact_mm`, `bidir_uncapped`).
  2. **Live Native Lean-in-the-Loop Mode (`B = 512`, depth cap `24`, `lgym` Lean 4.19.0):** All `24` confirmation roots for `sym_prov_readout` (`3` seeds), `rel_prov_guarded` (seed `107`), `geom_mm`, `exact_mm` (`lean_cap = 2,000` live Lean calls), and `bidir_uncapped` (`lean_cap = 2,000` live Lean calls). Every novel successor executes `conv => lhs/rhs; rw [h_i (term suffix)]` inside Lean 4.19.0 with strict goal-equality checking (`0` mismatches permitted). Every closed episode is replayed natively, kernel-checked (`0` axioms), and compiled standalone via `lean --threads=1`.

---

## 4. Confirmatory Hypothesis Hierarchy (FWER $\alpha \le 0.05$) & Estimation Contrasts

All root-level tests treat the **theorem root** ($R = 24$) as the sole independent unit (averaging across the 3 frozen seeds within root first for multi-seed arms).

1. **`H1` (Primary Closure vs Best Parameter-Free Heuristic, one-sided $\alpha_1 = 0.025$):**
   `sym_prov_readout` achieves strictly higher root-level closure rate at `B = 512` than `max(geom_mm, graph_bridge)` via one-sided root-paired sign-flip randomization (`100,000` draws).
2. **`H2` (Native Verifier Call & End-to-End Proof-Time Reduction vs `bidir_uncapped`, gated on `H1`, one-sided $\alpha_1 = 0.025$):**
   `sym_prov_readout` requires strictly fewer verifier calls than `bidir_uncapped` across all `24` roots (and achieves lower end-to-end native Lean wall-clock time on closed roots than `bidir_uncapped` even when `bidir_uncapped` is censored at `2,000` live Lean calls).
3. **`H3` (Verifier Call Reduction vs Exact 3-Hop Lookahead `exact_mm`, parallel branch one-sided $\alpha_2 = 0.025$):**
   `sym_prov_readout` requires strictly fewer verifier calls than `exact_mm` at `B = 512` across all `24` roots.

### Pre-Declared Estimation-Only Contrasts (Root-Paired 95% Bootstrap CIs, No Binary Claim)
- `rel_prov_guarded` minus `sym_prov_readout` and `hybrid_sym_rec_opened` minus `sym_prov_readout` (testing whether neural recurrence gains utility at $L \in \{16, 20, 24\}$).
- `sym_prov_readout` minus `lbf16_pos1_opened` and `compute_nonrecurrent` (learned readouts without relational envelopes).
- `sym_prov_readout` minus `absint_mm` (learned readout gain over 0-parameter relational abstract interpretation).
- Native Lean latency breakdown: `init` elaboration time, per-call tactic latency (mean/p50/p95 stratified by $L \in \{16, 20, 24\}$), cache hit rate, standalone certificate compile time, and end-to-end proof time.
- **Scope condition:** Inference is conditional on the 3 frozen LBF-17/18 training seeds (`107, 109, 113`) and the restricted equational unary-composition fragment; no retraining is performed in this confirmation.
