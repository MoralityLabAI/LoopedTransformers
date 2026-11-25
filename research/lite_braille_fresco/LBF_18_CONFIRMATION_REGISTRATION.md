# LBF-18 Revised Independent Sealed-Cohort Confirmation Registration (Frozen Pre-Unsealing Protocol)

**Status:** FROZEN BEFORE PARSING ANY SEALED `ordinary` OR `ood` SPLIT  
**Timestamp (UTC+8):** `2026-11-25T02:28:00+08:00`  
**Base Commits:** Merged LBF-17 `3d8243818858c19c5031c36e2bf7e328f831cf25` ([PR #343](https://github.com/MoralityLabAI/LoopedTransformers/pull/343)), [PR #348](https://github.com/MoralityLabAI/LoopedTransformers/pull/348) (`404e1483d53e9394fef12f1a705407a87bfe451b`), and Fable's preserved signed referee audit commit `8102fb2190e3c0d30a5a7fad93694d4eaa0af6bd` (`origin/research/lbf-independent-referee-review-20261124`).

---

## 1. Pre-Unsealing Attestation & Frozen Cryptographic Hashes

Prior to committing this registration, **zero** sealed theorem roots in `20261227/{ordinary,ood}` (`32` roots) or `20261221/{ordinary,ood}` (`32` roots) have been parsed or evaluated (`"sealed_outcomes_opened": false` in [`results/lbf18/calibration_phase_summary.json`](results/lbf18/calibration_phase_summary.json)). Only raw compressed/uncompressed byte stream SHA-256s were verified.

### 1.1 Frozen Evaluation Code, Formalization, & Phase 1 Calibration Artifacts (Canonical LF SHA-256)

| Artifact | Canonical LF SHA-256 | Notes |
|---|---|---|
| [`lbf18_incremental_fresco.py`](lbf18_incremental_fresco.py) | `767b55c87779b8d9e9bd9101507383eee6a5a592661fee6cf8b5769e15c7d275` | Guarded transfer, decoupled `u64` support, symbolic controls, cost sweep, Phase 2 runner |
| [`lean/LBF18IncrementalAndCollisions.lean`](lean/LBF18IncrementalAndCollisions.lean) | `80b1810b58144b0203eeda57ff9e9038890a59267438b704be4312179b93c1a9` | `22/22` zero-axiom theorems verified in Lean 4.19.0 (`--threads=1`) |
| [`rust_trellis/src/lib.rs`](rust_trellis/src/lib.rs) | `5d94aad21a8f3a29d8fb9e08c75838150ce7dd11fcab3ee8766a327dac089f71` | Guarded `u64` bitset & continuous envelope kernel + `min_pos_f32` monitor |
| [`LBF_18_INCREMENTAL_FRESCO_SPECIFICATION.md`](LBF_18_INCREMENTAL_FRESCO_SPECIFICATION.md) | `92859202f791746edd1ffdedaf886ab24bc3b7249eb732891e6a4a38e5963f9f` | Formal specification of `Theorem A`, `Theorem B`, Stage A/B/C caching, and cost accounting |
| [`results/lbf18/calibration_phase_summary.json`](results/lbf18/calibration_phase_summary.json) | `fcca51cecd21bf7c91ba728de9afb6bf036e9268fb1c89d1d73d37db651ade0a` | Complete Phase 1 open-cohort soundness, clean timing, RSS, factorial, & stress audit |
| [`results/lbf18/calibration_episodes.jsonl.gz`](results/lbf18/calibration_episodes.jsonl.gz) | `fd3435c1247fbef4154fbfca07579748500ca3592935b5f0e0f9504294016e32` | Decompressed canonical LF SHA-256 across all `12,960` calibration episodes |
| LBF-17 Frozen Suite (`seed = 20261227`) | `a85f402f674423efd4967f6d3b2128450964a200a40f2b0d47f40ab2b7cfc8e2` | LF: `a2a8e6b18040ada5f7e9d06e8e3144f7cb64b950bccf92b444d440b92454d225` |
| LBF-16 Frozen Suite (`seed = 20261221`) | `ed10fcf5bc72c9cc0f0ba7847fa5ebc11d9ec04c6a3a98a3b17c0ef316957375` | Lineage-disjoint from `20261227` training set |

### 1.2 Frozen Checkpoint Roster (`18` Checkpoints Across Seeds `107, 109, 113`)

All 24 historical checkpoints in [`results/lbf17/manifest.json`](results/lbf17/manifest.json) remain unmodified. The 18 checkpoints entering sealed confirmation are:

| Architecture | Params | Seed `107` SHA-256 | Seed `109` SHA-256 | Seed `113` SHA-256 |
|---|---:|---|---|---|
| `rel_prov_opened` (LBF-17 frozen) | `53,078` | `6cf18a21106065fe0054e10cf1bbd7d54120e283dabdb23b10b63bfc9122b582` | `e096ad083229918f29794faedfc0aa6d7ee7e7422716d77ef4a1bbbb66a1cb6b` | `a21d6bfc2d14b8acd7714494bfbe4fb4edb7ea06548a98969f2f35e81609b52a` |
| `sym_prov_readout` (LBF-17 frozen) | `53,078` | `5826ff4b9c91fbbd4c268c00e99ab91a2c799e622130013bbbaf28aec10f9e48` | `9c1017b3e68e988482e8210abe098afe0484caf7b1e70b4c44f5aba872f91032` | `bb8cea9b2f0479b6b814b05eb445ae9f779c74cb5be35e480c2bc89a6647aa33` |
| `compute_nonrecurrent` (LBF-17 frozen) | `53,070` | `d30f303dae37a99895acfbe78a167d6e314d52c42bef33bb75afa965cb8c5671` | `ff7081cbedafb44b376709c49059df6414a7330fb5061181ebd5b00c03f59e34` | `57026c64aaa117bb595c4744e3cce19d9a77c1d4a850165cf2db57b604af99a0` |
| `param_nonrecurrent` (LBF-17 frozen) | `53,093` | `0ad40940334a4d8b25b6bdd4c19ced43fdc262982eb5af52dff2c691f5326c58` | `44c224707bb5104a3582a3a93e3c3453c039e822b7516bc7c2972ac1dfe7ee52` | `28fa4cb6037b4a063b070c40b4f83dbc1526de1b06bd1dd2d9f9a7df176049fd` |
| `lbf16_pos1_opened` (LBF-16/17 frozen) | `53,047` | `481ba8969a8e2ce0f48f5bbac71b94c258fd62969b721f71dd1a667a318196aa` | `b827b1b5599802c9c65191fe5b2e8e57af731c3eef67cb71ad27ad50a75a7456` | `3b7b64a934e72ae91e169705ac5d16d19fd3208c9ce900843ee83a73ffa1b84b` |
| `hybrid_sym_rec_opened` (LBF-18 train-only) | `53,062` | `acd81d302d614f331417495ef55e83e576b5d7de799c84f04685d93270ca3680` | `9f8616bc2580ad07c426a545eaf13494f49d658d538f03d49a6ab2bc2f366fcd` | `381417dc9533ec7e999ba4c9ee7c0e174ba7c663abd057dba6baf84c5bfd2dde` |

### 1.3 Frozen Hyperparameters & Train-Only Adaptive Halting Margins
- Primary search budget curve: $B \in \{16, 32, 64, 96, 128\}$, primary budget $B = 128$, depth cap $D = 14$, congruence fibers $F_p = 4$, recurrent passes $K = 3$, `margin_floor = 1e-6`.
- Adaptive pass-halting thresholds calibrated exclusively on `20261227/train` (`2,031` frame-seed pairs, `>= 98.5%` Pass-3 top-action agreement): **`gamma1 = 2.0`**, **`gamma2 = 0.278`**.

---

## 2. Summary of Completed Phase 1 Open-Cohort Repairs & Calibration Diagnostics

1. **Guarded Soundness (`Theorem A`):**
   - Kernel-verified in Lean 4.19.0 (`theorem_A_fable_guarded_prov_adj2_soundness` & `theorem_A_pr348_mask_guarded_prov_adj2_soundness`, `0` axioms).
   - On `R* = {abc ↔ def, ab ↔ de, bf ↔ cc}`, `legacy` loses `(1, 'c')` across $L \in \{3, 4, 10, 11, 13\}$ and all 3 neural seeds (`act = 0.0`), while both `fable_len_guard` and `pr348_mask_guard` achieve `0` false negatives (`act in [0.674, 0.818]`).
   - On the `100`-system nested-pair-enriched audit cohort (`2,565` scanned systems, `3.90%` nested-pair rate, `2,605` words), `legacy` has symbol false negatives on **`64 / 100` systems (`420` words)**; **`fable_len_guard` and `pr348_mask_guard` both achieve `0 / 100` systems (`0` words)**.
2. **Decoupled Boolean Support (`Theorem B` & `CE-3`):**
   - Pure-Boolean `u64` bitset support achieves `0` missing domain cells at all depths $K \in \{1, 2, 3, 4, 6, 8\}$ on both `rel_prov_opened` (`min_pos_f32 = 0.3445` at $K=3$) and `rel_prov_standard` (where `float32` underflows to `2.11e-17` at $K=3$ and `0.0` at $K \ge 4$).
3. **Calibration (`R = 48` Independent Roots) Baseline Summary ($B = 128$):**
   - `rel_prov_guarded` (Fable guard, Rust cold): `100.0%` closure (`48/48` on all 3 seeds), `29.08` charged work, `125.2` unique enumerated successors, `159.75 ms` cold (`78.52 ms` warm), `244.93 MB` isolated clean-process peak RSS (`144/144` exact trajectory match with uncached PyTorch, incremental PyTorch, warm cache, and `pr348_mask_guard`).
   - `sym_prov_readout` (guarded, Rust cold): `100.0%` closure, `30.19` charged work (`rel_prov_guarded - sym_prov_readout = -1.10` work, `95% CI [-2.51, +0.05]`, `p = 0.1001`), `134.1` unique enumerated successors, `79.26 ms` cold, `200.96 MB` isolated peak RSS.
   - `hybrid_sym_rec_opened` (guarded, Rust cold): `100.0%` closure, `29.56` charged work (`-0.48` work diff, `p = 0.3665`), `138.18 ms`.
   - Zero-parameter symbolic controls: `absint_mm` (`93.75%`, `44.94` work, `178.2` unique enum), `absint1_mm` (`89.58%`, `50.81` work), `exact_mm` (`100.0%`, `31.42` charged work, **`3,389.0` unique concrete ball successors**), `geom_mm` (`81.25%`, `59.10` work), `graph_bridge` (`81.25%`, `58.92` work), `bidir@128` (`62.50%`, `84.52` work, `1.43 ms`), uncapped `bidir@4096` (`100.0%`, `147.54` work & unique enum, `3.42 ms`).
   - Break-even synthetic verifier cost vs `bidir@4096`: **`0.65 ms`** (`sym_prov_readout` cold), **`0.63 ms`** (`rel_prov_guarded` warm), **`1.32 ms`** (`rel_prov_guarded` cold).

---

## 3. Registered Sealed-Cohort Confirmation Design (`64` Lineage-Disjoint Roots)

### 3.1 Sealed Cohorts (`4 x 16 = 64` Independent Roots, Executed Exactly Once)
1. `lbf17_ordinary`: `16` roots (`seed = 20261227`, $L \in \{10, 11\}$)
2. `lbf17_ood` (length-shift): `16` roots (`seed = 20261227`, $L = 13$)
3. `lbf16_ordinary`: `16` roots (`seed = 20261221`, $L \in \{10, 11\}$)
4. `lbf16_ood` (length-shift): `16` roots (`seed = 20261221`, $L = 13$)

Pooled strata: **`32` ordinary roots** (`lbf17_ordinary + lbf16_ordinary`), **`32` length-shift $L=13$ roots** (`lbf17_ood + lbf16_ood`), and **`64` total sealed roots**. All inference treats the **theorem root** $r$ (averaging across the 3 frozen seeds `107, 109, 113` within root first) as the sole independent unit.

### 3.2 Registered Arms (`19` Total: `9` Learned + `10` Symbolic)
- **Primary arm:** `rel_prov_guarded` (`rel_prov_opened` frozen weights + `fable_len_guard` + decoupled `u64` Boolean support + incremental Rust trellis, cold cache per episode).
- **Guard & adaptive comparators:** `rel_prov_mask_guard` (`pr348_mask_guard`), `rel_prov_legacy` (registered LBF-17 unguarded transfer), `rel_prov_adaptive` (`gamma1 = 2.0, gamma2 = 0.278`).
- **Primary symbolic-envelope / non-recurrent learned comparators:** `sym_prov_readout` (guarded `u64` bitsets + frozen learned readout, primary architecture comparator), `hybrid_sym_rec_opened`, `compute_nonrecurrent`, `param_nonrecurrent`, `lbf16_pos1_opened`.
- **Symbolic controls:** `bidir` ($B = 128$), `bidir_b4096` (uncapped up to $4096$), `astar`, `geom_mm`, `graph_bridge`, `exact_lookahead`, `uniform` (`3` streams), `absint1_mm`, `absint_mm`, `exact_mm`.

### 3.3 Confirmatory Hypothesis Hierarchy (Fixed Sequence, Total FWER $\alpha \le 0.05$)
Following Fable's confirmation protocol ([`REFEREE_REPORT.md` §11](review/independent_referee_20261124/REFEREE_REPORT.md)), confirmatory testing uses a strict fixed-sequence gate:

1. **`H1` (Ordinary Cohort Penalized Work vs Best Heuristic, $R = 32$, one-sided $\alpha_1 = 0.025$):**
   Let $\bar{W}_r(\texttt{rel\_prov\_guarded})$ be the 3-seed mean penalized work at $B = 128$ on ordinary root $r$, and $W_r^* = \min\bigl(W_r(\texttt{geom\_mm}),\ W_r(\texttt{graph\_bridge})\bigr)$ be the **rootwise minimum** of the two strongest heuristic baselines. Test $H_{1,0}: \mathbb{E}[\bar{W}_r - W_r^*] \ge 0$ vs $H_{1,1}: \mathbb{E}[\bar{W}_r - W_r^*] < 0$ via one-sided root-paired sign-flip randomization (`100,000` draws, `seed = 20261861`).
2. **`H2` (Length-Shift $L = 13$ Penalized Work vs Best Heuristic, $R = 32$, gated on `H1`, one-sided $\alpha_1 = 0.025$):**
   Conditional on `H1` passing, test $H_{2,0}: \mathbb{E}[\bar{W}_r - W_r^*] \ge 0$ vs $H_{2,1}: \mathbb{E}[\bar{W}_r - W_r^*] < 0$ on the `32` length-shift $L = 13$ roots (`seed = 20261862`).
3. **`H3` (Ordinary All-3-Seed Closure Rate $> 88\%$, $R = 32$, parallel branch one-sided $\alpha_2 = 0.025$):**
   Let $K_{\text{all3}} = \sum_{r=1}^{32} \prod_{s \in \{107,109,113\}} C_{r,s}(\texttt{rel\_prov\_guarded})$ be the count of ordinary roots closed by **all three** frozen seeds at $B = 128$. Test $H_{3,0}: p \le 0.88$ vs $H_{3,1}: p > 0.88$ via exact upper-tail binomial test ($0.88^{32} = 0.01678 \le 0.025$, requiring `32 / 32`).

### 3.4 Pre-Declared Estimation-Only Contrasts & Secondary Operational Screens
Reported with root-paired `95%` bootstrap intervals and two-sided sign-flip $p$-values (no confirmatory binary claim):
- **`rel_prov_guarded` minus `sym_prov_readout`** (pre-declared equivalence margin **`±2.0` charged work**) to measure how much learned 3-pass word recurrence adds beyond exact symbolic `prov4_adj2` propagation + learned readout.
- **`rel_prov_guarded` minus `absint_mm`** and **`sym_prov_readout` minus `absint_mm`** to separate the gain of learned readout from zero-parameter abstract-interpretation MM.
- **`rel_prov_guarded` minus `exact_mm`**, `hybrid_sym_rec_opened`, `rel_prov_mask_guard`, `rel_prov_legacy`, `compute_nonrecurrent`, `param_nonrecurrent`, and `lbf16_pos1_opened`.
- **Unique enumerated successors (`rel_prov_guarded` minus `bidir_b4096`)** and **`0, 1, 10, 100 ms` synthetic successor-cost totals** across all 19 arms.
- **Online Equivalence, Soundness, & Lean 4.19.0 Kernel Certification (`Screen C4`):**
  1. Uncached guarded PyTorch must match incremental guarded Rust path-for-path on all `64 × 3 = 192` primary sealed episodes (`0` divergences).
  2. Exact Boolean support `supp_bits_u64` must achieve **`0` symbol false negatives** against exact concrete 3-BFS across every frontier word encountered on all 64 sealed roots.
  3. **100%** of closed sealed episodes across all evaluated arms at $B = 128$ (plus `bidir@4096`) must compile standalone in Lean 4.19.0 with zero axioms (`Proofs_*.lean` archived in `results/lbf18/sealed_confirmation_certificates/`), and all per-episode records across all budgets must be archived in `results/lbf18/sealed_episodes.jsonl.gz`.
- **Scope condition:** All results are conditional on the 3 frozen LBF-17 training seeds (`107, 109, 113`). No reruns or threshold edits are permitted after `--phase confirm-sealed` begins.
