# OLR-07B: local CPU/RAM native diagnostic-reader results

**Status:** native validity **PASS** (`native_valid: true`); calibration headroom gate **FAIL** (`cal_headroom: false`, `fit_gate: false`). Learned fitting deliberately skipped (`0` updates, `0` trained parameters); full classical evaluation completed and archived as negative evidence closing this action family.
**Registration:** [OLR_07B_REGISTRATION.md](OLR_07B_REGISTRATION.md) · **Runbook:** [OLR_07B_LOCAL_RUN.md](OLR_07B_LOCAL_RUN.md) · **Prior audit:** [OLR_07A_RESULTS.md](OLR_07A_RESULTS.md)
**Protocol baseline commit:** `8d04c36c2368d6f8cefc636296d50ff231b5cba1` (`olr07b_cpu.py` and `olr07a_library.py` unmodified)
**CI smoke verification:** [45337841132](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45337841132) (`success`, 49.53 s on `8d04c36c`; local Windows smoke matched counts and scores bit-for-bit)
**Pinned Lean source:** `leanprover/lean4` `v4.19.0` commit `6caaee842e9495688c1567e78c0e68dbb96942aa`
**Prior 07A archive custody:** artifact [14890402416](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45327039231/artifacts/14890402416) (`494,970` bytes, SHA-256 `1bd5225cfef801941bc29c5466e7c6223c69a68e53a46737d711534cfb532814`, preserved unmodified)

## Native validity & cohort census

All 55 registered roster files elaborated with zero errors. Four independent fresh-process replays per scored file (`candidate0`, `candidate1` at HB500; `human0`, `human1` at HB20000) recorded zero state, certifiability, or receipt-category drift across 17,184 scored candidate receipts (34,368 candidate replays plus 2,882 human replays).

| Role | Families | Roster / Scored Files | Selected | Scored | Exclusions (`uncertifiable`) | Human Certified | Human Rate | Certified Menu Actions | Scored Menu Receipts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TRAIN | `Nat` (124), `List` (477), `Option` (61), `Char` (6), `String` (1) | 26 / 23 | 670 | 669 | 1 | 667 | 99.70% | 1,918 | 8,028 |
| Calibration | `Int` (155) | 8 / 6 | 155 | 155 | 0 | 155 | 100.00% | 403 | 1,860 |
| Evaluation | `Array` (363), `Vector` (130), `Fin` (19), `UInt` (96) | 21 / 19 | 616 | 608 | 8 | 608 | 100.00% | 1,275 | 7,296 |
| **Total** | **10 families (disjoint by role)** | **55 / 48** | **1,441** | **1,432** | **9** | **1,430** | **99.86%** | **3,596** | **17,184** |

- All three roles exceed registered count floors (`>=100` TRAIN, `>=30` CAL, `>=100` EVAL) and the `>=95%` human closure validity floor.
- The two TRAIN human replay non-closures occurred in `src/Init/Data/List/Find.lean` (`List.findFinIdx?_subtype`, `List.findIdx?_subtype`). Zero human failures occurred in Calibration or Evaluation.
- Zero files from 07A were reused; exact old goal strings and `(file, decl)` keys were excluded using authenticated manifest `846958ede542bedd510d865b3559e6f4711a97963583084c2d1028e70da2328e`.

## Source calibration gate & conditional training

Frozen 4-action static order fit on TRAIN: `['exact?', 'simp_all', 'omega', 'simp [*]']`.

| Policy (4-call budget) | CAL Closed / 155 | CAL Rate | CAL Mean Calls |
| --- | ---: | ---: | ---: |
| `fixed` (selected base) | 127 | 81.94% | 1.6968 |
| `binary` | 127 | 81.94% | 1.6968 |
| `diagnostic` | 127 | 81.94% | 1.6968 |
| `local_binary` | 121 | 78.06% | 1.7613 |
| `local_diagnostic` | 117 | 75.48% | 1.8387 |
| **12-action menu ceiling (oracle)** | **127** | **81.94%** | — |

- **Calibration headroom (`cal_ceiling - cal_best`):** `0 / 155` (**0.00 pp**; required `>= 10.00 pp`) -> **FAIL**.
- **Best deployable closure:** **81.94%** (required `<= 80.00%`) -> **FAIL**.
- **Conditional neural training:** Skipped per registered protocol (`fit_gate: false`, `model_training_updates: 0`, **0 trained parameters** across all 15 matched checkpoints). For protocol reference only, the unfitted registered architectures contain `146,049` parameters for each tied arm (`bi`, `one_way`, `bi1`, `binary`) and `322,508` parameters for `direct`.

## Fresh held-out evaluation (`Array` / `Vector` / `Fin` / `UInt`)

Evaluation candidate receipts were collected only after freezing `frozen_policy.json`.

| Policy (4-call budget) | EVAL Closed / 608 | EVAL Rate | EVAL Mean Calls | Gap to Ceiling |
| --- | ---: | ---: | ---: | ---: |
| `fixed` (selected base) | 429 | 70.56% | 2.2615 | -0.66 pp (-4 roots) |
| `binary` | 429 | 70.56% | 2.2615 | -0.66 pp (-4 roots) |
| `diagnostic` | 426 | 70.07% | 2.2615 | -1.15 pp (-7 roots) |
| `local_binary` | 414 | 68.09% | 2.2862 | -3.13 pp (-19 roots) |
| `local_diagnostic` | 410 | 67.43% | 2.3289 | -3.78 pp (-23 roots) |
| **12-action menu ceiling (oracle)** | **433** | **71.22%** | — | — |

- **Paired file-cluster bootstrap (`diagnostic - binary`, 19 files, 2,000 resamples):** **-0.49 pp** (`-0.004934`), 95% CI **[-1.20 pp, 0.00 pp]** (`[-0.012048, 0.0]`). Diagnostic failure categories do not outperform binary pass/fail feedback or the static 4-action order.
- Only 4 of 608 fresh evaluation roots are closed by any menu action outside `fixed`: three in `src/Init/Data/Vector/Find.lean` (`Vector.getElem?_zero_filter`, `Vector.getElem_zero_filter`, `Vector.find?_filter`, solved by `simp`) and one in `src/Init/Data/UInt/Lemmas.lean` (`UInt16.toUInt64_ofNatTruncate_of_le`, solved by `decide`/`trivial`).

## Infrastructure notes & resource telemetry

- **Host environment:** Windows 11 (`10.0.26200`), AMD Ryzen AI 9 365 (10 cores / 20 logical threads), 47.12 GiB physical RAM, Python `3.12.10` (`torch==2.5.1+cpu`, `numpy==2.5.4`, `psutil==7.2.2`), Lean `4.19.0` (`6caaee842e94`, CPU-only). Because concurrent host workloads saturated CPU (~100% utilization), the run stayed at the registered resource budget (`--threads 8 --workers 2 --ram 12G --timeout 1800`) rather than scaling up.
- **Infrastructure fixes committed (zero protocol change):**
  1. Added `--priority {idle,below_normal,normal}` (default `below_normal`) to `run_olr07b.py`. An initial smoke attempt at hardcoded `below_normal` stalled at ~2.5% of one core under concurrent `normal`-priority host workloads; running with `--priority normal` under the same Windows Job Object memory cap resolved starvation cleanly.
  2. Added `Job.cpu_seconds()` (`JobObjectBasicAccountingInformation`) in `lcgrt/ramcap.py` and logged `job_cpu_seconds` in `resource_latest.json` so OS-accounted cumulative CPU time across Python and exited `lgym.exe` workers is captured directly.
- **Full run telemetry (`resource_latest.json`):**
  - Wall time: **1,019.28 s** (16.99 min; `seconds_this_invocation: 1,006.68 s`; worker sums: trace `777.04 s`, native select `1,043.32 s` = `candidate0 333.19 s`, `candidate1 322.81 s`, `human0 194.95 s`, `human1 192.37 s`)
  - Cumulative OS-accounted Job CPU time: **1,782.27 s** (29.70 CPU-min)
  - Hard memory cap: **12 GiB** Windows process-tree committed-memory Job Object
  - Peak Job committed memory: **1,524,137,984 bytes** (**1.42 GiB**, 11.8% of 12 GiB cap)
  - Peak sampled process-tree RSS: **2,130,251,776 bytes** (**1.98 GiB**) across peak **4** processes
  - Resume/failure count: **0** interruptions during full run; all 55 trace and 48 scored native file checkpoints completed in a single invocation.
- **Local smoke telemetry (`--threads 4 --workers 1 --ram 8G --priority normal`):** wall `440.35 s`, Job peak commit `646,991,872 B` (617 MiB), peak sampled RSS `983,044,096 B` (937 MiB), exact match to CI smoke `45337841132`.

## Artifacts & custody

All 664 manifest-listed files in `olr07b-full_AUDIT.zip` and all 15 files in `olr07b-full_FABLE.zip` were independently SHA-256 verified against `audit_manifest.json` (`b388335c52ffb4b40b63e3cc77a37202c8c70553901a115635dc74f1d735b1ad`), with zero secrets or local user paths in any JSON receipt.

| Artifact | Path | Size | SHA-256 |
| --- | --- | ---: | --- |
| Full raw audit archive (for ChatGPT audit) | [results/olr07b/olr07b-full_AUDIT.zip](results/olr07b/olr07b-full_AUDIT.zip) | 2,482,014 B (2.37 MiB) | `64dedf29138a27d43130db135f711569f2738c5ee4505707eb8c144abbb88b1e` |
| Compact Fable review bundle (post-audit) | [results/olr07b/olr07b-full_FABLE.zip](results/olr07b/olr07b-full_FABLE.zip) | 68,483 B (66.9 KiB) | `4e1aeded4dbf07225e3d3dd8f4a6d5c792b546a19920092a0c4a59a655d03620` |
| Summary JSON | [results/olr07b/summary.json](results/olr07b/summary.json) | 10,109 B | `caaab5cfd24005223239020d2b82c506657c49f9e40cfa895cf8c7eef98e4032` |
| Contract JSON | [results/olr07b/contract.json](results/olr07b/contract.json) | 6,398 B | `3a5504a09953a29cc8706f4cd0ec6da08f2072abdefda26a875ab6ee10d3d68b` |
| Frozen Policy JSON | [results/olr07b/frozen_policy.json](results/olr07b/frozen_policy.json) | 172 B | `84dd6e7bd119c16ce531d712224d7b083ba370750d67e09b65adca9274d0fd8f` |
| Resource Telemetry JSON | [results/olr07b/resource_latest.json](results/olr07b/resource_latest.json) | 521 B | `1d1b85fffd5b400aee5db3e93f9232bc550b7d67c4a451912cb3081a2678d55f` |
| Audit Manifest JSON | [results/olr07b/audit_manifest.json](results/olr07b/audit_manifest.json) | 73,435 B | `b388335c52ffb4b40b63e3cc77a37202c8c70553901a115635dc74f1d735b1ad` |

Key verified inner audit digests (inside `olr07b-full_AUDIT.zip`):
- `registry.json`: `d5cde5012d5312305b0b6e19b3ea823db8e55bdb95ef83dddfc71dfb2e8c212c`
- `all_receipts.json`: `63b8fcd1aabe0fe9f659ec593f9e68cb4baca1bdd75d9660a19a08234a411b6e`
- `cal_classical.json`: `9c9df92b30794d555170ef6e52c7fdaf034f11c33a9d931f924cccb210511932`
- `evaluation_episodes.json`: `c750819e23d86de6cad951d816d7fb4fde93e6410cf898f3f0bb8c1a02b75bde`

## Scientific verdict

**Negative result — close the 12-action terminal library tactic family.** Native validity is cleanly established (`99.86%` human replay closure across 1,432 fresh Lean 4.19.0 library roots, zero replay drift), fixing the OLR-07A human-audit validity failure. However, a fixed 4-tactic schedule (`exact?`, `simp_all`, `omega`, `simp [*]`) already exhausts **100.0% (127/127)** of the 12-action oracle ceiling on `Int` calibration and **99.08% (429/433)** on held-out `Array`/`Vector`/`Fin`/`UInt` evaluation (0.66 pp remaining gap), while empirical diagnostic feedback degrades evaluation closure by `-0.49 pp` (95% CI `[-1.20 pp, 0.00 pp]`). Per [OLR_07B_REGISTRATION.md](OLR_07B_REGISTRATION.md), no neural checkpoints were fit, `architecture_promotion_eligible` remains `false`, and this action family is closed. Retain the verified checkpointed native receipt collector, Job Object memory/CPU telemetry, and audit bundles as engineering evidence; hand `olr07b-full_AUDIT.zip` to ChatGPT for independent audit before passing `olr07b-full_FABLE.zip` to Fable.
