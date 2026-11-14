# `mechinterp/loopscope03`: Three-Ledger Iteration Pricing, Conditional Cache Refill, and Path-Dependent Search Memory

`LOOPSCOPE-03` implements a three-ledger state decomposition $S_t = V_t \oplus C_t \oplus R_t$ and iteration-pricing suite across frozen looped-transformer checkpoints (`TRM` primary `gaoxin492` and replication `yagizdevre`, `LoopedChase` `Q4` 12-arm suite, `tomg-group-umd/huginn-0125`, and `Nanbeige/Nanbeige4.2-3B` stationarity admission control).

## Directory Layout

- [`docs/registration.md`](docs/registration.md) — Pre-registration locked at commit `b1ec77550997cbc5ae2cd84618af74e8f34e38e7` prior to held-out inference (`H3.1`–`H3.5`).
- [`docs/report.md`](docs/report.md) — Complete evaluation report covering the Stationarity Admission Gate, Conditional Refill capacity ladder (`C0`–`C6`) and $z_H \times z_L$ cross, Resume-Test loop-equivalent iteration pricing, Fork-Test transient divergence / C-voting (reconciling Anil et al., 2022 and Kubo et al., 2026), causal $R_t$ swaps, and `Huginn-0125` cache contraction.
- [`ledger_lib.py`](ledger_lib.py) — Exact `float64` `P_head` SVD projectors, `Post-RMSNorm` sphere calibration (`autocast(enabled=False)`), exact Sudoku candidate-mask & unit-conflict extractors, periodic visible-board cycle detector, 2-layer/4-layer RoPE+SwiGLU `RefillTransformer`, and exact McNemar / paired permutation tests.
- [`scripts/conditional_refill.py`](scripts/conditional_refill.py) — Stage 1 runner (`train_900` / `val_100` capacity sweep `C0`–`C6` + `items256` & `bench100` closed-loop $z_H \times z_L$ cross at $s \in \{16, 32, 64\}$).
- [`scripts/resume_and_fork.py`](scripts/resume_and_fork.py) — Stage 2 & Stage 3 runner (stationarity admission gate, `LoopedChase` + two-checkpoint `TRM` telescoping Resume Test, `K=8` C-voting & transient fork divergence, and matched-step causal $R_t$ swap / revisit audit).
- [`scripts/huginn_ledger_probe.py`](scripts/huginn_ledger_probe.py) — Supplementary `Huginn-0125` multi-init path-variance contraction, cross-fork `CacheMean` / donor-$R_t$ swap, and $t=4$ Resume Test.
- [`results/`](results/) — Committed JSON receipts (`stage1_conditional_refill.json`, `stage2_3_resume_and_fork.json`, `huginn_cache_and_fork.json`).
- [`tests/test_loopscope03.py`](tests/test_loopscope03.py) — Unit and receipt-integrity tests (`pytest mechinterp/loopscope03/tests`).
