# LOOPSCOPE-02

Does two-pass looping in a 3B language model (`Nanbeige4.2-3B`) buy multi-step constraint-solving capability over an untied 4B model (`Qwen3.5-4B-Base`), and what mechanistically separates both from a 7M-parameter Tiny Recursive Model (`TRM`) across 16 outer ACT steps (`288` inner passes) on the exact same stratified 9×9 Sudoku boards?

Every hypothesis, dataset lock, donor pairing, and decision threshold was registered on `main` (`1f205d6`) before evaluating any held-out test board; see [docs/registration.md](docs/registration.md) and the full synthesis in [docs/report.md](docs/report.md).

## Setup

- **Benchmark (`bundle_sha256: 59459ae0...`, `seed = 20261113`):** 256 distinct 9×9 Sudoku boards from `sapientinc/sudoku-extreme-1k`, stratified by exact synchronous naked+hidden single propagation $G_{\text{prop}}$ into `S1_naked1` (`n=64`), `S2_prop2` (`n=64`), `S3_prop3plus` (`n=64`), and zero-single `S4_stalled` (`n=64`), plus the 100 canonical full boards of `test_sudoku_bench.csv` (`49` propagation-solvable, `51` search-required stalled).
- **Models:**
  1. `gaoxin492/TinyRecursiveModels-Sudoku-Extreme-att` (`step_21700`, 7M 2-layer Attention-TRM, `16 × 3 × 6`) + independent replication `yagizdevre/trm-sudoku-extreme-att` (`model.pt`).
  2. `Nanbeige/Nanbeige4.2-3B` @ `451ed48c` (22-layer looped LM trained at `num_loops=2`, evaluated across `t = 1..8`).
  3. `Qwen/Qwen3.5-4B-Base` @ `1001bb4d` (32-layer untied base LM) + pure-SDPA sensitivity arm `Qwen/Qwen3-4B-Instruct-2507` @ `cdbee75f` (36 layers).

## Results

| ID | Pre-registered focus | Key statistic | Registered verdict |
|---|---|---|---|
| **P1** | Capability triangulation across `S1`–`S4` | `Nanbeige` `t=2` `0.129` vs `Qwen3.5-4B` `0.211` (`Qwen3-4B-Inst` `0.254`; `S2–S4` `-0.010`, $p=0.824$); `TRM` `s=16` `0.938` cell / **`0.082 -> 0.883` full board (`1.000` on `bench100`, `0.920` repl.)** | **Not supported** *(single-cell `s=1` starts at `0.863` after 18 inner passes, capping `s=1->16` cell gain at `+0.074` while full-board jumps `+0.801`)* |
| **P2** | `ARG-02` counterfactuals & attractor repair | `TRM` 10-cell donor corruption repair **`0.000 -> 0.887` full board** ($p=9.3\times 10^{-69}$), kick contraction `1.00 -> 0.303`; `Nanbeige` non-contractive (`0.712` at `t=8`) | **Not supported** *(same single-cell `z_H(1)` ceiling on `open16`/`donor16`: `+0.074` / `+0.109`)* |
| **P3** | SVD usage: pass multiplexing vs stationary operator + role split | `Nanbeige` $\bar{r}_{\text{P1,P2}} = 0.355$ (`L12 = 0.085`, `L22 = 0.277`); `TRM` $\bar{r}_{\text{outer},z_H} = \mathbf{0.992}$, $\bar{r}_{\text{role}}(z_L,z_H) = \mathbf{0.765}$ | **Weak** *(`Nanbeige` `0.355 < 0.40` Weak bar due to layer-selective multiplexing)* |
| **P4** | Unrolled 6-step `TCA`: early fragility masking + specificity | `TRM` **`F_1 / F_6 = 1,808.4`**, while `|z_read| > 3 = 22.04%` (`81.6×` null, flat across steps `1..6`) | **Not supported** *(exposes **Detached-ACT vs BPTT Lyapunov Dichotomy**; `TCA` cancels `6,400×` gradient swing)* |
| **P5** | `Q11` $G_{\text{prop}}$ compactification, staircase, non-monotone search & dark fiber | Commutation **`AUC = 1.000`**; 48-$H$-cycle staircase **`0.653`** (null `0.495`); stalled solve **`1.000`** (`100%` flips, `98–99%` prior conflicts); **`reset_zL` `0.990` vs `lm_head`-nullspace dark-fiber erasure `0.070` (`-93` pp)** | **Supported (`5/5`)** |

## Layout

- `docs/`: [registration.md](docs/registration.md) (pre-registration + execution audit note) and [report.md](docs/report.md) (full synthesis & positioning against 2026 literature).
- `scripts/`: `sudoku_prop.py` ($G_{\text{prop}}$ stratifier & bundle builder), `trm_sudoku.py` (`PureSudokuTRM`, counterfactuals, fiber erasure, `TCA`, `Q11`), `llm_sudoku.py` (`Nanbeige` & `Qwen` runner), `score_loopscope02.py` (CPU replay scorer).
- `tests/`: `test_loopscope02.py` (unit tests + exact committed-receipt replay check).
- `results/`: `benchmark_bundle.json`, per-model JSON receipts, and `loopscope02_report.{json,md}`.

## Reproduce

Run from the repository root (`vramorama` recommended on shared GPUs):

```bash
python -m pytest mechinterp/loopscope02/tests -q
python -m mechinterp.loopscope02.scripts.trm_sudoku --device cuda
python -m mechinterp.loopscope02.scripts.llm_sudoku --profile qwen4b --device cuda
python -m mechinterp.loopscope02.scripts.llm_sudoku --profile qwen3_4b_inst --device cuda
python -m mechinterp.loopscope02.scripts.llm_sudoku --profile nanbeige --device cuda
python -m mechinterp.loopscope02.scripts.score_loopscope02
```
