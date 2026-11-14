# LOOPSCOPE-03 Evaluation Report: Three-Ledger Iteration Pricing, Conditional Cache Refill, and Path-Dependent Search Memory

**Pre-registration commit:** `b1ec77550997cbc5ae2cd84618af74e38e7` (`mechinterp/loopscope03/docs/registration.md`, locked prior to held-out inference)  
**Date:** 2026-11-14  
**Receipts:**
- `results/stage1_conditional_refill.json` (`5,019.3 s`, Stage 1 capacity ladder + $z_H \times z_L$ cross + $s \in \{16, 32, 64\}$ cycle audit)
- `results/stage2_3_resume_and_fork.json` (`17,551.8 s`, Stage 2 Resume Test + Stage 3 Fork Test / C-voting across `Nanbeige4.2-3B`, `LoopedChase`, and both `TRM` checkpoints)
- `results/huginn_cache_and_fork.json` (`411.7 s`, supplementary path-independence & conditional-cache audit on `tomg-group-umd/huginn-0125`)

---

## 1. Executive Summary & Pre-Registered Verdict Ledger (`H3.1`–`H3.5`)

Fixed-budget ($s=16$) hidden-state erasure confounds two distinct effects in stationary, input-re-injected loops $S_{t+1} = F_x(S_t)$: (i) **recomputable cache** that the operator rebuilds in $\Delta\tau \ge 1$ additional iterations, and (ii) **cross-buffer backup channels** when only $(I - P_{\text{head}})z_H^{\text{board}}$ is ablated while $z_L$ and the 16 prefix tokens $z_H^{\text{pref}}$ remain live. `LOOPSCOPE-03` decomposes the carried state at outer step $t$ into **three disjoint ledgers** $S_t = V_t \oplus C_t \oplus R_t$:

1. **$V_t$ (Visible / Head-Readable State):** $z_H^{\text{board}}(t) P_{\text{head}}$ (`rank = 11` on the 81 Sudoku cells; `rank = 16` in `LoopedChase`).
2. **$C_t$ (Conditional Cache):** $\hat{H}_\theta(x, V_t)$, the sphere-calibrated component of the **entire** hidden state $H_t = \big(z_H^{\text{board}}(t) P_\perp,\; z_H^{\text{pref}}(t),\; z_L(t)\big)$ predictable from puzzle clues $x$ and visible logits $V_t$ alone (fitted strictly on `train_900 = train.csv[0:900]`, validated on `val_100 = train.csv[900:1000]`, `0` overlap with `items256` and `bench100`). Because an external monitor observes $(x, V_t)$, computation in $V_t \oplus C_t$ is **monitorable in principle**.
3. **$R_t$ (Path-Dependent Memory):** $R_t = H_t - \hat{H}_\theta(x, V_t)$, existing solely because of the trajectory taken. **Only $R_t$ is genuinely dark.**

| Hypothesis | Verdict | Pre-registered threshold | Measured outcome across models & checkpoints |
|---|---|---|---|
| **Admission Gate** | **Exact split** | $\bar{r}_{\text{outer}} \ge 0.90$ & transplant $L_\infty < 10^{-5}$ | `Nanbeige4.2-3B` **fails** ($\bar{r}=0.355$, loop-8 acc `0.090` vs. loop-2 `0.129` $\to$ **Regime I**). `TRM` ($\bar{r}=0.9925$, $L_\infty=0.0$), `LoopedChase`, and `Huginn-0125` ($L_\infty=0.0$) **pass**. |
| **`H3.1`** ($z_L \times z_H$ backup channel) | **Supported** | Joint `V_only` drop $\ge 15$ pp vs. clean & exceeds single-channel erasure | Primary `gaoxin492` `items256` `WB-Stall@16`: single-channel `0.749` (`mean_all_zH`) / `0.765` (`reset_zL`) / `0.792` (`mean_zL`), whereas joint `V_only` drops to **`0.617` / `0.612`** (`-21.9` / `-22.4` pp, $p \le 4.6\times 10^{-10}$) and traps **`33.3%`–`36.6%`** of stalled boards in exact periodic cycles through $s=64$ (`0.519` / `0.486` and `43.2%`–`44.8%` cycles on `yagizdevre`). |
| **`H3.2`** (Propagation cache sufficiency) | **Supported** | `LoopedChase` ($\lambda \ge 0.1$) & `TRM` `WB-Prop`: `VC` $\ge 97\%$ at $s=16$ (`100%` at $s=32$) & $\text{Price}(R_t) < 0.25$ | `LoopedChase` $\lambda=0.1$: all-token `VC = 1.000` (`3/3` seeds), `Price(C_4) = +5.02`, **`Price(R_4) = 0.00` loops**. `TRM` `WB-Prop`: `VC_C5 = 1.000` (`73/73`) at $s=16, 32$ on **both** checkpoints; primary `Price(R_t) = +0.03, +0.01, 0.00, 0.00, 0.00` loops ($t=1,2,4,8,12$). |
| **`H3.3`** (Post-stall $R_t$ & capacity saturation) | **Partially supported** *(checkpoint-asymmetric)* | `|R^2(C6)-R^2(C5)| < 0.02`; significant `VC` stalled deficit & `Price(R_t) > 0.5` with flip $\rho > 0.25$ | Primary `val_100` $R^2$ saturates (`0.844 -> 0.863`, $\Delta=0.0184 < 0.02$). Primary `VC_C5/C6` recovers ~63% of `V_only` drop (`0.617 -> 0.760/0.743`) while leaving a **`-7.7` to `-9.3` pp deficit at $s=16$** ($p \le 0.0013$) and **`-8.7` to `-10.4` pp at $s=64$** ($p \le 3.1\times 10^{-5}$), with `Price(R_8) = +0.92` loops on clean-solved `WB-Stall` ($p=0.0062$) and **`+7.65` loops on active `WB-Stall`** ($n=17, p=0.0128$). Flip correlation with single-step `Price(R_4)` is modest ($\rho=+0.198, p=0.011 < 0.25$), whereas $\|R_t\|_2^2$ correlates strongly ($\rho=+0.819$ overall / $+0.794$ stalled at $t=2$, $+0.461$ at $t=4$, $p=0.0002$). On `yagizdevre`, clean $R_t$ is **under-damped** (`25.1%` periodic cycles; clean `WB-Stall` degrades `0.656 -> 0.590` from $s=16$ to $s=64$); `VC_C4–C6` **denoises** it to **`0.743`–`0.754` at $s=64`** (`+15.3` to `+16.4` pp above clean, $p \le 2.5\times 10^{-5}$; `Price(R_4) = -10.03` on `bench100` active stalled, $p=0.0070$). |
| **`H3.4`** (Transient fork divergence + C-voting) | **Supported** | `100%` asymptotic solved agreement, $\ge 2\times$ `WB-Stall`/`WB-Prop` divergence, & C-voting gain | Solved forks agree **`100.0%`** across every run (Anil et al., 2022), while peak transient divergence ($\sigma=0.35$) is **`2.27×`** higher on `WB-Stall` (`0.215` vs. `0.095`). Label-free `K=8` C-voting (Kubo et al., 2026) matches the **`K=8` oracle union `100.0%`**, lifting `items256` `WB-Stall` **`0.836 -> 0.923 / 0.940`** ($p \le 3.1\times 10^{-5}$) on primary and **`0.656 -> 0.858`** (`+20.2` pp, $p=1.5\times 10^{-11}$; `bench100` `0.920 -> 1.000`) on replication. |
| **`H3.5`** (Causal $R_t$ swap & revisit signature) | **Partially supported** *(refined mechanism)* | `Swap-R_B > Zero-R` ($p<0.01$) on `WB-Stall` (null on `WB-Prop`) + elevated revisits on $R_t$ loss | `Swap_R_B` steers step-$(t+1)$ choices toward donor $B$ above `Zero_R` on `items256` `WB-Stall` across both checkpoints (`+4.5, +3.7, +2.6` pp on primary, $p=8\times 10^{-4}, 2\times 10^{-4}, 0.014$ [chart-local vs. `Swap_R_C`: $p=0.059\text{--}0.38$]; `+13.5, +11.8, +10.6` pp vs. `Zero_R` and `+14.1, +12.3, +11.7` pp vs. `Swap_R_C` on replication, all $p=0.0002$), while null on `WB-Prop` ($p \ge 0.32$). Stale `Rewind_R_A1` raises 9-cell unit and cell revisits ($p \le 0.0056$), whereas `Zero_R` (`VC`) raises *global 81-cell limit cycles* (`1.6% -> 7.1%–9.3%`) without raising local 9-cell unit revisits. |

---

## 2. Measurement 1: Conditional Refill, Capacity Saturation, and the $z_H \times z_L$ Cross

### 2.1 Manifold Calibration & Held-Out `val_100` Refill $R^2$
Because every `TRMBlock` ends in `Post-RMSNorm` ($\|v\|_2 = \sqrt{512}$), each board token's head-orthogonal radius $\|z_{H,i}^\perp(t)\|_2 = \sqrt{512 - \|V_{t,i}\|_2^2}$ is deterministically fixed by $V_{t,i}$. Recombining $V_t$ with any sphere-calibrated refill $\hat{H}_\theta(x, V_t)$ achieves exact `float32` logit parity (`max_f32_logit_diff = 2.29e-5` on `gaoxin492`, `1.53e-5` on `yagizdevre`) and near-zero continuous hidden-perturbation leak (`1.49e-6` and `3.76e-6`). Without sphere calibration (`VC_C5_raw_uncalibrated`), conditional-mean Jensen shrinkage collapses `items256` `WB-Stall@16` to `0.612` (`gaoxin492`) and `0.415` (`yagizdevre`).

| Capacity tier $\mathcal{F}_k$ (`train_900` fit) | Primary `val_100` Joint $R^2$ (Prop / Stall) | Primary $R^2(z_H^\perp / z_L / z_H^{\text{pref}})$ | Repl. `val_100` Joint $R^2$ (Prop / Stall) | Repl. $R^2(z_H^\perp / z_L / z_H^{\text{pref}})$ |
|---|---:|---:|---:|---:|
| `C0_init` (`H_init`, `L_init` sphere fill) | `-2.104` (`-3.257` / `-1.986`) | `-0.690` / `-2.002` / `-4.818` | `-1.256` (`-1.369` / `-1.240`) | `-0.525` / `-1.285` / `-1.598` |
| `C0_mean` (global `train_900` sphere mean) | `0.000` (`0.000` / `0.000`) | `0.000` / `0.000` / `0.000` | `0.000` (`0.000` / `0.000`) | `0.000` / `0.000` / `0.000` |
| `C1_linear_local` (cell-local ridge) | `0.545` (`0.628` / `0.537`) | `0.380` / `0.542` / `0.759` | `0.335` (`0.323` / `0.337`) | `0.294` / `0.359` / `0.192` |
| `C2_linear_peer` (+ candidate mask & unit probs) | `0.573` (`0.645` / `0.566`) | `0.419` / `0.571` / `0.766` | `0.422` (`0.393` / `0.426`) | `0.378` / `0.456` / `0.205` |
| `C3_tf_2L_128` (`2L d=128` RoPE+SwiGLU) | `0.756` (`0.799` / `0.752`) | `0.722` / `0.753` / `0.837` | `0.640` (`0.589` / `0.647`) | `0.678` / `0.677` / `0.340` |
| `C4_tf_2L_256` (`2L d=256` RoPE+SwiGLU) | `0.803` (`0.846` / `0.799`) | `0.785` / `0.800` / `0.862` | `0.676` (`0.617` / `0.685`) | `0.747` / `0.713` / `0.360` |
| `C5_tf_2L_512` (`2L d=512`, `TRM` capacity match) | `0.844` (`0.888` / `0.840`) | `0.836` / `0.843` / `0.877` | `0.689` (`0.631` / `0.697`) | `0.774` / `0.726` / `0.356` |
| `C6_tf_4L_512` (`4L d=512`, `2×` `TRM` depth) | **`0.863`** (**`0.909`** / **`0.858`**) | **`0.853`** / **`0.862`** / **`0.889`** | **`0.722`** (**`0.660`** / **`0.730`**) | **`0.813`** / **`0.760`** / **`0.370`** |
| `C5_tf_2L_512_stepclock` (+ explicit step $t$) | `0.822` (`0.863` / `0.817`) | `0.815` / `0.818` / `0.872` | — | — |

Key observations:
- **Large spatial multi-cell cache ($C_1 \to C_5/C_6$):** Roughly **85–86%** of primary `gaoxin492` hidden variance (`90.9%` on `WB-Prop`, `85.8%` on `WB-Stall`) and **76–81%** of `yagizdevre` board/$z_L$ variance is deterministic multi-cell cache $\hat{H}_\theta(x, V_t)$ computable from $(x, V_t)$ via self-attention (`C5/C6` jumps $+29$–$30$ points in $R^2$ over linear peer ridge `C2`).
- **Capacity saturation & step-clock invariance:** Doubling Transformer depth from `C5` (`2L-512`, matching `TRM`) to `C6` (`4L-512`) adds only `+0.0184` joint $R^2$ on primary (`+0.0182` on stalled, meeting `< 0.02`) and `+0.0326` on replication, while feeding explicit step index $t$ (`C5_stepclock`) does not improve held-out $R^2$ (`0.822` vs. `0.844`), ruling out a hidden scalar loop counter as the unpredicted residual.

### 2.2 Closed-Loop $z_H \times z_L$ Cross & Extended Horizon ($s \in \{16, 32, 64\}$) on `items256`

| Closed-loop intervention ($s \ge 1$) | Prim. Prop `@16 / @64` | Prim. Stall `@16 / @64` | Prim. `S4@16 / @64` | Prim. Stall Cycle% | Repl. Prop `@16 / @64` | Repl. Stall `@16 / @64` | Repl. Stall Cycle% |
|---|---:|---:|---:|---:|---:|---:|---:|
| `clean` (full $V \oplus C \oplus R$) | **1.000 / 1.000** | **0.836 / 0.896** | **0.547 / 0.719** | `1.6%` | 0.877 / 0.890 *(ever 1.00)* | 0.656 / **0.590** *(ever 0.842)* | **`25.1%`** |
| `live_zH__reset_zL` (`z_L <- L_init`) | 1.000 / 1.000 | 0.765 / 0.798 | 0.453 / 0.500 | `14.8%` | 0.890 / 0.904 | 0.628 / 0.672 | `23.0%` |
| `live_zH__mean_zL` (`z_L <- mean`) | 1.000 / 1.000 | 0.792 / 0.852 | 0.469 / 0.594 | `3.3%` | 0.959 / 0.945 | 0.645 / 0.639 | `25.1%` |
| `mean_board_zH__live_zL` (`v012` style) | 1.000 / 1.000 | 0.710 / 0.809 | 0.359 / 0.562 | `7.1%` | 0.959 / 0.959 | 0.601 / 0.699 | `24.0%` |
| `mean_all_zH__live_zL` (+ prefix mean) | 0.973 / 1.000 | 0.749 / 0.836 | 0.438 / 0.594 | `10.4%` | 0.986 / 1.000 | 0.623 / 0.689 | `23.5%` |
| **`V_only__mean_board_zH__reset_zL`** | 0.959 / 0.959 | **0.612 / 0.628** | **0.234 / 0.266** | **`36.6%`** | 0.945 / 0.945 | **0.486 / 0.536** | **`43.2%`** |
| **`V_only__mean_all`** (`z_H, z_L <- mean`) | 0.932 / 0.945 | **0.617 / 0.650** | **0.281 / 0.328** | **`33.3%`** | 0.918 / 0.932 | **0.519 / 0.546** | **`44.8%`** |
| `V_only__init_all` (`H_init, L_init` sphere) | 0.041 / 0.014 | 0.000 / 0.000 | 0.000 / 0.000 | `99.5%` | 0.096 / 0.068 | 0.000 / 0.000 | `78.7%` |
| `refill_all_zH__live_zL` (`C5` on $z_H$) | 1.000 / 1.000 | 0.820 / 0.874 | 0.516 / 0.641 | `2.2%` | **1.000** / 0.973 | **0.727 / 0.792** | `8.7%` |
| `live_zH__refill_zL` (`C5` on $z_L$) | 1.000 / 1.000 | 0.781 / 0.847 | 0.438 / 0.578 | `4.4%` | 0.932 / 0.973 | 0.661 / 0.721 | `17.5%` |
| `VC_C1_linear_local` ($V \oplus C_1$) | 0.986 / 1.000 | 0.672 / 0.721 | 0.234 / 0.281 | `8.7%` | 0.945 / 0.986 | 0.596 / 0.661 | `10.4%` |
| `VC_C2_linear_peer` ($V \oplus C_2$) | 0.986 / 1.000 | 0.656 / 0.732 | 0.250 / 0.328 | `13.7%` | 0.945 / 0.973 | 0.628 / 0.699 | `12.0%` |
| `VC_C3_tf_2L_128` ($V \oplus C_3$) | 0.986 / 0.986 | 0.716 / 0.760 | 0.359 / 0.422 | `19.7%` | 0.973 / 0.986 | 0.656 / 0.716 | `20.2%` |
| `VC_C4_tf_2L_256` ($V \oplus C_4$) | **1.000 / 1.000** | 0.727 / 0.749 | 0.391 / 0.453 | `13.1%` | 0.986 / 0.986 | **0.694 / 0.754** | `17.5%` |
| **`VC_C5_tf_2L_512`** ($V \oplus C_5$) | **1.000 / 1.000** | **0.760 / 0.792** | **0.453 / 0.531** | `9.3%` | **1.000** / 0.986 | **0.683 / 0.743** | `14.8%` |
| **`VC_C6_tf_4L_512`** ($V \oplus C_6$) | **1.000 / 1.000** | **0.743 / 0.809** | **0.438 / 0.531** | **`7.1%`** | 0.973 / 0.959 | **0.689 / 0.749** | **`13.7%`** |
| `VC_C5_raw_uncalibrated` (no sphere norm) | 0.959 / 0.973 | 0.612 / 0.667 | 0.203 / 0.281 | `15.8%` | 0.781 / 0.726 | 0.415 / 0.454 | `33.9%` |

*(On `bench100`, primary `VC_C5`/`VC_C6` solves `49/49` `WB-Prop` at $s=16, 32, 64$ and `50/51` `->` `51/51` [`1.000`] `WB-Stall` by $s=32..64$; replication `VC_C6` solves `48/49` `WB-Prop` and **`51/51` [`1.000`] `WB-Stall` at $s=32$** vs. `47/51` [`0.922`] clean at $s=16$ and `43/51` [`0.843`] clean at $s=64$.)*

**Why the two checkpoints diverged in `v012` is now completely explained:**
1. **$z_H$ and $z_L$ act as mutual backup channels (`H3.1`):** Ablating only $(I-P_{\text{head}})z_H^{\text{board}}$ (`0.710 -> 0.809` at $s=64$) or only $z_L$ (`0.792 -> 0.852`) lets the loop reconstruct missing state from the unablated buffer. Erasing *both* (`V_only`) traps **`33.3%`–`36.6%`** (`gaoxin492`) and **`43.2%`–`44.8%`** (`yagizdevre`) of stalled boards in **deterministic visible-board limit cycles** that `4×` extra iterations ($s=64$) cannot escape (`0.628`–`0.650` vs. `0.896` clean, McNemar $p = 6.8\times 10^{-13}$).
2. **Primary `gaoxin492` has productive dark damping $R_t$, whereas replication `yagizdevre` has under-damped $R_t$:** On `gaoxin492`, Conditional Cache `VC_C5/C6` eliminates two-thirds of the `V_only` deficit and cuts terminal cycles from `33.3%` to `7.1%–9.3%`, but a statistically significant **`-7.7` to `-9.3` pp gap at $s=16`** ($p \le 0.0013$) and **`-8.7` to `-10.4` pp gap at $s=64`** ($p \le 3.1\times 10^{-5}$; `-18.8` pp on `S4@64`) persists because `gaoxin492`'s $R_t$ breaks remaining search cycles (`1.6%` clean cycle rate). Conversely, clean `yagizdevre` reaches `84.2%` `ever_solved_by_s64` on `WB-Stall` (`100%` on `WB-Prop`) yet **destabilizes out of solved basins or oscillates (`25.1%` cycle rate)**, degrading from `0.656` ($s=16$) to `0.590` ($s=64$). Projecting `yagizdevre` onto its Conditional Cache $\hat{H}_\theta(x, V_t)$ **filters out this oscillatory $R_t$ drift**, lifting `WB-Stall` to **`0.743`–`0.754` at $s=64`** (`+15.3` to `+16.4` pp above clean, McNemar $p = 2.8\times 10^{-6}$ to $2.5\times 10^{-5}$).

---

## 3. Measurement 2: Resume Test — Iteration Pricing in Loop-Equivalents

For every stationarity-admitted operator, transplanting $\emptyset$ (`Fresh`, $\tau_0$), $V_t$ (`V_only`), $V_t \oplus C_t$ (`VC_C6`), or $S_t = V_t \oplus C_t \oplus R_t$ (`Full`) at step $t$ into a clean run ($T_{\max} = 32$) satisfies the telescoping identity with **`0.0` numerical error** across all checkpoints and steps:
$$\min\big(t, \tau_0(x)\big) \;=\; \text{Accrued}(V_t) + \text{Price}(C_t) + \text{Price}(R_t).$$

### 3.1 `LoopedChase` (`12` `Q4` Arms, Hop-12 Resume at $t=4$)
- **$\lambda = 0.1$ (`3/3` extrapolating seeds):** All-token `V_only` (erasing orthogonal fibers of *both* cursor and edge-table tokens) fails (`0.172`), whereas all-token Conditional Cache `VC` restores **`1.000` (`3/3` seeds)**! At $t=4$, `Accrued(V_4) = -4.10` loops (all-token `V_only` requires rebuilding the edge cache), **`Price(C_4) = +5.02` loops**, and **`Price(R_4) = 0.00` loops (`0.00` on every seed)**.
- **$\lambda = 0.01$ (`2/3` extrapolating seeds):** Extrapolating seeds 1–2 give `VC = 1.000` and **`Price(R_4) = 0.00` loops** (seed 3 non-extrapolator gives `+0.62`; mean `+0.21`).
- **$\lambda = 1.0$ (`3/3` extrapolating seeds):** Seed 3 is pure head-readable (`V_only = 1.000`, `Price(C_4) = 0.00`, `Price(R_4) = 0.00`), whereas seeds 1–2 route intermediate hop state through $R_t$ (`Price(R_4) = +3.06` and `+5.46` loops; mean `+2.84`), matching the seed-level heterogeneity found in `v010`–`v012`.

### 3.2 `TRM` Sudoku Iteration Prices (`items256`, $T_{\max} = 32$)

| Checkpoint & Subset | Step $t$ | $n$ | $\tau_0$ | $\text{Accrued}(V_t)$ | $\text{Price}(C_t)$ ($p$) | $\text{Price}(R_t)$ ($p$) | Sub-channel $z_L / z_H$ | $\|R_t\|_2^2 / \|H_t\|_2^2$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Primary `WB-Prop`** | `1` | `73` | `1.81` | `+0.85` | `+0.12` (`0.26`) | **`+0.03`** (`0.76`) | `+0.01 / +0.11` | `0.182` |
| **Primary `WB-Prop`** | `2` | `73` | `1.81` | `+1.68` | `+0.03` (`0.49`) | **`+0.01`** (`1.00`) | `+0.27 / +0.03` | `0.067` |
| **Primary `WB-Prop`** | `4, 8, 12` | `73` | `1.81` | `+1.81` | **`0.00`** (`1.00`) | **`0.00`** (`1.00`) | `0.00 / 0.00` | `0.051..0.032` |
| **Primary Clean-Solved `WB-Stall`** | `2` | `158` | `4.41` | `+1.25` | `+0.37` (`0.40`) | `+0.38` (`0.44`) | `+0.23 / +0.57` | `0.167` |
| **Primary Clean-Solved `WB-Stall`** | `4` | `158` | `4.41` | `+1.38` | **`+0.91` (`0.027`)** | `+0.46` (`0.18`) | `+0.78 / +0.59` | `0.093` |
| **Primary Clean-Solved `WB-Stall`** | `8` | `158` | `4.41` | `+2.70` | `-0.11` (`0.80`) | **`+0.92` (`0.0062`)** | `+0.54 / +0.49` | `0.054` |
| **Primary Active `WB-Stall` ($\tau_0 > t$)** | `2` | `69` | `7.54` | `+0.29` | `+0.84` (`0.40`) | `+0.87` (`0.44`) | `+0.52 / +1.30` | `0.289` |
| **Primary Active `WB-Stall` ($\tau_0 > t$)** | `4` | `39` | `10.72` | `-0.97` | **`+3.44` (`0.040`)** | `+1.54` (`0.25`) | `+3.18 / +1.64` | `0.254` |
| **Primary Active `WB-Stall` ($\tau_0 > t$)** | `8` | `17` | `16.41` | `+2.29` | `-1.94` (`0.59`) | **`+7.65` (`0.0128`)** | `+4.41 / +3.24` | `0.232` |
| **Primary Active `WB-Stall` ($\tau_0 > t$)** | `12` | `8` | `22.75` | `+9.00` | `+0.75` (`0.91`) | `+2.25` (`0.56`) | `+2.12 / +4.88` | `0.225` |
| **Repl. Clean-Locked `WB-Stall`** | `2` | `113` | `16.39` | `-2.42` | `-0.50` (`0.72`) | **`+4.66` (`0.0028`)** | `+7.27 / +5.39` | `0.192` |
| **Repl. Clean-Locked `WB-Stall`** | `4` | `113` | `16.39` | `-2.09` | `+0.12` (`0.94`) | **`+4.01` (`0.0048`)** | `+3.03 / +5.95` | `0.129` |
| **Repl. Clean-Locked `WB-Stall`** | `8` | `113` | `16.39` | `-1.81` | `-1.32` (`0.40`) | **`+4.19` (`0.0144`)** | `+1.52 / +3.37` | `0.252` |
| **Repl. `bench100` Active `WB-Stall`** | `4` | `29` | `23.21` | `+5.76` | `+4.31` (`0.16`) | **`-10.03` (`0.0070`)** | `-8.10 / -2.10` | `0.128` |

- **Two-phase handoff (`C_t` then `R_t`) on actively searching stalled puzzles:** On primary `gaoxin492` puzzles still searching at $t=4$ ($\tau_0 > 4$, mean $\tau_0 = 10.72$), visible guesses $V_4$ contain temporary conflicts (`Accrued(V_4) = -0.97`), Conditional Cache $C_4$ carries **`+3.44` loops** ($p=0.040$) of constraint state, and $R_4$ carries `+1.54` loops. For the hardest puzzles still searching at $t=8$ ($n=17$, mean $\tau_0 = 16.41$), **`Price(R_8) = +7.65` outer steps** (`95% CI [+2.53, +13.12]`, $p=0.0128$), split across both $z_L$ (`+4.41`) and $z_H$ (`+3.24`).
- **Dark residual energy vs. conflict-flip count:** While single-step $\text{Price}(R_4)$ has modest Spearman correlation with clean cell-flip count (`\rho = +0.189, p=0.0044` overall; `\rho = +0.198, p=0.011` within `WB-Stall`, below the pre-registered `0.25` threshold because already-solved boards have $\text{Price}(R_t)=0$), relative dark energy $\|R_t\|_2^2 / \|H_t\|_2^2$ tracks puzzle cell-flip count strongly (`\rho = +0.819` overall / **`+0.794` within `WB-Stall` at $t=2$**, `+0.461` at $t=4$, `+0.431` at $t=8$, `+0.254` at $t=12$, all $p \le 0.0016$).

---

## 4. Measurement 3: Fork Test, C-Voting, Causal $R_t$ Swap, and `Huginn-0125` Audit

### 4.1 Reconciling Anil et al. (2022) Path Independence with Kubo et al. (2026) C-Voting (`K = 8` Forks)

| Checkpoint & Split | Noise $\sigma$ | Asymptotic Solved Pair Agreement | Peak Divergence `WB-Prop` ($t=1..8$) | Peak Divergence `WB-Stall` ($t=1..8$) | Ratio Stall / Prop | Canonical `WB-Stall` (`All`) | `K=8` **C-Voting** `WB-Stall` (`All`) | `K=8` Oracle Union `WB-Stall` (`All`) | $p_{\text{McNemar}}$ (`WB-Stall`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Primary `items256` | `0.35` | **`1.000`** | `0.0946` | `0.2150` | **`2.27×`** | `0.836` (`0.883`) | **`0.923` (`0.945`)** | `0.923` (`0.945`) | `3.05e-5` |
| Primary `items256` | `1.00` | **`1.000`** | `0.3623` | `0.4775` | `1.32×` | `0.836` (`0.883`) | **`0.940` (`0.957`)** | `0.940` (`0.957`) | **`3.81e-6`** |
| Primary `bench100` | `0.35` | **`1.000`** | `0.0953` | `0.1731` | **`1.82×`** | `1.000` (`1.000`) | **`1.000` (`1.000`)** | `1.000` (`1.000`) | `1.00` |
| Repl. `items256` | `0.35` | **`1.000`** | `0.2404` | `0.3485` | **`1.45×`** | `0.656` (`0.719`) | **`0.858` (`0.898`)** | `0.858` (`0.898`) | **`1.46e-11`** |
| Repl. `bench100` | `0.35` | **`1.000`** | `0.2698` | `0.3362` | `1.25×` | `0.922` (`0.920`) | **`1.000` (`1.000`)** | `1.000` (`1.000`) | `0.125` |

This reconciles **Anil et al. (2022)** with **Kubo et al. (2026)** quantitatively:
1. **Asymptotic path independence holds on the valid basin (`100.0%` agreement):** Because `sudoku-extreme` boards have unique solutions and `TRM` checks Sudoku constraints via re-injected clues, every pair of forks that reaches a valid fixed point converges to the **exact same board (`1.000` agreement)**.
2. **Transient path dependence is concentrated on stalled search (`2.27×` divergence):** On `WB-Prop`, perturbed starts contract rapidly to zero divergence (`0.087 -> 0.000` by $s=12$). On `WB-Stall`, symmetry-breaking guesses split forks onto distinct search trajectories (`0.215` peak divergence at $\sigma=0.35$; `0.478` at $\sigma=1.00$).
3. **Label-free C-voting equals the `K=8` oracle union (`100.0%` match):** Unsolved forks remain in conflict and exhibit depressed visible top-1 margin/confidence, so Kubo et al.'s parameter-free confidence selector picks a solved fork **whenever at least one of the 8 forks solves the board**, lifting primary `items256` overall solve from **`88.3% -> 95.7%`** (`WB-Stall`: **`83.6% -> 94.0%`**, $+19$ stalled boards, $p=3.81\times 10^{-6}$) and replication `items256` from **`71.9% -> 89.8%`** (`WB-Stall`: **`65.6% -> 85.8%`**, $+37$ stalled boards, $p=1.46\times 10^{-11}$) and `bench100` to **`100/100`**!

### 4.2 Causal $R_t$ Swap & Refuted-State Revisit Granularities (`items256` `WB-Stall`)

At matched steps $t \in \{3, 5, 8\}$, all continuations share identical step-$t$ visible logits $V_t^{(A)}$ and conditional cache $C_t^{(A)}$:
- **Causal trajectory steering (`Swap_R_B` vs. `Zero_R` and `Swap_R_C`):** On `WB-Prop`, swapping $R_t^{(B)}$ is causally null ($p = 0.32\text{--}1.00$). On `items256` `WB-Stall`, `Swap_R_B` significantly steers step-$(t+1)$ cell commitments toward donor $B$ above `Zero_R` at **all three steps on both checkpoints**:
  - Primary `gaoxin492`: **`+4.50` pp** ($p=0.0008$) at $t=3$, **`+3.71` pp** ($p=0.0002$) at $t=5$, **`+2.59` pp** ($p=0.0136$) at $t=8$ (vs. unrelated-board `Swap_R_C`: `+3.22` pp [$p=0.059$], `+1.52` pp [$p=0.24$], `+1.26` pp [$p=0.38$], indicating that part of `Swap_R_B` on `gaoxin492` acts by kicking $V_t^{(A)}$ off its local trajectory while a chart-local component aligns with $B$).
  - Replication `yagizdevre`: **`+13.50, +11.84, +10.62` pp** above `Zero_R` ($p=0.0002$ at all three steps) and **`+14.13, +12.28, +11.69` pp** above `Swap_R_C_unrelated` (**$p=0.0002$ at all three steps**; `bench100`: `+10.58, +8.36, +8.34` pp, $p=0.0002$).
- **Refuted-state revisit signature:** Rewinding dark memory to stale step-1 state (`Rewind_R_A1`) significantly increases revisits of $A$'s already-refuted 9-cell Sudoku units (`+0.81, +1.74, +1.88` pp on primary, $p = 0.0056, 0.0002, 0.0002$; `+1.61, +2.20` pp on replication at $t=3,5$, $p=0.0002$) and wrong cell assignments (`0.133 -> 0.176`, `0.115 -> 0.191`, `0.094 -> 0.191` on primary). By contrast, setting $R_t = 0$ (`Zero_R_persist`) does *not* increase local 9-cell unit revisits (`-0.59` to `+0.31` pp on primary; `-2.24` to `-7.10` pp on replication because `VC` denoises `yagizdevre`), even though over 64 steps `VC` raises *global 81-cell exact limit cycles* on primary `gaoxin492` from `1.6%` to `7.1%–9.3%`. Thus $R_t$ functions as **global trajectory momentum / cycle-breaking damping** rather than a local symbolic tabu table.

### 4.3 Supplementary `Huginn-0125` Path-Independence & Cache Audit (`32` Prompts $\times$ `K = 4` Random Starts, `32` Steps)
- **Exact stationarity:** State-transplant $L_\infty$ diff `= 0.0`.
- **Rapid contraction to path-independent cache:** Starting from independent Gaussian latent initializations (`path_var_share = 1.007`, cross-fork cosine `0.0013` at $t=0$), `Huginn-0125`'s re-injected 4-layer core contracts path-dependent variance share monotonically to **`0.389` ($t=1$) $\to$ `0.202` ($t=2$) $\to$ `0.089` ($t=4$) $\to$ `0.029` ($t=8$) $\to$ `0.0041` ($t=16$) $\to$ `0.00057` ($t=32$)** (`99.94%` path-independent cache; cross-fork latent cosine `0.9997`, top-1 agreement `1.000` at $t=16$).
- **Fork swap & Resume price:** At $t=8$, leave-one-out cross-fork `CacheMean` matches clean next-step top-1 token predictions **`100.0%` (`32/32`)** and donor-$R_8$ swap flips only `3.1%` (`1/32`); $t=4$ Resume Test prices `Accrued(Cache_4) = +2.50` loops vs. `Price(R_4) = +0.50` loops.
