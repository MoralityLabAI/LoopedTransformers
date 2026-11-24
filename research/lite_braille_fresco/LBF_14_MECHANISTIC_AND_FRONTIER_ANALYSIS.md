# LBF-14: Continuous Recurrent Tensor Mechanics, Gate-Opening Causality, and Bidirectional Frontier Geometry

**Date:** 2026-11-24 · **Parent receipts:** [LBF-13 results](LBF_13_BUDGET_FRESCO_RESULTS.md) · [LBF-14 results](LBF_14_BIDIRECTIONAL_FRESCO_RESULTS.md) · **Machine-readable ledger:** [`results/lbf14/mechanistic_diagnostics.json`](results/lbf14/mechanistic_diagnostics.json) · **Reproducer:** [`lbf14_mechanistic_analysis.py`](lbf14_mechanistic_analysis.py).

---

## 1. Continuous Recurrent Tensor Analysis (LBF-14 Boundary Archive)

We analyze all 256 predecision execution-boundary records in `raw_boundary_tensors.pt.gz` (`128` decisions for `standard` gate bias $-6.0$; `128` for `opened` gate bias $-2.0$; seeds `107` and `109`; 2 recurrent `ab` passes per decision) together with the four frozen 57,475-parameter checkpoints.

### 1.1 Gate strengths, learned parameter drift, and message-injection norms

| Metric (Pass 2 unless noted, $n=128$/arm) | `standard` ($\text{bias}=-6$) | `opened` ($\text{bias}=-2$) | Ratio (`opened`/`standard`) |
|---|---:|---:|---:|
| `gate_a` sigmoid strength (Pass 1 mean $\pm$ SD) | $0.00240 \pm 0.00028$ | $0.12501 \pm 0.00314$ | $52.0\times$ |
| `gate_a` sigmoid strength (Pass 2 mean $\pm$ SD) | $0.00317 \pm 0.00126$ | $0.14893 \pm 0.00495$ | $47.0\times$ |
| `gate_b` sigmoid strength (Pass 2 mean $\pm$ SD) | $0.00291 \pm 0.00068$ | $0.11402 \pm 0.00881$ | $39.2\times$ |
| Learned `gate_a.gate.bias` shift $\Delta b$ (107 / 109) | $+0.0154$ / $+0.0181$ | $+0.0056$ / $+0.0052$ | — |
| Learned `gate_a.gate.weight` Frobenius norm (107 / 109) | $0.5071$ / $0.6959$ | $0.6796$ / $0.9295$ | $1.38\times$ |
| `gate_a` message contribution RMS $\|\text{gate\_a}(z,\text{msg})\|_{\text{rms}}$ | $0.00477 \pm 0.00329$ | $0.33152 \pm 0.17729$ | $69.5\times$ |
| `gate_b` message contribution RMS $\|\text{gate\_b}(a,\text{msg})\|_{\text{rms}}$ | $0.00246 \pm 0.00109$ | $0.31199 \pm 0.22352$ | $126.6\times$ |
| Relative injection norm $\|\text{gate\_a}(z,\text{msg})\|_F / \|z\|_F$ | $0.158\%$ | $10.849\%$ | $68.5\times$ |
| Relative injection norm $\|\text{gate\_b}(a,\text{msg})\|_F / \|a\|_F$ | $0.068\%$ | $7.650\%$ | $112.9\times$ |

Because `MessageGate` initializes `gate.weight = 0` and `gate.bias = -6`, the sigmoid derivative $\sigma(-6)(1-\sigma(-6)) = 0.00247$ attenuates gate gradients by $400\times$ across the 360 AdamW updates, trapping `standard` below $0.16\%$ relative state norm. Setting initial bias to $-2$ ($\sigma(-2) = 0.1192$, derivative $0.1050$) restores physical message injection to $10.85\%$ (`gate_a`) and $7.65\%$ (`gate_b`) of residual block input norm.

### 1.2 Effective rank, slot collapse, and residual-bank collinearity

| Representational geometry metric | `standard` | `opened` | Mechanistic interpretation |
|---|---:|---:|---|
| Outgoing bank slot effective rank $\operatorname{erank}_{8\times 32}$ (Pass 1 $\to$ Pass 2) | $1.112 \to 1.054$ | $1.075 \to 1.034$ | Near-complete rank-1 collapse across the 8 nominal workspace slots |
| Outgoing bank slot stable rank $\|B\|_F^2 / \|B\|_2^2$ (Pass 1) | $1.019$ | $1.012$ | $>98\%$ of spectral energy resides in a single slot mode |
| Off-diagonal cosine similarity across the 8 bank slots (Pass 1) | $0.9779 \pm 0.0047$ | $0.9861 \pm 0.0040$ | Explains why cyclic slot `roll` is nearly invariant |
| Coda slot effective rank $\operatorname{erank}_{8\times 32}$ (Pass 1 $\to$ Pass 2) | $1.067 \to 1.038$ | $1.047 \to 1.024$ | Symmetric unmasked self-attention homogenizes slots |
| Population effective rank $\operatorname{erank}_{128\times 32}$ of anchor `coda[:,0,:]` (P1 $\to$ P2) | $2.017 \to 1.979$ | $1.776 \to 1.804$ | Low-dimensional cross-state summary representation |
| Directional alignment $\cos(\text{bank}_1, z_1)$ entering Pass 2 | $0.9789$ (med $0.9946$) | $0.9863$ (med $0.9942$) | `LN((a_1+b_1)/2)` is nearly collinear with residual $z_1 = b_1$ |
| Directional alignment $\cos(\Delta^{\text{msg}}\text{coda}, \Delta^{\text{res}}\text{coda})$ | $0.5090 \pm 0.0507$ | $0.4622 \pm 0.0441$ | Bank injection largely reinforces the existing residual drift |

### 1.3 Inter-pass displacement and candidate margin dynamics

Decomposing total Pass-1-to-Pass-2 change $\Delta^{\text{tot}} = y^{(2)}_{\text{normal}} - y^{(1)}$ into **pure residual-state recurrence** $\Delta^{\text{res}} = y^{(2)}_{\text{zero}} - y^{(1)}$ (since `proj` has no bias, `gate(x, 0) = 0` identically) and **causal message-bank contribution** $\Delta^{\text{msg}} = y^{(2)}_{\text{normal}} - y^{(2)}_{\text{zero}}$, plus **slot-addressing contribution** $\Delta^{\text{slot}} = y^{(2)}_{\text{normal}} - y^{(2)}_{\text{roll}}$:

| Displacement / margin metric ($n=128$) | `standard` | `opened` |
|---|---:|---:|
| Total coda RMS displacement $\|\Delta^{\text{tot}}\text{coda}\|_{\text{rms}}$ | $1.4711 \pm 0.1854$ | $1.9522 \pm 0.1057$ |
| Residual-only coda RMS displacement $\|\Delta^{\text{res}}\text{coda}\|_{\text{rms}}$ | $1.4683 \pm 0.1838$ | $1.5769 \pm 0.1871$ |
| Message-bank coda RMS displacement $\|\Delta^{\text{msg}}\text{coda}\|_{\text{rms}}$ | $0.0058 \pm 0.0036$ | $0.5753 \pm 0.3930$ |
| Slot-addressing coda RMS displacement $\|\Delta^{\text{slot}}\text{coda}\|_{\text{rms}}$ | $0.0009 \pm 0.0004$ | $0.0298 \pm 0.0044$ |
| Outcome-logit max shift: Total / Residual / Message / Slot (mean) | $0.02571$ / $0.02567$ / $0.00015$ / $0.00006$ | $0.04319$ / $0.03714$ / $0.00864$ / $0.00111$ |
| Outcome-logit max shift: Message (`zero`) / Slot (`roll`) (max) | $0.00097$ / $0.00018$ | $0.02688$ / $0.00633$ |
| Candidate probability margin $|m|$ (Pass 1 $\to$ Pass 2 mean) | $0.02927 \to 0.02778$ ($\Delta = -0.00149$) | $0.02379 \to 0.02189$ ($\Delta = -0.00190$) |
| Top-choice flips: P1$\to$P2 (`normal`) / `normal` vs `zero` / `normal` vs `roll` | $4$ / $0$ / $0$ | $3$ / $4$ / $0$ |

---

## 2. Causal Sensitivity and Transmission of Task-Relevant Information

To separate residual-state recurrence from message-bank content and slot addressing, we evaluate all four checkpoints on the complete set of **146 exact counterfactual calibration frames** along smaller-layer trajectories (`49` non-tied frames with $C_0 \ne C_1$, `3` mixed-closure frames) and in full **closed-loop calibration search** (`32` roots) across six causal regimes:
1. `pass1`: single pass ($k=1$, neither residual nor message recurrence).
2. `zero`: Pass 2 with zeroed message bank ($B=0$, **pure residual-state recurrence**).
3. `msg_only_frozen_resid`: Pass 2 with residual state reset to `prelude(z)` while keeping intact message bank $B_1$ (**message bank isolated from residual drift**).
4. `roll`: Pass 2 with norm-matched cyclic slot permutation `B_1.roll(1, dims=1)` (**destroys slot identity, preserves global vector direction**).
5. `ortho_rot`: Pass 2 with isometric feature-space rotation $B_1 Q$, $Q \in O(32)$ (**preserves exact Frobenius/slot norms, scrambles feature directions**).
6. `normal`: intact residual + message-bank recurrence.

| Checkpoint | Regime | Frame MSE | Non-tied Pearson $r(m, m^*)$ | Non-tied sign acc | Mean signed margin gain $\operatorname{sign}(m^*)(m - m^{(1)})$ | Closed-loop closures | Penalized work |
|---|---|---:|---:|---:|---:|---:|---:|
| `107_standard` | `pass1` | $0.13517$ | $0.1956$ | $59.18\%$ | $0.00000$ | **22/32** | $73.81$ |
| `107_standard` | `zero` (resid only) | $0.13529$ | $0.1749$ | $59.18\%$ | $-0.00049$ | 21/32 | $74.22$ |
| `107_standard` | `msg_only_frozen_resid` | $0.13517$ | $0.1957$ | $59.18\%$ | $+0.00000$ | **22/32** | $73.81$ |
| `107_standard` | `roll` / `ortho_rot` / `normal` | $0.13529$ | $0.1750$ | $59.18\%$ | $-0.00049$ | 21/32 | $74.22$ |
| `107_opened` | `pass1` | $0.13612$ | $0.2262$ | $51.02\%$ | $0.00000$ | 21/32 | $74.97$ |
| `107_opened` | `zero` (resid only) | $0.13642$ | $0.2314$ | $51.02\%$ | $-0.00004$ | 21/32 | $74.97$ |
| `107_opened` | `msg_only_frozen_resid` | $0.13617$ | $0.2418$ | $51.02\%$ | $+0.00143$ | 21/32 | $75.47$ |
| `107_opened` | `roll` (slot-rotated) | $0.13641$ | $0.2406$ | $51.02\%$ | $+0.00079$ | 21/32 | $74.97$ |
| `107_opened` | `ortho_rot` ($O(32)$) | $0.13637$ | $0.2344$ | $51.02\%$ | $+0.00030$ | 21/32 | $74.97$ |
| `107_opened` | `normal` (intact) | $0.13642$ | $0.2402$ | $51.02\%$ | $+0.00074$ | 21/32 | $75.47$ |
| `109_standard` | `pass1` | $0.14780$ | $0.2246$ | $59.18\%$ | $0.00000$ | 21/32 | $74.03$ |
| `109_standard` | `zero` / `roll` / `normal` | $0.14894$ | $0.1985$ | $57.14\%$ | $-0.00212$ | 21/32 | $74.03$ |
| `109_opened` | `pass1` | $0.13701$ | $0.1764$ | $55.10\%$ | $0.00000$ | 20/32 | $75.28$ |
| `109_opened` | `zero` (resid only) | $0.13654$ | $0.2041$ | $59.18\%$ | $+0.00120$ | 20/32 | $74.34$ |
| `109_opened` | `msg_only_frozen_resid` | $0.13662$ | $0.1825$ | $57.14\%$ | $+0.00027$ | 20/32 | $74.34$ |
| `109_opened` | `roll` (slot-rotated) | $0.13621$ | $0.2131$ | $59.18\%$ | $+0.00184$ | 20/32 | $74.34$ |
| `109_opened` | `ortho_rot` ($O(32)$) | $0.13671$ | $0.2068$ | $61.22\%$ | $+0.00132$ | 20/32 | $74.34$ |
| `109_opened` | `normal` (intact) | $0.13614$ | $0.2154$ | $59.18\%$ | $+0.00197$ | 20/32 | $74.34$ |

**Causal conclusions (Task 3):**
1. **Residual-state recurrence dominates pass-to-pass change:** In `standard`, 99.8% of inter-pass displacement is residual drift (`zero` vs `pass1`), which slightly degrades non-tied utility alignment on both seeds (`107_standard` `pass1` closes 22/32 vs 21/32 at Pass 2).
2. **Opening gates transmits a weak global rank-1 feature vector, not structured slot memory:** In `opened`, intact `normal` beats `zero` on non-tied frame correlation ($0.2402$ vs $0.2314$ on seed 107; $0.2154$ vs $0.2041$ on seed 109) and `ortho_rot` attenuates that bump ($0.2344$; $0.2068$), confirming feature-direction sensitivity. However, **`roll` matches `normal` identically** ($0.2406$; $0.2131$) because all 8 slots carry virtually the same vector ($\cos = 0.986$), and closed-loop proof closure never improves (`41/64` across `normal`, `zero`, `roll`).
3. **Root architectural cause:** In `PPT-B` (`ppt_gym/core.py`), individual candidates `cand_emb` never enter the recurrent loop (`z` only pools `cand_emb.mean(0)` into slot 4), so recurrent passes can only shift a single shared 32-d readout vector `read[:,0,:]` applied bilinearly to static byte-bag candidate embeddings.

---

## 3. Why Symbolic Bidirectional Search Outperforms Learned A* Ranking (Task 4)

We evaluate exact graph distances $d^*(s, t)$, admissible lower-bound shells, branching asymmetry, opposite-frontier meeting surfaces, and budget survival curves across LBF-13 calibration (`24` roots, seed `20261207`) and LBF-14 calibration/train (`32` + `48` roots, seed `20261209`), without touching sealed splits.

### 3.1 Exact decomposition of the A*-to-Bidirectional gap (`WORK = 128`, `DEPTH = 16`)

| Search policy / diagnostic control | LBF-13 Cal (`24`) | LBF-14 Cal (`32`) | LBF-14 Train (`48`) | Mechanism isolated |
|---|---:|---:|---:|---|
| Unidirectional forward BFS (`fwd_bfs`) | 8/24 (33.3%) | 10/32 (31.2%) | 15/48 (31.2%) | Blind single-ball volume $O(b^{d^*})$ |
| Unidirectional backward BFS (`bwd_bfs`) | 6/24 (25.0%) | 10/32 (31.2%) | 14/48 (29.2%) | Reverse single-ball volume |
| Classical `astar` ($f = g + \lceil h/3 \rceil$) | 11/24 (45.8%) | 19/32 (59.4%) | 28/48 (58.3%) | Admissible 3-letter Hamming bound |
| **`oracle_astar_both` (exact $d^*(w,t)$ tie-break!)** | **11/24 (45.8%)** | **19/32 (59.4%)** | **28/48 (58.3%)** | **Exact ceiling of ANY tie-breaker inside $f$** |
| Anti-balanced bidirectional BFS (`bidir_larger`) | 8/24 (33.3%) | 10/32 (31.2%) | 15/48 (31.2%) | Loses branching-asymmetry protection |
| Alternating-depth bidirectional BFS (`bidir_depth`) | 15/24 (62.5%) | 22/32 (68.8%) | 29/48 (60.4%) | Radius halving $d^*/2$ without size balance |
| Smaller-layer bidirectional BFS (`bidir_smaller`) | **17/24 (70.8%)** | **23/32 (71.9%)** | **30/48 (62.5%)** | Radius halving + branching-asymmetry balance |
| Node-level cross-frontier oracle (`node_bidir_oracle`) | **24/24 (100%)** | **31/32 (96.9%)** | **46/48 (95.8%)** | Eliminates intra-layer cousin waste |

### 3.2 Quantitative contribution of the four structural factors

1. **Frontier geometry & admissible $f$-shell lock-in (explains 100% of LBF-13's 11/24 ceiling):**
   On LBF-13 calibration, mean shortest proof length is $d^* = 5.58$ steps while the initial admissible lower bound $\lceil d_H(s, t)/3 \rceil$ averages $2.17$ — an underestimate of **$3.42$ steps** ($2.75$ on LBF-14 cal; $3.25$ on LBF-14 train). Because `astar` and all LBF-13 learned arms lock the primary heap key to $f(w) = g(w) + \lceil d_H(w, t)/3 \rceil$, every state with $f(w) < f_{\max}(\text{witness})$ must be popped before the proof path can advance. **Even `oracle_astar_both` using exact shortest-path distance $d^*(w, t)$ for both frontier ties and sibling ordering solves 11/24 on LBF-13 (saving only 4 successors: 2,027 vs 2,031), 19/32 on LBF-14 cal, and 28/48 on LBF-14 train** — solving $100\%$ of $d^* \le 4$ roots and $0\%$ of $d^* \ge 8$ roots (`0/13` for $d^* \ge 6$ in LBF-13).
2. **Radius halving & multi-target meeting opportunities (+16.7 pp to +37.5 pp over unidirectional BFS):**
   Alternating bidirectional expansion (`bidir_depth`) jumps from `8/24` (`fwd_bfs`) to `15/24` (+29.2 pp) on LBF-13 and `10/32` to `22/32` (+37.5 pp) on LBF-14 cal. Beyond $O(b^{d^*/2})$ volume reduction, every examined successor in `bidir_smaller` tests $O(1)$ membership against an average of **11.62** (LBF-13 cal), **10.71** (LBF-14 cal), and **11.71** (LBF-14 train) discovered opposite-frontier states simultaneously, compared with **1** target state in unidirectional A*.
3. **Branching asymmetry (+8.3 pp over `bidir_depth`, +37.5 pp to +40.6 pp over `bidir_larger`):**
   Mean endpoint branching is $\bar{b}_+ = 4.72\text{–}4.96$ vs $\bar{b}_- = 4.79\text{–}5.19$ with mean initial asymmetry $|b_+ - b_-|/(b_+ + b_-) = 11.5\%\text{–}13.9\%$ (up to $\pm 5$ neighbors/word), compounding into $3\times\text{–}7\times$ layer size ratios by depth 2 (e.g., `[49, 15]`, `[7, 35]`). Expanding the smaller frontier (`bidir_smaller`) beats `bidir_depth` (`17/24` vs `15/24`; `23/32` vs `22/32`) and more than doubles anti-balanced `bidir_larger` (`17/24` vs `8/24`; `23/32` vs `10/32`).
4. **Remaining search budget & intra-layer cousin waste (identifies the exact opportunity for LBF-15):**
   Closure under `bidir_smaller` scales across budgets $B \in \{16, 32, 64, 96, 128, 256, 512\}$ from `2 -> 4 -> 11 -> 13 -> 17 -> 22 -> 24` (LBF-13 cal) and `5 -> 8 -> 18 -> 20 -> 23 -> 27 -> 30` (LBF-14 cal). Crucially, among the **9** LBF-14 calibration roots and **18** LBF-14 train roots unsolved by `bidir_smaller` at $B=128$, **8/9 (88.9%)** and **17/18 (94.4%)** already contain discovered opposite-frontier pairs $(u_i, v_j) \in \mathcal{F}_0 \times \mathcal{F}_1$ separated by $\le 3$ rewrites (`3/9` and `6/18` separated by **1 single rewrite**!), which whole-layer FIFO expansion fails to reach before exhausting `WORK = 128` on unpromising cousin words in the same layer. Node-level cross-frontier selection raises oracle-feasible closure at $B=128$ from `23/32` to `31/32` and `30/48` to `46/48`.
