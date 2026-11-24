# LBF-18: Guarded Abstract-Interpreter Fresco, Incremental Caching, and Independent Confirmation Specification

Stacked on merged `main` commit `3d8243818858c19c5031c36e2bf7e328f831cf25` ([PR #343](https://github.com/MoralityLabAI/LoopedTransformers/pull/343)), incorporating [PR #348](https://github.com/MoralityLabAI/LoopedTransformers/pull/348) (`404e1483d53e9394fef12f1a705407a87bfe451b`) and Fable's independent referee audit bundle preserved verbatim at signed commit `8102fb2190e3c0d30a5a7fad93694d4eaa0af6bd` (`origin/research/lbf-independent-referee-review-20261124`). All 24 historical LBF-17 checkpoints in [`results/lbf17/manifest.json`](results/lbf17/manifest.json) and all historical LBF-16 checkpoints in [`results/lbf16/manifest.json`](results/lbf16/manifest.json) remain untouched.

Standalone zero-axiom Lean 4.19.0 formalization: [`lean/LBF18IncrementalAndCollisions.lean`](lean/LBF18IncrementalAndCollisions.lean) (**`22/22` theorems compiled with zero axioms** — no `propext`, no `Classical.choice`, no `Quot.sound`).

---

## 1. Guarded Root-Suppression Theorem (`Theorem A`) and Conservative Mask-Guard Comparison

### 1.1 Referee Counterexample `R*` Against Legacy LBF-17 Root-Suppression
In LBF-17 (`symbolic_prov4_adj2` and `FastProvenanceBigramCell`), Pass-2 (`pass_idx == 1`) transitions firing oriented rule $(u \to v) \in R^{\leftrightarrow}$ of width $\ell = |u| = |v|$ at position $p$ inside congruence fiber $f \in \{0, \dots, F_p - 1\}$ were suppressed whenever $p \bmod F_p = f$ and $v = w_0[p : p + \ell]$. On Fable's mixed-width counterexample
$$
R^* = \{\texttt{abc} \leftrightarrow \texttt{def},\ \texttt{ab} \leftrightarrow \texttt{de},\ \texttt{bf} \leftrightarrow \texttt{cc}\}, \qquad w_0 = \texttt{abc}\cdots,
$$
Pass 1 fires $\texttt{abc} \to \texttt{def}$ at $p = 0$ into Fiber $0$, leaving `f` modified at coordinate $2$; Pass 2 must fire $\texttt{de} \to \texttt{ab}$ at $p = 0$ ($\ell = 2 < 3$) to reach $\texttt{abf}\cdots$ (not $w_0$) before Pass 3 fires $\texttt{bf} \to \texttt{cc}$ at $p = 1$ to reach $\texttt{acc}\cdots$. Unguarded suppression drops $\texttt{de} \to \texttt{ab}$, creating a concrete 3-step false negative (`referee_rstar_legacy_unsound_and_both_guards_sound`).

### 1.2 Theorem A (Guarded Congruence-Fibered `Adj2` Transfer Soundness)
Let $\Sigma$ be a finite alphabet, $L \ge 1$, $R^{\leftrightarrow}$ a length-preserving rewrite system with rule widths in $[2, \ell_{\max}]$, and $w_0 \in \Sigma^L$. Define Pass-1 exact fiber seeding into $F_p$ congruence fibers by routing every 1-step rewrite at redex origin $p_1$ into fiber $f = p_1 \bmod F_p$ followed by exact bigram abstraction $\alpha_2$ and Viterbi trellis reduction, and retain $\{w_0\}$ in an explicit root slot (`slot 0`). At Pass 2 (`step = 1`) only, compare two guards for suppressing $(u \to v)$ of width $\ell$ at position $p$ in fiber $f$ when $p \bmod F_p = f$ and $v = w_0[p : p + \ell]$:

1. **Fable's Modulus + Max-Origin-Length Guard (`fable_len_guard`, Primary Guard):**
   Suppress iff
   $$
   F_p \ge \ell_{\max} \quad \text{and} \quad \ell_{\text{orig}}^{\max}(w_0, p) := \max\bigl(\{0\} \cup \{|u'| : (u' \to v') \in R^{\leftrightarrow},\ w_0[p : p + |u'|] = u'\}\bigr) \le \ell.
   $$
   *Proof sketch (`fable_suppressed_pass2_returns_root_or_stays` & `theorem_A_fable_guarded_prov_adj2_soundness`):* Let $w_1$ be any concrete 1-step successor of $w_0$ placed in fiber $f$ via redex $(u_1 \to v_1)$ of width $\ell_1 \le \ell_{\max}$ at origin $p_1 \equiv f \equiv p \pmod{F_p}$, and suppose $w_1 \to_{(u \to v) @ p} w_2$.
   - **Case $p_1 = p$:** Since $w_0[p : p + \ell_1] = u_1$, the guard implies $\ell_1 \le \ell_{\text{orig}}^{\max}(w_0, p) \le \ell$. Outside $[p, p + \ell_1) \subseteq [p, p + \ell)$, $w_1$ equals $w_0$; inside $[p, p + \ell)$, $(u \to v)$ writes $v = w_0[p : p + \ell]$. Hence $w_2 = w_0$, which is already retained in `slot 0`.
   - **Case $p_1 \neq p$:** Since $p_1 \equiv p \pmod{F_p}$ and $F_p \ge \ell_{\max} \ge \max(\ell_1, \ell)$, the intervals $[p_1, p_1 + \ell_1)$ and $[p, p + \ell)$ are strictly disjoint (`mod_collision_disjoint_intervals`). Thus $w_1[p : p + \ell] = w_0[p : p + \ell]$. Because $(u \to v)$ applies to $w_1$ at $p$ and writes $v = w_0[p : p + \ell]$, we have $u = w_1[p : p + \ell] = w_0[p : p + \ell] = v$, so $w_2 = w_1$ is a stutter already present in fiber $f$.
   Induction with `localAdj2_unsuppressed_firing_sound` establishes zero-axiom soundness for all horizons $k \ge 0$.

2. **PR #348 Conservative Changed-Coordinate Mask Guard (`pr348_mask_guard`, Comparator Guard):**
   Let $D_f(w_0) = \{i \in \{0, \dots, L-1\} : \exists w_1 \in S_f(w_0),\ w_1[i] \neq w_0[i]\}$ be the bitmask of coordinates modified by any Pass-1 successor in fiber $f$. Suppress iff
   $$
   D_f(w_0) \subseteq [p, p + \ell).
   $$
   *Soundness & Comparison (`theorem_A_pr348_mask_guarded_prov_adj2_soundness`, `referee_modulus_assumption_necessary_for_len_guard`, `referee_mod_collision_len_guard_sharper_than_mask_guard`):*
   - `pr348_mask_guard` requires no modulus hypothesis ($F_p \ge 1$). Conversely, without the explicit $F_p \ge \ell_{\max}$ gate, `len_guard` fails on $w_0 = \texttt{abcd}$, $R = \{\texttt{abc} \leftrightarrow \texttt{efd},\ \texttt{dd} \leftrightarrow \texttt{cd}\}$ at $F_p = 2 < 3$; our Rust and Python `fable_len_guard` enforce `(Fp >= actual_max_rlen)` explicitly at runtime so stress tracks with $\ell_{\max} > F_p$ remain 100% sound.
   - Under modulo collisions ($w_0 = \texttt{abccab}$, $R = \{\texttt{ab} \leftrightarrow \texttt{de}\}$, $F_p = 4$), $D_0 = \{0, 1, 4, 5\} \nsubseteq [0, 2)$, so `pr348_mask_guard` declines suppression while `fable_len_guard` suppresses. On exact word-set fibers `fable_len_guard` is strictly sharper (excluding $w_0$ from Fiber 0), whereas on $\alpha_2$ (`Adj2`) fibers when $F_p \ge \ell_{\max}$, any second colliding redex $p_1 \neq p$ already deposits $w_0$'s arcs on $[p - 1, p + \ell]$ into Fiber $f$ at Pass 1, rendering `fable_len_guard` and `pr348_mask_guard` bit-identical on `Adj2` states while both cleanly exclude the LBF-17 chimera $\texttt{cc}$ on $w_0 = \texttt{abc}$ (`lbf17_cex1_chimera_excluded_under_both_guards`).
   - Single-rule horizon overshoot (`referee_ce5_single_rule_horizon_overshoot`): on $w_0 = \texttt{abeeab}$ ($L = 6$) with $\{\texttt{ab} \leftrightarrow \texttt{cd}\}$ and $F_p = 4$, commuting rewrites at $p \in \{0, 4\} \equiv 0 \pmod 4$ merge in Fiber 0 at Pass 1, admitting 2-step $\texttt{cdeecd}$ at Pass 1 (sound overapproximation, bounded precision loss).

---

## 2. Exact Decoupled Boolean Support (`Theorem B`) & Float32 Margin Floor (`CE-3`)

In finite IEEE-754 `float32`, repeated multiplicative rule firings and soft trellis products could in principle underflow positive continuous activations toward `0.0` at large depth $K$ (`CE-3`). LBF-18 eliminates floating-point soundness coupling by construction:

1. **Decoupled Pure-Boolean `u64` Support State:** Both the compiled Rust kernel (`lbf18_rel_prov_word_envelopes_f32` in [`rust_trellis/src/lib.rs`](rust_trellis/src/lib.rs)) and the PyTorch reference (`IncrementalFrescoController` in [`lbf18_incremental_fresco.py`](lbf18_incremental_fresco.py)) maintain an exact `u64` Boolean bitset tensor `supp_bits_u64` of shape `(K, Fp + 1, L - 1)` computed purely via integer bitwise shifts and masks under `Theorem A`, **strictly independent of `float32` neural weights and glyph brightness**.
2. **Exact-Mask-Guided Soft Trellis & Online Margin Monitor:** In guarded modes, soft continuous trellis pruning masks inactive arcs using the exact Boolean `u64` bitset while tracking `min_pos_f32` — the minimum `float32` channel maximum across all exact-Boolean-supported cells — against `margin_floor = 1e-6` (with optional fail-closed exception `fail_on_margin_underflow=True`). Even if `bi_cell` weights are scaled by $10^{-9}$, `exact_boolean_pos1_support` retains `0` false negatives.

---

## 3. Three-Stage Incremental Factorization (`Stage A / B / C`)

Let $U_t = (u_1, \dots, u_{m_0})$ and $V_t = (v_1, \dots, v_{m_1})$ be the forward and backward frontiers at step $t$ under checkpoint $\theta$, guard mode $g \in \{\texttt{fable\_len\_guard}, \texttt{pr348\_mask\_guard}, \texttt{legacy}, \texttt{none}\}$, and intervention mode $\mu$:

1. **Stage A — Word-Level Multi-Pass Envelope Cache ($\mathcal{C}_{\text{word}}$):** Keyed by
   $$\kappa_{\text{word}}(w) = \bigl(\mathrm{sha256}(\theta),\ \mathrm{sig}(\theta),\ g,\ \mathrm{canon}(R),\ \pi_{\text{env}}(\mu),\ K,\ w\bigr).$$
   Evaluates `pos_cell`, guarded `bi_cell`, `supp_bits_u64`, and static overlap factors **only** on newly discovered frontier words $\Delta U_t \cup \Delta V_t$ (`incremental_word_map_equiv`).
2. **Stage B — Static Pairwise Subterm Ribbon Cache ($\mathcal{C}_{\text{pair}}$):** Keyed by
   $$\kappa_{\text{pair}}(u, v) = \bigl(\mathrm{sha256}(\theta),\ \mathrm{sig}(\theta),\ g,\ \mathrm{canon}(R),\ \pi_{\text{env}}(\mu),\ K,\ u,\ v\bigr).$$
   Caches static `ribbon_in -> Conv1d -> GELU -> mm/max pool` vectors (`3 × 76` floats per pair) and static $(d_H, \lceil d_H / 3 \rceil, \mathrm{span})$, evaluating only newly formed frontier pairs (`incremental_pair_map_equiv`).
3. **Stage C — Dynamic Cross-Frontier Axial Competition:** Recomputes dynamic path lengths $(g_0, g_1)$, depth feasibility `tot_lb > D`, budget fraction $b_{\text{rem}}/B$, frontier log-ratio $\Delta \log m$, `prev_pot` persistence, and `FactoredAxialFrescoBlock` competition live at every decision step. Cache reuse is rejected fail-closed (`keyed_cache_rejects_mismatch`) upon any weight mutation or spatial scramble mode (`scramble_window`, `scramble_ribbon`).

---

## 4. Decisive Symbolic Controls, Hybrid Controller, and Adaptive Pass Halting

To separate the contribution of congruence-fibered abstract propagation from learned neural recurrence, LBF-18 evaluates all arms under identical frontier expansion rules:

- **`absint1_mm` (`FastAbsIntOracle`, $K=1$, `0` learned parameters):** Admissible meet-in-the-middle lower bound $\hat{d}_1(u, v) \in \{0, 1, 2, 3\}$ from 1-hop `u64` bitset envelopes plus residual mismatch lower bound.
- **`absint_mm` (`FastAbsIntOracle`, $K=3$, `0` learned parameters):** Parameter-free 3-hop guarded `prov4_adj2` abstract-interpretation lower bound $\hat{d}_3(u, v) \in \{0, \dots, 7\}$.
- **`exact_mm` (`ExactBallOracle`, radius $r = 3$, `0` learned parameters):** Exact concrete 3-ball intersection oracle (counting both logical frontier expansions and concrete ball successor generations).
- **`sym_prov_readout` (`53,069` frozen LBF-17 params, guarded `u64` bitsets):** Exact 3-hop symbolic `prov4_adj2` bitset features with 1-pass learned readout (`0` recurrent passes).
- **`hybrid_sym_rec_opened` (`53,062` params, trained on `20261227/train` only):** Exact progressive $k = 1, 2, 3$ `u64` bitset bridges fed into a 3-pass recurrent `FactoredAxialFrescoBlock`.
- **`rel_prov_guarded` / `rel_prov_mask_guard` / `rel_prov_adaptive` (`53,069` frozen LBF-17 params):** Full 3-pass learned relational-provenance fresco under `fable_len_guard`, `pr348_mask_guard`, and train-calibrated 1–3 pass margin halting $(\gamma_1, \gamma_2)$.
- **Uncapped Bidirectional BFS (`bidir@4096`):** Quantifies exact convergence work and unique successor enumerations when `bidir` is freed from the $B = 128$ budget cap.

---

## 5. Complete Cost Accounting & Synthetic Successor-Cost Sweep (`0, 1, 10, 100 ms`)

Clean inference episodes run with zero profiling hooks, zero tensor-recording clones, and single-thread BLAS/OMP. Post-episode accounting records separately:
1. **Logical charged successor work** (`work` and penalized work `work + [not closed]`).
2. **Unique 1-hop successor enumerations** (`unique_succ_enumerated`: exact count of transitions across distinct words whose 1-hop neighborhood is inspected by the simulator or Pass-1 envelope).
3. **Repeated successor enumerations** (`repeated_succ_enumerated` with warm/cold cache vs uncached counterfactual `repeated_succ_if_uncached`).
4. **Rule-window substring checks** split into simulator checks (`sim_rule_window_checks`) and model/oracle checks (`model_rule_window_checks`).
5. **Clean isolated wall-clock latency (`ms`)** (cold and warm cache) and **OS kernel peak RSS (`bytes`)** (`GetProcessMemoryInfo` / `VmHWM`).
6. **Synthetic per-successor cost sweep** $c_{\text{succ}} \in \{0, 1, 10, 100\}\text{ ms}$ under both external-verifier-charged cost ($T_{\text{clean}} + c_{\text{succ}} \cdot W_{\text{pen}}$) and enumerated-successor cost ($T_{\text{clean}} + c_{\text{succ}} \cdot N_{\text{uniq}}$), pinpointing the exact economic break-even $c_{\text{succ}}^*$ where abstract propagation pays for its envelope overhead.

---

## 6. $2 \times 2$ Recurrence Factorial, Collision Thresholds, and Root-Level Statistics

- **$2 \times 2$ Bank Factorial (`shapley_sum_eq_twice_joint`, `marginal_sum_minus_joint_eq_interaction`):** Evaluates $(E, M) \in \{1, 0\}^2$ under both zeroing and spatial rolling to report interaction $I_{EM} = Y_{11} - Y_{01} - Y_{10} + Y_{00}$ before Shapley shares $(\phi_E, \phi_M)$, explicitly noting that inference lesions measure **representation reliance** on frozen weights whereas comparisons against `sym_prov_readout`, `absint_mm`, and `compute_nonrecurrent` measure **incremental value of learned recurrence**.
- **Breakdown Thresholds of $F_p = 4$:** Tested across 5 scaling tracks (`L <= 32`, `|Sigma| <= 12`, `ell_max <= 6`, `k <= 5`, adversarial `0 mod 4` collisions, and 16-step search), confirming `cex4a_fp4_spurious_fp5_exact`, `cex4b_intrafiber_step2_fork`, and `cex4c_horizon_compression_long_word`.
- **Root-Level Statistical Unit (`R` Independent Roots vs `S = 3` Seeds):** All confidence intervals and hypothesis tests aggregate over the $S = 3$ seeds (`107, 109, 113`) to the root mean $\bar{Y}_r$ first, reporting between-root variance, seed variance, $\mathrm{ICC}$, and $N_{\text{eff}}$.
