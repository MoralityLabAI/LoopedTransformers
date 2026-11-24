# LBF-18: Incremental Relational Fresco, Interaction Theory, and Provenance Collision Thresholds

Stacked on merged `main` commit `3d8243818858c19c5031c36e2bf7e328f831cf25` ([PR #343](https://github.com/MoralityLabAI/LoopedTransformers/pull/343)), freezing all 24 LBF-17 checkpoints in [`results/lbf17/`](results/lbf17/manifest.json). Standalone zero-axiom Lean 4.19.0 formalization: [`lean/LBF18IncrementalAndCollisions.lean`](lean/LBF18IncrementalAndCollisions.lean) (`12/12` theorems compiled with zero `propext`, zero `Classical.choice`, zero `Quot.sound`).

---

## 1. Exact Three-Stage Factorization of the Relational-Provenance Fresco

Let $U_t = (u_1, \dots, u_{m_0}) \in (\Sigma^L)^{m_0}$ and $V_t = (v_1, \dots, v_{m_1}) \in (\Sigma^L)^{m_1}$ be the active forward and backward rewrite frontiers at decision step $t$ for theorem case $c = (w_{\text{src}}, w_{\text{tgt}}, R)$ under checkpoint parameters $\theta$, pass count $K \in \{1, 2, 3\}$, and intervention mode $\mu$.

### 1.1 Stage A: Word-Level Multi-Pass Relational Envelope ($\mathcal{C}_{\text{word}}$)
In `RelationalProvenanceFrescoNet`, the $K$-pass word-envelope operator
$$
E_{\theta, R, \pi_{\text{env}}(\mu)}^{(K)}(w)
= \Bigl(\mathbf{1}_w,\ \mathbf{1}_{w}^{(2)},\ d_{\text{pub}}(R, w),\ \bigl\{(q_w^{(k)}, b_w^{(k)}, \mathrm{bi}_w^{(k)}, \beta_w^{(k)})\bigr\}_{k=0}^{K-1}\Bigr)
$$
depends exclusively on the dependency tuple
$$
\kappa_{\text{word}}(w) = \bigl(\mathrm{sha256}(\theta),\ \mathrm{ver}(\theta),\ \mathrm{canon}(R),\ \pi_{\text{env}}(\mu),\ K,\ w\bigr),
$$
where $\pi_{\text{env}}(\mu) = \texttt{normal}$ for $\mu \in \{\texttt{normal}, \texttt{ablate\_axial}, \texttt{mat\_zero}, \texttt{mat\_roll}\}$, $\pi_{\text{env}}(\mu) = \texttt{env\_zero}$ for $\mu \in \{\texttt{env\_zero}, \texttt{env\_mat\_zero}\}$, and $\pi_{\text{env}}(\mu) = \texttt{env\_roll}$ for $\mu \in \{\texttt{env\_roll}, \texttt{env\_mat\_roll}\}$.

- **Incremental Frontier Update Identity (`incremental_word_map_equiv`):** When search step $t$ expands node $u_{i^*} \in U_t$ into novel successors $\Delta U_t$, the updated frontier $U_{t+1} = (U_t \setminus \{u_{i^*}\}) \cup \Delta U_t$ evaluates $E$ **only** on words $w \in \Delta U_t \setminus \operatorname{dom}(\mathcal{C}_{\text{word}})$, yielding bit-identical word envelopes (up to $\le 10^{-6}$ SIMD batching associativity) to recomputing $U_{t+1} \oplus V_t$.
- **Fail-Closed Rejection (`keyed_cache_rejects_mismatch`):** Cache reuse is strictly rejected if any component of $\kappa_{\text{word}}$ changes, or when $\mu = \texttt{scramble\_window}$ (whose random coordinate permutation in LBF-17 depends on $(m_0, m_1)$).

### 1.2 Stage B: Static Pairwise Subterm Ribbon Cache ($\mathcal{C}_{\text{pair}}$)
For all modes $\mu \notin \{\texttt{scramble\_window}, \texttt{scramble\_ribbon}\}$, the $K$-pass pooled subterm ribbon vector for any word pair $(u, v) \in \Sigma^L \times \Sigma^L$,
$$
\Pi_{\theta, R, \pi_{\text{env}}(\mu)}^{(k)}(u, v)
= \operatorname{Pool}_{\text{mm},\max}\!\Bigl(\operatorname{GELU}\bigl(\operatorname{Conv1d}_\theta\bigl(W_{\text{rib}}\,\Phi^{(k)}(E(u), E(v), \mathbf{1}_{u \neq v})\bigr)\bigr)\Bigr) \in \mathbb{R}^{2 d_m},
$$
together with static pair geometry $(d_H(u, v),\ \lceil d_H(u, v)/3 \rceil,\ \mathrm{span}(u, v))$, is invariant to frontier sizes $(m_0, m_1)$, path distances $(g_0(u), g_1(v))$, remaining budget $b_{\text{rem}}$, depth cap $D$, persistence `prev_pot`, and matrix-bank state $B_m^{(k)}$. Caching $\Pi^{(0..K-1)}(u, v)$ (`3 × 76` floats = `912 B` per pair) under
$$
\kappa_{\text{pair}}(u, v) = \bigl(\mathrm{sha256}(\theta),\ \mathrm{ver}(\theta),\ \mathrm{canon}(R),\ \pi_{\text{env}}(\mu),\ K,\ u,\ v\bigr)
$$
reduces step-$t$ $O(m_0 m_1 L)$ ribbon convolutions to only newly formed pairs $|\Delta U_t| \cdot m_1$ (or $m_0 \cdot |\Delta V_t|$) while rejecting cross-size reuse under $\mu \in \{\texttt{scramble\_window}, \texttt{scramble\_ribbon}\}$.

### 1.3 Stage C: Dynamic Cross-Frontier Axial Competition
At every decision step $t$, Stage C assembles cached $\Pi^{(k)}(u_i, v_j)$ with live dynamic public features
$$
\mathrm{pub}_t(u_i, v_j) = \bigl(g_0/D,\ g_1/D,\ (g_0+g_1)/D,\ d_H/L,\ \mathrm{lb}/D,\ (D - \mathrm{tot\_lb})_+/D,\ \mathbf{1}_{\mathrm{feas}},\ \mathrm{span}/L,\ b_{\text{rem}}/B,\ \Delta\log m,\ \mathrm{pers}_t,\ \mathrm{prev\_v}_t\bigr)
$$
and executes `pair_in -> pair_prelude -> fresco_gate -> FactoredAxialFrescoBlock(infeas_t)` across passes $k = 0 \dots K-1$, preserving exact feasibility masking, frontier competition, budget conditioning, and persistence semantics.

---

## 2. Bitset-Continuous Sparse Trellis Isomorphism

In `FastProvenanceBigramCell`, every rule MLP ends in a strictly positive `Softplus` gate multiplied by the structural match product $\pi_{r,p} \ge 0$:
$$
\mathrm{act}_{r,p} = \operatorname{Softplus}(\mathrm{MLP}(\dots)) \odot \pi_{r,p} > 0 \iff \pi_{r,p} > 0 \quad (\texttt{zero\_preserving\_gate\_support}).
$$
Combined with rational saturation $\rho_\varepsilon(x) = x / (x + 10^{-4})$, the Boolean support at each provenance fiber $f \in \{0, \dots, F_p - 1\}$ and edge $p \in \{0, \dots, L-2\}$ for any alphabet $|\Sigma| \le 8$ (`|\Sigma|^2 <= 64`) fits inside a single hardware **`u64` register**:
- Bit $(a \cdot |\Sigma| + b)$ is `1` iff bigram $(a, b)$ has nonzero support.
- Forward-backward Viterbi trellis filtering reduces to bitwise shifts and masks over $L - 1$ `u64` words before continuous channel arithmetic is touched, skipping the `83%–93%` inactive `(f, p, a, b)` cells and unactivated rewrite rules with zero approximation error.

---

## 3. Hybrid Symbolic-Envelope / Recurrent-Readout & Adaptive Pass Allocation

1. **Hybrid Symbolic-Envelope / Recurrent-Readout (`hybrid_sym_rec_opened`, `53,074` params):**
   Replaces learned multi-pass neural word convolutions with exact `u64`-bitset multi-horizon (`k = 1, 2, 3`) `prov4_adj2` + `pos1` envelopes (zero neural FLOPs in word propagation, `100%` recall, `99.05%` precision), projected equivariantly per pass $k \in \{1, 2, 3\}$ into a **3-pass recurrent `FactoredAxialFrescoBlock`** (`fresco_gate` bias `-1.5`). Unlike `sym_prov_readout` ($K=1$ static readout), pass $k$ progressively feeds horizon-$k$ symbolic bridges into the recurrent cross-frontier bank $B_m^{(k)}$.
2. **Adaptive 1–3 Pass Allocation (`rel_prov_adaptive`, `0` extra parameters on frozen `rel_prov_opened`):**
   Let $s_{(1)}^{(k)} \ge s_{(2)}^{(k)}$ be the top two feasible candidate action logits after pass $k \in \{1, 2, 3\}$, with margin $\Delta s^{(k)} = s_{(1)}^{(k)} - s_{(2)}^{(k)}$. The controller halts at pass $k=1$ if total feasible actions $\le 1$ or $\Delta s^{(1)} \ge \gamma_1$, halts at pass $k=2$ if $\Delta s^{(2)} \ge \gamma_2$, and otherwise executes pass $k=3$. Thresholds $(\gamma_1, \gamma_2)$ are calibrated exclusively on `20261227/train` frames prior to confirmation registration.

---

## 4. Exact $2 \times 2$ Recurrence Factorial & Shapley Attribution

For any metric $Y(E, M)$ across envelope-bank state $E \in \{1, 0\}$ and matrix-bank state $M \in \{1, 0\}$:
- **Joint recurrence effect:** $\Delta_{\text{joint}} = Y_{11} - Y_{00}$.
- **Interaction term (`marginal_sum_minus_joint_eq_interaction`):**
  $$I_{EM} = Y_{11} - Y_{01} - Y_{10} + Y_{00} = (Y_{11} - Y_{01}) + (Y_{11} - Y_{10}) - \Delta_{\text{joint}}.$$
- **Exact Shapley values (`shapley_sum_eq_twice_joint`):**
  $$\phi_E = \tfrac{1}{2}\bigl((Y_{11} - Y_{01}) + (Y_{10} - Y_{00})\bigr), \qquad \phi_M = \tfrac{1}{2}\bigl((Y_{11} - Y_{10}) + (Y_{01} - Y_{00})\bigr), \qquad \phi_E + \phi_M = \Delta_{\text{joint}}.$$
Additive credit shares $\phi_E / \Delta_{\text{joint}}$ and $\phi_M / \Delta_{\text{joint}}$ are reported only alongside $I_{EM}$ and its root-level `95%` bootstrap CI.

---

## 5. Where $F_p = 4$ Provenance Fibers Succeed and Where They Break

### 5.1 Disjoint-Collision Commutation Theorem (`mod_collision_disjoint_intervals`)
If all rules satisfy width $\ell \le F_p$, any two distinct 1-step redexes at $p_1 < p_2$ colliding in the same congruence fiber ($p_1 = F_p q_1 + r,\ p_2 = F_p q_2 + r$) satisfy $p_1 + \ell \le p_1 + F_p \le p_2$. Their modified intervals $[p_1, p_1 + \ell_1)$ and $[p_2, p_2 + \ell_2)$ are strictly disjoint, so their joint rewrite commutes and is **concretely reachable in 2 steps** (`disjoint_mod4_rewrites_commute_in_2_steps`).

### 5.2 Three Constructive Breakdown Regimes of $F_p = 4$ (Kernel-Verified in Lean 4.19.0)
1. **Rule Width Exceeding Fiber Count ($\ell > F_p$, `cex4a_fp4_spurious_fp5_exact`):** For $w_0 = \texttt{abbbebbbc}$ ($L=9$) and width-5 rules $\texttt{abbbe} \leftrightarrow \texttt{dbbbf}$ ($p_1 = 0$) and $\texttt{ebbbc} \leftrightarrow \texttt{fbbbd}$ ($p_2 = 4$), both redexes contend for the unique `e` at index 4 yet collide in Fiber $0 \equiv 4 \pmod 4$, hallucinating the permanent chimera $\texttt{dbbbfbbbd}$ (`|C| = 3`); $F_p = 5$ restores exactness.
2. **Secondary Intra-Fiber Forking at Depth $k \ge 3$ (`cex4b_intrafiber_step2_fork`):** Even when $\ell \le 3$, $w_0 = \texttt{fabbc}$ takes a single step-1 transition $\texttt{fa} \to \texttt{aa}$ at $p=0$ into Fiber 0 (`aabbc`), then forks incompatibly at step 2 inside Fiber 0 ($\texttt{ab} \to \texttt{de}$ vs $\texttt{bbc} \to \texttt{efa}$), enabling $\texttt{def} \to \texttt{ccc}$ at step 3 regardless of step-1 $F_p$.
3. **Multi-Redex Horizon Compression on Longer Words (`cex4c_horizon_compression_long_word`):** On $w_0 = \texttt{abefabefab}$ ($L=10$) with $\texttt{ab} \leftrightarrow \texttt{cd}$ at $p \in \{0, 4, 8\}$ (all $\equiv 0 \pmod 4$), $F_p = 4$ merges all $m = 3$ commuting rewrites at Pass 1, admitting 3-step word $\texttt{cdefcdefcd}$ within bounded horizon $k = 2 < m$.

---

## 6. Root-Level Statistical Unit Separation (`R` Independent Roots vs `S = 3` Seeds)

For $R$ independent theorem roots ($R = 48$ calibration; $R = 16$ per sealed cohort) evaluated across $S = 3$ training seeds (`107, 109, 113`), the independent observational unit is strictly the root mean $\bar{Y}_r = \frac{1}{S}\sum_{s=1}^S Y_{r,s}$. Between-root variance $\hat{\sigma}_{\text{root}}^2$, within-root seed variance $\hat{\sigma}_{\text{seed|root}}^2$, intraclass correlation $\mathrm{ICC}$, effective sample size $N_{\text{eff}} = \frac{RS}{1 + (S-1)\mathrm{ICC}}$, all-3-seed consensus closure $\frac{1}{R}\sum_{r=1}^R \prod_{s=1}^S C_{r,s}$, and root-paired bootstrap / sign-flip permutation intervals are reported separately from raw $R \times S$ episode totals.
