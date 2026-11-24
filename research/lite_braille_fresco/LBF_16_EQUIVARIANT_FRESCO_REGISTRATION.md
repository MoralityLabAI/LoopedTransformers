# LBF-16 Mathematical Specification and Registration: $\mathfrak{S}_6$-Equivariant Horizon-Separated Meeting Fresco

Registered 2026-11-24 **before** generating the LBF-16 suite (`20261221`), fitting any LBF-16 checkpoint, or scoring any LBF-16 root. Stacked on archived LBF-15 results (`03c2d6a8`). Pinned rewrite simulator `cd9ac784c5804ec2fe595bb6a01d2697fc016610` / blob `204a1d012a4ebe8e80d8bd722590742b7e7f02ba`.

---

## 1. Architectural Revision Motivated by LBF-15's Falsifying Controls

In [LBF-15](LBF_15_MEETING_POTENTIAL_RESULTS.md), `opened_normal` beat strong symbolic `bidir` by `+11.11 pp` (`88.19%` vs `77.08%`, `-31.6%` penalized work), but causal shams falsified two representational choices:
1. `opened_scramble` (`93.06%`) outperformed `opened_normal` (`88.19%`) because $3 \times 3$ `Conv2d` across planar fringe rows blurred true bridge children with competing dead-end siblings (`Moran's I = -0.025`).
2. `opened_zero` (`90.97%`) and `recurrent_standard` (`91.67%`) outperformed `opened_normal` (`88.19%`) because fixed character lookup embeddings `nn.Embedding(7, 40)` violated $\mathfrak{S}_6$ alphabet-permutation symmetry across independent rule-system lineages, causing opened recurrent banks to overfit training-set symbol identities, while per-cell `LayerNorm` erased rewrite support sparsity and overwrote Pass-1 (`1`-hop) anchors at Pass $K$.

---

## 2. Mathematical Specification of LBF-16

### 2.1 Exact $\mathfrak{S}_6$ Alphabet-Permutation Equivariance and Support-Preserving Soft-Rewrite Envelope
Let $\mathbf{1}_w \in \{0, 1\}^{L \times 6}$ denote the one-hot encoding of discovered word $w \in \Sigma^L$, and let $(X_e, Y_e) \in \{0, 1\}^{\ell_e \times 6} \times \{0, 1\}^{\ell_e \times 6}$ ($\ell_e \in \{2, 3\}$) denote the oriented rewrite rules $e \in \{1, \dots, 2|R|\}$. Every channel operation acts identically across the alphabet axis $\sigma \in \{0,\dots,5\}$:
\[
H_w^{(0)}[p, \sigma, :] = \tanh\!\bigl(W_{\text{init}}[\mathbf{1}_w[p,\sigma],\; \mathbf{1}_{p=0}\mathbf{1}_w[p,\sigma],\; \mathbf{1}_{p=L-1}\mathbf{1}_w[p,\sigma]]\bigr) \in \mathbb{R}^{C_e}, \qquad B_w^{(0)} = 0.
\]
At pass $k \in \{1, \dots, K\}$ ($K=3$), define the **zero-preserving nonnegative envelope gate** $G_{\text{env}}$ (`bias = -6.0` for `standard`, `-1.5` for `opened`):
\[
G_{\text{env}}(H, B) = \sigma(W_g[H; B] + b_g) \odot \Bigl(\operatorname{ReLU}(B) \frac{\operatorname{softplus}(W_p^\top)}{\sqrt{C_e}}\Bigr), \qquad
Q_w^{(k)} = \operatorname{clamp}_{[0, 1.5]}\!\bigl(\mathbf{1}_w \otimes \mathbf{1}_{C_e} + G_{\text{env}}(H_w^{(k-1)}, \mathcal{I}_{\text{mode}}(B_w^{(k-1)}))\bigr).
\]
For each rule $e$ of length $\ell_e \in \{2, 3\}$ and window $p \in \{0, \dots, L - \ell_e\}$, compute exact $\mathfrak{S}_6$-invariant per-slot matches and their **multiplicative conjunction**:
\[
m_{w, e, p, q}^{(k)} = \sum_{\sigma=1}^{6} Q_w^{(k)}[p+q, \sigma, :] \, X_e[q, \sigma], \qquad
\pi_{w, e, p}^{(k)} = \prod_{q=0}^{\ell_e - 1} m_{w, e, p, q}^{(k)} \in \mathbb{R}_{\ge 0}^{C_e}.
\]
Rule firing writes strictly onto matched RHS symbols via nonnegative $\operatorname{softplus}$ modulation:
\[
R_w^{(k)}[p+q, \sigma, :] = \sum_{e, p} \Bigl(\psi_{\ell_e}\bigl([\pi_{w,e,p}^{(k)};\, m_{w,e,p,0:\ell_e}^{(k)}]\bigr) \odot \pi_{w, e, p}^{(k)}\Bigr) Y_e[q, \sigma], \qquad
B_w^{(k)} = \tanh\!\bigl(R_w^{(k)} + 0.65\,\operatorname{ReLU}(\mathcal{I}_{\text{mode}}(B_w^{(k-1)}))\bigr).
\]

**Support-Sparsity & Equivariance Properties:**
1. **$\mathfrak{S}_6$ Equivariance:** For every permutation $\tau \in \mathfrak{S}_6$, $B_{\tau(w)}^{(k)}[p, \tau(\sigma), :] = B_w^{(k)}[p, \sigma, :]$ identically, so inner products over $\sigma$ are strictly $\mathfrak{S}_6$-invariant on unseen rule-system lineages.
2. **Exact Horizon Support:** Because $\tanh(0) = 0$ and $\pi_{w,e,p}^{(k)} = 0$ unless every window slot has positive support in $Q_w^{(k)}$, $B_w^{(k)}[p, \sigma, :] > 0$ **only if** symbol $\sigma$ is reachable at index $p$ in $\le k$ overlapping rewrites from $w$. Under `mode="zero"`, $\mathcal{I}_{\text{zero}}(B_w^{(k-1)}) = 0 \implies \operatorname{supp}(B_w^{(k)}) = \operatorname{supp}(B_w^{(1)})$ (frozen at $1$-hop reachability).

### 2.2 Horizon-Separated Subterm Ribbon and Feasibility-Masked Set-Equivariant Axial Fresco
To prevent multi-hop reachability ($k \ge 2$) from obscuring immediate $1$-hop closures, anchor $A_w := B_w^{(1)}$ at Pass $1$ and define multi-hop novelty $N_w^{(k)} := \operatorname{ReLU}(B_w^{(k)} - A_w)$ ($N_w^{(1)} \equiv 0$). Form the $\mathfrak{S}_6$-invariant **1D subterm spatial ribbon** $Z_{ij, p}^{(k)} \in \mathbb{R}^{d_m}$ along word coordinates $p = 0, \dots, L-1$:
\[
O_{ij, p}^{(k)} = \Bigl[\langle Q_{u_i,p}^{(k)}, Q_{v_j,p}^{(k)}\rangle_\Sigma;\; \langle A_{u_i,p}, A_{v_j,p}\rangle_\Sigma;\; \langle N_{u_i,p}^{(k)}, N_{v_j,p}^{(k)} + \tfrac{1}{2}A_{v_j,p}\rangle_\Sigma + \tfrac{1}{2}\langle A_{u_i,p}, N_{v_j,p}^{(k)}\rangle_\Sigma;\; \langle B_{u_i,p}^{(k)}, \mathbf{1}_{v_j,p}\rangle_\Sigma;\; \langle \mathbf{1}_{u_i,p}, B_{v_j,p}^{(k)}\rangle_\Sigma;\; \mathbf{1}_{u_i[p]\ne v_j[p]}\Bigr],
\]
followed by spatial `Conv1d(kernel_size=3, padding=1)` along $p \in \{0, \dots, L-1\}$, pooled over mismatched indices and global maxima into pair representation $P_{ij}^{(k)} \in \mathbb{R}^{d_m}$.

Across frontier sets $\mathcal{U}^{(t)} \times \mathcal{V}^{(t)}$, let $F_{ij} = \mathbb{I}[g_0(u_i) + g_1(v_j) + \max(1, \lceil d_H(u_i, v_j)/3 \rceil) \le D]$ be the public feasibility mask. `AxialSetFrescoBlock` applies **$\mathfrak{S}_{m_0} \times \mathfrak{S}_{m_1}$ permutation-equivariant** feasibility-masked row/column attention and feasible-partner mean centering, updating potential monotonically around Pass 1:
\[
M_{ij}^{(t, 1)} = w_{\text{pot}}^\top H_{ij}^{(1)} + b_{\text{pot}}, \qquad
M_{ij}^{(t, k)} = M_{ij}^{(t, k-1)} + \delta \cdot w_{\text{pot}}^\top (H_{ij}^{(k)} - P_{ij}^{(k)}) \quad (k \ge 2).
\]
Thus `scramble` permutes the **1D subterm coordinate lattice $p = 0, \dots, L-1$** (testing contiguous subword spatial structure), while frontier rows/columns are strictly set-equivariant.

---

## 3. Pre-Registered Suite, Matched Budgets, and Decision Screens

- **Fresh untouched generator seed:** `20261221` (`64` train, `48` calibration, `16` sealed ordinary, `16` sealed OOD = `144` unique canonical rule-system lineages, verified disjoint from all `576` lineages of seeds `20261204`, `20261205`, `20261207`, `20261209`, and `20261215`).
- **Matched parameter budget (`<= 57,475`):**
  - `recurrent_opened` ($K=3$, `ce=21, dm=38`, `gate_bias=-1.5`): `53,047` params.
  - `recurrent_standard` ($K=3$, `ce=21, dm=38`, `gate_bias=-6.0`): `53,047` params (identical non-gate initialization and draw schedule).
  - `unstructured_recurrent` ($K=3$, pointwise hidden `226`, `gate_bias=-1.5`): `53,007` params (`-0.075%`).
  - `nonrecurrent` ($K=1$, untied `nr_extra` hidden `73`): `53,067` params (`+0.038%`).
- **Training protocol:** Three seeds `107, 109, 113`; `240` mini-batch AdamW updates (`batch_size = 6` frames/step, cosine LR `3e-3 -> 2e-4`, pass weights $w_k = k^2$) on train-only frames from `geom_mm` and `graph_bridge` rollouts with relative geodesic bridge targets and counterfactual continuation advantages. All `64` training references and `24` counterfactual continuations certified axiom-free in Lean 4.19.0.
- **Evaluated conditions (`48` calibration roots):**
  - Controls: `astar`, `bidir`, `geom_mm`, `graph_bridge`, `uniform` (5 streams).
  - Learned controllers & causal interventions: `nonrecurrent`, `unstructured_recurrent`, `recurrent_standard`, `opened_normal`, `opened_zero`, `opened_roll`, `opened_scramble`, `opened_no_persist` at primary cap `WORK = 128`, plus full matched-budget curves across `B ∈ {32, 64, 96, 128}` (including all shams across budgets to measure tight-budget multi-hop causal sensitivity). Every successful witness at `B = 128` plus all `48` calibration references compiled standalone and axiom-free in Lean 4.19.0.
- **Pre-declared screens:**
  1. **Primary Screen 1A (vs Symbolic `bidir` at `B = 128`):** `opened_normal` exceeds `bidir` by $\ge +10.0\text{ pp}$ closure AND reduces mean penalized work by $\ge 25\%$, with all 3 seeds strictly above `bidir`.
  2. **Primary Screen 1B (vs Strongest Explicit Heuristic $\max(\texttt{geom\_mm}, \texttt{graph\_bridge})$ at `B = 128`):** `opened_normal` exceeds the strongest explicit heuristic by $\ge +5.0\text{ pp}$ closure AND reduces mean penalized work by $\ge 20\%$, with all 3 seeds strictly above the strongest explicit heuristic.
  3. **Secondary Screen 2A (Recurrence & Cross-Frontier Structure):** Mean budget-curve closure AUC (`B ∈ {32, 64, 96, 128}`) and/or `B = 128` closure of `opened_normal` exceeds parameter-matched `nonrecurrent`, `unstructured_recurrent`, and `recurrent_standard` (with lower or equal penalized work at `B = 128`).
  4. **Secondary Screen 2B (Causal Memory & Subterm Spatial Coherence):** `opened_normal` outperforms `opened_zero`, `opened_roll`, and `opened_scramble` on budget-curve closure AUC (`B ∈ {32, 64, 96, 128}`) and achieves non-negative contrast at `B = 128` without the sham inversions seen in LBF-15.
