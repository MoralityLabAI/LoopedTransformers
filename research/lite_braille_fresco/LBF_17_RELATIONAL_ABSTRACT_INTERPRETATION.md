# LBF-17 Mathematical Specification: Relational Abstract Interpretation and Provenance-Preserving Equivariant Frescoes

Companion formalization: [`lean/LBF17AbstractInterp.lean`](lean/LBF17AbstractInterp.lean) (`15/15` constructive theorems compiled standalone with zero axioms in Lean 4.19.0, commit `6caaee842e94`).

---

## 1. Repairing the LBF-16 Horizon-Support Theorem via Galois Connections

### 1.1 Concrete Rewrite Domain and Positionwise Cartesian Abstraction
Fix finite alphabet $\Sigma$ ($|\Sigma| = 6$) and word length $L \ge 2$. Let $\mathcal{C} = (\mathcal{P}(\Sigma^L), \subseteq)$ be the concrete powerset lattice of word sets, and for a symmetric rewrite system $R$ with oriented rules $E_R = \{(x_e \to y_e, \ell_e)\}_{e=1}^{2|R|}$ ($\ell_e \in \{2, 3\}$), define the monotone 1-step reflexive-rewrite operator
\[
F_R(S) = S \cup \bigl\{ w' \in \Sigma^L : \exists w \in S,\; \exists (x_e \to y_e) \in E_R,\; \exists p \in \{0,\dots,L-\ell_e\},\; w_{p:p+\ell_e} = x_e \land w' = w_{0:p}\,y_e\,w_{p+\ell_e:L} \bigr\}.
\]
Then $F_R^k(\{w_0\})$ is the exact concrete $\le k$-step reachable word set from $w_0$.

Define the **positionwise ($1$-gram) support lattice** $\mathcal{A}_1 = (\{0, 1\}^{L \times \Sigma}, \le)$ with abstraction $\alpha_1 : \mathcal{P}(\Sigma^L) \to \mathcal{A}_1$ and concretization $\gamma_1 : \mathcal{A}_1 \to \mathcal{P}(\Sigma^L)$:
\[
\alpha_1(S)[p, \sigma] = \mathbb{I}[\exists w \in S : w_p = \sigma], \qquad
\gamma_1(A_1) = \bigl\{ w \in \Sigma^L : \forall p \in \{0,\dots,L-1\},\; A_1[p, w_p] = 1 \bigr\}.
\]

**Theorem 1.1 (Positionwise Galois Connection; Lean: `galois_pos1`).**
$(\alpha_1, \gamma_1)$ forms a monotone Galois connection $\mathcal{P}(\Sigma^L) \underset{\alpha_1}{\overset{\gamma_1}{\leftrightarrows}} \mathcal{A}_1$: for all $S \subseteq \Sigma^L$ and $A_1 \in \mathcal{A}_1$, $\alpha_1(S) \le A_1 \iff S \subseteq \gamma_1(A_1)$. Its closure operator $\gamma_1(\alpha_1(S)) = \prod_{p=0}^{L-1} \{w_p : w \in S\}$ is the **independent-attribute Cartesian product** of positionwise symbol marginals.

### 1.2 Revised Horizon-Support Soundness Theorem (Replacing LBF-16 §2.1 Property 2)
On any abstract state $A_1 \in \mathcal{A}_1$ with $\gamma_1(A_1) \ne \emptyset$, the best abstract transformer $F_R^{\sharp, 1} := \alpha_1 \circ F_R \circ \gamma_1$ coincides with multiplicative window matching:
\[
F_R^{\sharp, 1}(A_1)[r, \tau] = A_1[r, \tau] \;\lor\; \bigvee_{(x_e \to y_e) \in E_R}\; \bigvee_{p : r \in [p, p+\ell_e)} \Bigl(\mathbb{I}[y_e[r-p] = \tau] \land \bigwedge_{q=0}^{\ell_e - 1} A_1[p+q, x_e[q]]\Bigr).
\]
In LBF-16's `SparseMultiplicativeRewriteCell`, because $\operatorname{softplus}(z) > 0$, $\sigma(z) > 0$, and $\tanh(z) > 0 \iff z > 0$ on $\mathbb{R}_{\ge 0}$, the strict channelwise positivity indicator $\operatorname{supp}(\mathbf{1}_w \vee B_w^{(k)})[p, \sigma] := \mathbb{I}[\mathbf{1}_w[p,\sigma]=1 \lor \exists c,\, B_w^{(k)}[p,\sigma,c] > 0]$ equals $(F_R^{\sharp, 1})^k(\alpha_1(\{w\}))$ for all $k \ge 1$.

**Theorem 1.2 (Repaired Horizon-Support Overapproximation Theorem; Lean: `repaired_lbf16_horizon_support_soundness`, `cex1_pos1_pass1_exact_pass2_spurious`).**
For every rewrite system $R$, initial word $w_0 \in \Sigma^L$, and pass horizon $k \ge 0$:
1. **Sound Overapproximation ($100\%$ Recall):** $F_R^k(\{w_0\}) \subseteq \gamma_1\bigl(\operatorname{supp}(\mathbf{1}_{w_0} \vee B_{w_0}^{(k)})\bigr)$, equivalently $\alpha_1(F_R^k(\{w_0\})) \le \operatorname{supp}(\mathbf{1}_{w_0} \vee B_{w_0}^{(k)})$. Thus $B_{w_0}^{(k)}[p, \sigma, :] = \mathbf{0}$ and $\mathbf{1}_{w_0}[p, \sigma] = 0$ jointly certify that $\sigma$ is **unreachable** at position $p$ in $\le k$ steps.
2. **Pass-1 Exactness ($k = 1$):** Because $\gamma_1(\alpha_1(\{w_0\})) = \{w_0\}$ is a singleton, $\operatorname{supp}(\mathbf{1}_{w_0} \vee B_{w_0}^{(1)}) = \alpha_1(F_R(\{w_0\}))$ is exact.
3. **Strict Overapproximation at $k \ge 2$ (Counterexample 1):** Let $L = 3$, $w_0 = \texttt{abc}$, and $R_1 = \{\texttt{ab} \leftrightarrow \texttt{de},\; \texttt{bc} \leftrightarrow \texttt{fa},\; \texttt{df} \leftrightarrow \texttt{cc}\}$. The entire concrete reachable component is $F_{R_1}^*(\{\texttt{abc}\}) = \{\texttt{abc}, \texttt{dec}, \texttt{afa}\}$, in which $\texttt{c}$ never appears at positions $0$ or $1$. At $k = 1$, $\alpha_1(\{\texttt{abc}, \texttt{dec}, \texttt{afa}\})$ supports $\{\texttt{a},\texttt{d}\}$ at $p=0$ (via $\texttt{dec}$) and $\{\texttt{b},\texttt{e},\texttt{f}\}$ at $p=1$ (via $\texttt{afa}$), whose Cartesian concretization $\gamma_1$ contains the **chimera** $\texttt{dfc} \notin F_{R_1}^*(\{\texttt{abc}\})$. At $k = 2$, multiplicative matching fires $\texttt{df} \to \texttt{cc}$, producing spurious support $B_{\texttt{abc}}^{(2)}[0, \texttt{c}, :] > 0$ and $B_{\texttt{abc}}^{(2)}[1, \texttt{c}, :] > 0$.

---

## 2. Hierarchy of Relational and Provenance-Preserving Frescoes

### 2.1 Adjacent $2$-Gram ($\mathcal{A}_2$) and Length-3 Subterm ($\mathcal{A}_3$) Abstractions
For $n \in \{2, 3\}$ and $L \ge n$, define $\mathcal{A}_n = (\{0, 1\}^{(L - n + 1) \times \Sigma^n}, \le)$ with
\[
\alpha_n(S)[p, u] = \mathbb{I}[\exists w \in S : w_{p:p+n} = u], \qquad
\gamma_n(A_n) = \bigl\{ w \in \Sigma^L : \forall p \in \{0,\dots,L-n\},\; A_n[p, w_{p:p+n}] = 1 \bigr\}.
\]

**Theorem 2.1 (Relational Galois Hierarchy & Trellis Exactness; Lean: `galois_adj2`, `galois_sub3`, `adj2_refines_pos1`, `sub3_refines_adj2`).**
1. $(\alpha_2, \gamma_2)$ and $(\alpha_3, \gamma_3)$ are monotone Galois connections satisfying the universal precision refinement
   \[
   S \;\subseteq\; \gamma_3(\alpha_3(S)) \;\subseteq\; \gamma_2(\alpha_2(S)) \;\subseteq\; \gamma_1(\alpha_1(S)) \qquad (\forall S \subseteq \Sigma^L,\; L \ge 3).
   \]
2. Because the overlap graph of consecutive $(n-1)$-grams along $p = 0, \dots, L-n$ is a 1D chain (treewidth $1$ over state space $\Sigma^{n-1}$), forward-backward trellis dynamic programming computes the exact reduced state $\tilde{A}_n = \alpha_n(\gamma_n(A_n))$ in $O(L|\Sigma|^n)$ operations ($36(L-1)$ entries for $n = 2$; $216(L-2)$ entries for $n = 3$).
3. On $\tilde{A}_2 = \alpha_2(\gamma_2(A_2)) \ne \mathbf{0}$, the **exact best transformer** $F_R^{\sharp, 2} = \alpha_2 \circ F_R \circ \gamma_2$ fires $(x_0 x_1 \to y_0 y_1)$ at $p$ iff $\tilde{A}_2[p, x_0, x_1] = 1$ (and $(x_0 x_1 x_2 \to y_0 y_1 y_2)$ at $p$ iff $\tilde{A}_2[p, x_0, x_1] \land \tilde{A}_2[p+1, x_1, x_2]$), writing interior RHS bigrams, left-boundary bigrams $(a, y_0)$ at $p-1$ gated by $\tilde{A}_2[p-1, a, x_0]$, right-boundary bigrams $(y_{\ell-1}, b)$ at $p+\ell-1$ gated by $\tilde{A}_2[p+\ell-1, x_{\ell-1}, b]$, and unmodified context edges outside $[p-1, p+\ell)$, followed by $\alpha_2 \circ \gamma_2$ trellis filtering.

**Differentiable Zero-Preserving Trellis Filter (Avoiding Depth Underflow):**
Naive serial multiplication $\tilde{e}_p = e_p \cdot \tanh(\max e_{p-1})$ attenuates positive activations as $O(c^{2L})$ along length-$L$ words. Using the smooth rational saturation $\rho_\varepsilon(x) = \frac{x}{x + \varepsilon}$ ($\varepsilon = 10^{-4}$, $x \ge 0$), we obtain $\rho_\varepsilon(0) = 0$ identically while $\rho_\varepsilon(x) \approx 1$ on $x \gg \varepsilon$:
\[
\vec{r}_0[\sigma_0] = \max_{\sigma_1} \rho_\varepsilon(E_0[\sigma_0, \sigma_1]), \quad
\vec{r}_{p+1}[\sigma_1] = \max_{\sigma_0}\bigl(\vec{r}_p[\sigma_0]\,\rho_\varepsilon(E_p[\sigma_0, \sigma_1])\bigr), \quad
\mathcal{T}_\varepsilon(E)_p[\sigma_0, \sigma_1] = E_p[\sigma_0, \sigma_1]\,\vec{r}_p[\sigma_0]\,\overleftarrow{r}_{p+1}[\sigma_1].
\]
$\mathcal{T}_\varepsilon$ is strictly $\mathfrak{S}_6$-equivariant, smooth on $\mathbb{R}_{\ge 0}$, and preserves the exact Boolean trellis support $\operatorname{supp}(\mathcal{T}_\varepsilon(E)) = \alpha_2(\gamma_2(\operatorname{supp}(E)))$ without exponential attenuation.

### 2.2 When Local Joint Support Loses Global Consistency, and Provenance Fiber Separation
Local $n$-gram abstractions eliminate chimeras whose incompatible redexes fail to agree on an $(n-1)$-gram overlap, **but lose global consistency whenever two mutually exclusive redex branches happen to share an $(n-1)$-gram bridge**:

1. **Counterexample 2 ($\mathcal{A}_2$ fails at $k=2$, $\mathcal{A}_3$ and provenance stay exact; Lean: `cex2_adj2_spurious_sub3_and_prov_exact`):**
   Let $w_0 = \texttt{abbc}$ ($L=4$) and $R_2 = \{\texttt{ab} \leftrightarrow \texttt{de},\; \texttt{bbc} \leftrightarrow \texttt{efa},\; \texttt{def} \leftrightarrow \texttt{ccc}\}$. The two redexes at $w_0$ overlap at $p=1$, giving concrete component $F_{R_2}^*(\{\texttt{abbc}\}) = \{\texttt{abbc}, \texttt{debc}, \texttt{aefa}\}$. At $k=1$, $\texttt{debc}$ contributes $(0:\texttt{de})$ and $\texttt{aefa}$ contributes $(1:\texttt{ef})$ and $(2:\texttt{fa})$, which glue across shared middle symbol $w_1 = \texttt{e}$ into $\texttt{defa} \in \gamma_2(A_2^{(1)})$, firing $\texttt{def} \to \texttt{ccc}$ at $k=2$. Length-3 support $\mathcal{A}_3$ keeps $A_3^{(1)}[0] = \{\texttt{abb}, \texttt{deb}, \texttt{aef}\}$ and excludes $\texttt{def}$.

2. **Counterexample 3 ($\mathcal{A}_3$ loses global consistency, branch provenance stays exact; Lean: `cex3_sub3_loses_global_consistency_prov_exact`):**
   Let $w_0 = \texttt{aebcd}$ ($L=5$) and $R_3 = \{\texttt{aeb} \leftrightarrow \texttt{def},\; \texttt{bcd} \leftrightarrow \texttt{ffa},\; \texttt{ff} \leftrightarrow \texttt{cc},\; \texttt{dec} \leftrightarrow \texttt{bbb}\}$. The two $w_0$-redexes compete for $w_0[2] = \texttt{b}$, giving exact component $F_{R_3}^*(\{\texttt{aebcd}\}) = \{\texttt{aebcd}, \texttt{defcd}, \texttt{aeffa}, \texttt{aecca}\}$ (size $4$; $\texttt{bbb}$ never occurs). Because $w_0[1] = \texttt{e}$, branch 1 ($\texttt{defcd}$) produces $(0:\texttt{def})$ while branch 2 ($\texttt{aeffa}$) produces $(1:\texttt{eff})$ and $(2:\texttt{ffa})$, sharing the **2-symbol overlap $\texttt{ef}$** at $1..2$! Hence $\texttt{deffa} \in \gamma_3(A_3^{(1)})$, stepping via $\texttt{ff} \to \texttt{cc}$ to $\texttt{decca}$ at $k=2$ and firing $\texttt{dec} \to \texttt{bbb}$ at $k=3$.

**Theorem 2.2 (Non-Backtracking Branch-Provenance Decomposition & Fiber Refinement; Lean: `provenance_fibers_refine_merged`, `nonbacktracking_provenance_soundness`, `cex1_relational_and_provenance_exact`).**
1. For any Galois connection $(\alpha, \gamma)$ and indexed family $\{S_f\}_{f \in \mathcal{F}}$, $\bigcup_{f \in \mathcal{F}} \gamma(\alpha(S_f)) \subseteq \gamma(\alpha(\bigcup_{f \in \mathcal{F}} S_f))$.
2. Every concrete walk of length $\le k$ from $w_0$ to $z$ either has $z = w_0$ (identity fiber) or decomposes via its last exit from $w_0$ into $w_0 \to w_1$ ($w_1 \ne w_0$) followed by a walk of length $\le k - 1$ inside $\{w \ne w_0\}$ that **never revisits $w_0$**.
3. **Reversible Backtracking Subtlety:** In reversible rewriting, naive 1-step branch fibers without root-backtracking suppression remain exact on `abc` through Pass $3$, but re-mix at Pass $4$ via $w_1 \to w_0 \to w_2 \to \text{chimera}$. Conversely, suppressing Pass-2 rewrites at fiber $f$'s origin window whose RHS equals $w_0$'s slice is **100% sound for all word lengths $L$** (since all positions outside the Pass-1 redex equal $w_0$, so restoring the redex slice yields $w_0$ identically, already in the identity fiber) while keeping fibers disjoint on Counterexamples 1, 2, and 3 simultaneously.

### 2.3 Exact $\mathfrak{S}_6$-Equivariant Spatial-Congruence Provenance Routing ($\mathcal{A}_{\text{prov4}\otimes\text{adj2}}$)
Routing Pass-1 redexes by rule-list index would depend on rule ordering; routing by symbol identity would break $\mathfrak{S}_6$ alphabet symmetry. Instead, routing Pass-1 redexes at window $p$ into **spatial-congruence fiber $f = p \bmod F_p$** ($F_p = 4$) depends solely on 1D position $p$:
- **$\mathfrak{S}_6$ Alphabet Equivariance:** Permuting $\Sigma$ by $\tau \in \mathfrak{S}_6$ permutes the bigram axes $(\sigma_0, \sigma_1) \mapsto (\tau(\sigma_0), \tau(\sigma_1))$ identically across every fiber $f \in \{0,1,2,3\}$.
- **Rule-Set & Frontier-Set Equivariance:** Summing over oriented rules $e$ is invariant to rule ordering, and per-word fiber banks $B_{w,\text{bi}}^{(k)} \in \mathbb{R}_{\ge 0}^{F_p \times (L-1) \times 6 \times 6 \times C_b}$ are equivariant under frontier permutations $\mathfrak{S}_{m_0} \times \mathfrak{S}_{m_1}$.
- **Empirical & Exact Support Equivalence:** On benchmark rewrite systems ($L \in \{10, 11\}$, $12$ reversible rules, $K = 3$), both symbolic $\mathcal{A}_{\text{prov4}\otimes\text{adj2}}$ and neural `FastProvenanceBigramCell` achieve **$100.00\%$ recall** (`0` false negatives) while cutting false-positive symbol-position cells by **$98.6\%$** relative to LBF-16's $\mathcal{A}_1$ (`99.37%` vs `68.16%` precision).

---

## 3. Factored Causal Mechanisms, Glyph Calibration, and Matched Compute

1. **Five Independent Causal Interventions (Factoring LBF-16's Combined Sham):**
   - `scramble_window`: permutes word/edge coordinates solely inside rewrite-window matching (`pos_cell` and `bi_cell`), leaving 1D `ribbon_conv`, `AxialSetFrescoBlock`, and recurrent banks intact.
   - `scramble_ribbon`: permutes 1D coordinates solely around `ribbon_conv`, leaving rewrite-window matching and banks intact.
   - `ablate_axial`: restricts `AxialSetFrescoBlock` at inference to diagonal self-values and zero partner centering, isolating cross-frontier partner competition with identical trained weights.
   - `env_zero` / `env_roll`: zeroes or cyclically shifts the word/provenance envelope banks $(B_w, B_{w,\text{bi}})$ at $k \ge 2$ while keeping the cross-frontier matrix bank $B_m$ intact.
   - `mat_zero` / `mat_roll`: zeroes or cyclically shifts the cross-frontier matrix bank $B_m$ at $k \ge 2$ while keeping $(B_w, B_{w,\text{bi}})$ intact.
2. **Calibrated Subterm Glyph Brightness:**
   Define zero-preserving $\mathfrak{S}_6$-invariant subterm brightness $\beta_w^{(k)}[p, \sigma] \in [0, 1)$ via $1 - \exp(-w_{\text{cal}}^\top z_{w,p,\sigma}^{(k)})$ with $w_{\text{cal}} = \operatorname{softplus}(\theta_{\text{cal}}) > 0$, mapping unsupported subterms ($z = \mathbf{0}$) to exact brightness $0$ while grading supported subterms against train-only concrete $k$-ball reachability $\alpha_1(F_R^k(\{w\}))$ (`bounded_balls`, zero extra oracle work).
3. **Matched Compute Accounting Across Budgets $B \in \{16, 32, 64, 96, 128\}$:**
   - Every legal successor examined during search or lookahead increments `s.work += 1` (including duplicates and `exact_lookahead` lookahead expansions).
   - Separate reporting of wall-clock latency (ms), exact `FlopCounterMode` neural FLOPs, rule-window checks, pairwise frontier evaluations, and peak tensor memory bytes across both parameter-matched (`param_nonrecurrent`, $K=1$) and parameter-plus-FLOPs-matched (`compute_nonrecurrent`, $K=1$, $\ge 1.04\times$ 3-pass FLOPs) nonrecurrent controllers.
