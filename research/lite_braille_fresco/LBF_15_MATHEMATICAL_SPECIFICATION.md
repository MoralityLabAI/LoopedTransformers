# LBF-15 Mathematical Specification: Geometric Meeting-Potential Fresco

**Date:** 2026-11-24 · **Status:** Formal derivation prior to LBF-15 implementation and suite generation · **Parent analysis:** [`LBF_14_MECHANISTIC_AND_FRONTIER_ANALYSIS.md`](LBF_14_MECHANISTIC_AND_FRONTIER_ANALYSIS.md).

---

## 1. Bidirectional Proof-Search State and Planar Tree-Fringe Manifold

Let $\Sigma = \{\texttt{a},\texttt{b},\texttt{c},\texttt{d},\texttt{e},\texttt{f}\}$ be the public unary-operator alphabet, $x_0, x_\star \in \Sigma^L$ ($x_0 \ne x_\star$) the source and target terms of common length $L$, and
\[
R = \bigl\{(a_r, b_r)\bigr\}_{r=1}^{|R|}, \qquad |a_r| = |b_r| \in \{2, 3\}
\]
a finite system of universally quantified reversible equations $\forall z : A,\; \tau(a_r, z) = \tau(b_r, z)$. Together $R$ induces a symmetric multigraph $\mathcal{G}_R = (\Sigma^L, \mathcal{E}_R)$ whose edges are legal oriented rewrites $m = (r, \delta, p, w')$ with $r \in \{0,\dots,|R|-1\}$, $\delta \in \{0,1\}$, and subword match at index $p \in \{0,\dots,L-|a_r|\}$.

At outer search step $t \ge 0$ under total examined-successor cap $B = 128$ and proof-length cap $D = 16$, let $W^{(t)} \in \{0,\dots,B\}$ denote cumulative examined legal successors and $b^{(t)} = B - W^{(t)}$ the remaining successor budget. The public bidirectional search state is
\[
S^{(t)} = \Bigl(\mathcal{U}^{(t)},\; \mathcal{V}^{(t)},\; V_0^{(t)},\; V_1^{(t)},\; g_0,\; g_1,\; \Pi_0,\; \Pi_1,\; R,\; b^{(t)}\Bigr),
\]
where:
- $V_0^{(t)}, V_1^{(t)} \subset \Sigma^L$ are the sets of already-discovered forward and backward words, with tree-depth maps $g_0 : V_0^{(t)} \to \{0,\dots,D\}$ and $g_1 : V_1^{(t)} \to \{0,\dots,D\}$ ($g_0(x_0)=0$, $g_1(x_\star)=0$) and parent pointers $\Pi_0, \Pi_1$.
- $\mathcal{U}^{(t)} = (u_1^{(t)}, \dots, u_{m_0}^{(t)}) \subseteq V_0^{(t)}$ and $\mathcal{V}^{(t)} = (v_1^{(t)}, \dots, v_{m_1}^{(t)}) \subseteq V_1^{(t)}$ are the **planar tree-fringe ordered** sequences of open (unexpanded) frontier words with $g_s < D$.

**Planar tree-fringe update law:** Expanding action $a = (s, i_\star) \in \mathcal{A}(S^{(t)}) := (\{0\}\times\{1,\dots,m_0\}) \sqcup (\{1\}\times\{1,\dots,m_1\})$ enumerates $\mathcal{N}_R(w_\star)$ in canonical tuple order $(r, \delta, p, w')$, charging $\Delta W = 1$ per examined move (including moves into $V_s^{(t)}$, stopping immediately if $W = B$ or if a successor $w' \in V_{1-s}^{(t)}$ satisfies $g_s(w_\star) + 1 + g_{1-s}(w') \le D$). Novel successors $(w'_1, \dots, w'_q)$ with $g_s(w_\star) + 1 < D$ are spliced in place of $w_\star$ at index $i_\star$. Inductively, $\mathcal{U}^{(t)}$ and $\mathcal{V}^{(t)}$ remain exact left-to-right leaves of the forward and backward search trees: siblings share $|i - i'| = 1$ and first cousins occupy contiguous index intervals.

---

## 2. Four Distinct Value Quantities in Bounded Proof Search

Prior null results in LBF-12, LBF-13, and LBF-14 arose from conflating distinct search quantities. We separate four formally distinct functions on $S^{(t)}$:

### 2.1 Optimal Path Length $\ell^*(u_i, v_j)$ (Graph Geodesic)
Let $d_R^*(u, v) := \inf\{k \ge 0 : u \xrightarrow{\mathcal{G}_R}^k v\}$ be the unweighted graph distance in $\mathcal{G}_R$. The optimal path length constrained to pass through discovered pair $(u_i, v_j) \in \mathcal{U}^{(t)} \times \mathcal{V}^{(t)}$ is
\[
\ell^*(u_i, v_j) := g_0(u_i) + d_R^*(u_i, v_j) + g_1(v_j), \qquad \ell^*(a) := \min_{\text{opp}} \ell^*(u_i, v_j).
\]
Because each rewrite alters at most $\max_r |a_r| \le 3$ characters, the public metric bound
\[
\underline{d}(u, v) := \left\lceil \frac{d_H(u, v)}{3} \right\rceil \le d_R^*(u, v)
\]
is strictly admissible, so $g_0(u_i) + \max(1, \underline{d}(u_i, v_j)) + g_1(v_j) > D \implies \ell^*(u_i, v_j) > D$. However, **small $\ell^*(a)$ is neither necessary nor sufficient for minimal search cost**: a state on a length-6 witness with branching factor $14$ may require more successor evaluations than a state on a length-8 witness with branching factor $2$.

### 2.2 Fixed-Policy Continuation Utility $U_{\pi_0}^{\text{cont}}(S^{(t)}, a)$
Fix a deterministic baseline continuation policy $\pi_0$ on open-frontier states. Let $S^{(t)} \oplus_b a$ be the state after executing forced expansion $a$ under remaining allowance $b$, and let $\Delta W_{\pi_0}(S^{(t)}, a; b) \in \{1, \dots, b\}$ be total examined successors consumed by $a$ followed by $\pi_0$ until closure or budget exhaustion. Define censored continuation cost and normalized continuation utility:
\[
C_{\pi_0}(S^{(t)}, a; b) :=
\begin{cases}
\Delta W_{\pi_0}(S^{(t)}, a; b), & \text{if closure with path length } \le D \text{ occurs within } b, \\
b + 1, & \text{otherwise},
\end{cases}
\qquad
U_{\pi_0}^{\text{cont}}(S^{(t)}, a; b) := 1 - \frac{C_{\pi_0}(S^{(t)}, a; b)}{b + 1} \in \Bigl[0, \frac{b}{b+1}\Bigr].
\]
**Baseline contamination property:** Suppose $\mathcal{U}^{(t)}$ already contains a 1-step closing state $a^\star$ ($C_{\pi_0}(S^{(t)}, a^\star; b) = c^\star \ll b$) alongside an irrelevant dead-end leaf $a_{\text{dead}}$ with degree $\deg(a_{\text{dead}}) = 3$. Force-expanding $a_{\text{dead}}$ wastes $3$ units of budget, after which $\pi_0$ immediately expands $a^\star$, yielding $C_{\pi_0}(S^{(t)}, a_{\text{dead}}; b) = 3 + c^\star$. Thus $U_{\pi_0}^{\text{cont}}(S^{(t)}, a_{\text{dead}}; b) = U_{\pi_0}^{\text{cont}}(S^{(t)}, a^\star; b) - \frac{3}{b+1}$ remains near $1$ despite $a_{\text{dead}}$ contributing zero proof progress.

### 2.3 Marginal Frontier Utility $\Delta U_{\pi_0}^{\text{marg}}(S^{(t)}, a)$ and Relative Advantage $A_{\pi_0}(S^{(t)}, a)$
To isolate the **causal contribution of candidate $a$ given the already-populated frontier**, compare forced expansion $S^{(t)} \oplus_b a$ against the **counterfactual frontier-deletion state** $S^{(t)} \ominus a$ (removing $a$ from its open frontier without expanding it; if $a$ is the sole open state on its side so bidirectional meeting through that frontier is impossible without $V_s^{(t)}$, run $\pi_0$ on the remaining frontier or assign $b+1$ upon exhaustion):
\[
\Delta U_{\pi_0}^{\text{marg}}(S^{(t)}, a; b) := U_{\pi_0}^{\text{cont}}(S^{(t)}, a; b) - U_{\pi_0}^{\text{cont}}(S^{(t)} \ominus a; b) = \frac{C_{\pi_0}(S^{(t)} \ominus a; b) - C_{\pi_0}(S^{(t)}, a; b)}{b + 1} \in [-1, +1],
\]
together with the **within-menu continuation advantage**
\[
A_{\pi_0}(S^{(t)}, a; b) := U_{\pi_0}^{\text{cont}}(S^{(t)}, a; b) - \max_{a' \in \mathcal{A}(S^{(t)})} U_{\pi_0}^{\text{cont}}(S^{(t)}, a'; b) = -\frac{C_{\pi_0}(S^{(t)}, a; b) - \min_{a'} C_{\pi_0}(S^{(t)}, a'; b)}{b + 1} \le 0.
\]
For the dead leaf $a_{\text{dead}}$ above, deleting $a_{\text{dead}}$ avoids any risk of $\pi_0$ visiting it ($C_{\pi_0}(S^{(t)} \ominus a_{\text{dead}}) \le c^\star < 3 + c^\star$), so $\Delta U_{\pi_0}^{\text{marg}}(S^{(t)}, a_{\text{dead}}) \le -\frac{3}{b+1} < 0$ (**strictly negative marginal utility**), whereas deleting the unique bridge $a^\star$ spikes cost to $b+1$, giving $\Delta U_{\pi_0}^{\text{marg}}(S^{(t)}, a^\star) = \frac{b + 1 - c^\star}{b + 1} \gg 0$ (**strictly positive marginal utility**).

### 2.4 Actual Search Cost $W(\pi, c, B)$ and Penalized Cost $\widetilde{W}(\pi, c, B)$
For any closed-loop policy $\pi$ executed from $S^{(0)}$ on problem $c$ under cap $B$:
- **Actual search cost** $W(\pi, c, B) \in \{0, \dots, B\}$ counts every legal successor examined in $\mathcal{N}_R(w)$, including duplicates.
- **Penalized search cost** $\widetilde{W}(\pi, c, B) := W(\pi, c, B)\,\mathbb{I}[\text{certified}(c, \pi)] + (B + 1)\bigl(1 - \mathbb{I}[\text{certified}(c, \pi)]\bigr)$, where $\text{certified}(c, \pi) = 1$ iff $\pi$ outputs a witness of length $\le D$ that replays from $(x_0, x_\star, R)$ and compiles standalone and axiom-free in Lean 4.19.0.
- **Auxiliary computational costs** — cross-frontier pair evaluations $N_{\text{pair}} = \sum_t m_0^{(t)} m_1^{(t)}$, neural policy calls $N_{\text{call}}$, inner recurrent passes $K N_{\text{call}}$, CPU runtime, and offline Lean kernel verification time — are accounted separately.

---

## 3. Cross-Frontier Potential Matrix and Recurrent Fresco Architecture

### 3.1 Cross-Frontier Potential Matrix $M^{(t, k)} \in \mathbb{R}^{m_0 \times m_1}$
At outer decision $t$ and inner recurrent pass $k \in \{1, \dots, K\}$, we construct a structured latent tensor $H^{(t, k)} \in \mathbb{R}^{m_0 \times m_1 \times d_m}$ and scalar meeting-potential matrix
\[
M_{ij}^{(t, k)} = \phi_\theta^{(k)}\!\bigl(u_i^{(t)}, v_j^{(t)}, R, b^{(t)}, \mathcal{U}^{(t)}, \mathcal{V}^{(t)}, \pi_{ij}^{(t)}\bigr).
\]

#### (a) Associative Rule Operator and Word-Position Recurrence
Embed each directed rewrite $(x_e \to y_e)$ ($e = 1, \dots, 2|R|$, padded to length $3$) into key-value pairs $(k_e, v_e) \in \mathbb{R}^d \times \mathbb{R}^d$:
\[
k_e = \tanh\!\bigl(W_{\text{lhs}} [E(x_e);\, |x_e|/3]\bigr), \qquad v_e = W_{\text{rhs}} [E(y_e) - E(x_e);\, |x_e|/3].
\]
Initialize per-position word states $U^{(0)} \in \mathbb{R}^{m_0 \times L \times d}$ and $V^{(0)} \in \mathbb{R}^{m_1 \times L \times d}$ from character, position, and frontier-side embeddings, with zero initial word banks $B_{U}^{(0)} = 0, B_{V}^{(0)} = 0$. At pass $k = 1, \dots, K$:
\[
\widetilde{U}^{(k)} = U^{(k-1)} + G_{\text{word}}\!\bigl(U^{(k-1)}, \mathcal{I}_{\text{mode}}(B_U^{(k-1)})\bigr), \qquad
Z_U^{(k)} = \operatorname{GELU}\!\bigl(\operatorname{Conv}_{1\text{d}, 3}(\widetilde{U}^{(k)})\bigr),
\]
\[
\Delta_R U^{(k)} = \operatorname{Softmax}\!\Bigl(\frac{(Z_U^{(k)} W_Q) K_R^\top}{\sqrt{d}}\Bigr) V_R, \qquad
U^{(k)} = \widetilde{U}^{(k)} + \operatorname{FFN}_w\!\bigl(\widetilde{U}^{(k)} + Z_U^{(k)} + \tfrac{1}{2}\Delta_R U^{(k)}\bigr),
\]
\[
B_U^{(k)} = \operatorname{LayerNorm}\!\bigl(\Delta_R U^{(k)} + \tfrac{1}{2} \mathcal{I}_{\text{mode}}(B_U^{(k-1)})\bigr),
\]
and identically for $(V^{(k)}, B_V^{(k)})$, where $G(x, m) = \sigma(W_g[x; m] + b_g) \odot (W_p m)$ is the gated message injector (`b_g = -6` for `standard`, `b_g = -2` for `opened`). Across $K$ passes, repeated application of $\operatorname{Conv}_{1\text{d}, 3}$ and rule attention $(K_R, V_R)$ propagates **overlapping multi-step rewrite compatibility** across adjacent character positions without generating successor strings.

#### (b) Position-Aligned Cross-Frontier Coupling and 2D Axial/Spatial Propagation
Project position states via $W_{dm} \in \mathbb{R}^{d \times d_m}$ and compute position-aligned inner-product features alongside twelve purely public geometric/budget/persistence scalars $f_{ij}^{\text{pub}} \in \mathbb{R}^{12}$ (normalized $g_0, g_1, g_0+g_1, d_H/L, \underline{d}/D$, depth slack, admissibility indicator $\mathbb{I}[g_0 + g_1 + \max(1,\underline{d}) \le D]$, mismatch span, $b^{(t)}/B$, log-frontier balance $\log\frac{1+m_0}{1+m_1}$, survival indicator $\mathbb{I}[(u_i, v_j) \in \mathcal{P}^{(t-1)}]$, and centered prior-step potential $\tilde{p}_{ij}^{(t-1)}$):
\[
A_{ij}^{(k)} = \frac{1}{L} \sum_{p=1}^{L} (U_{i,p}^{(k)} W_{dm}) \odot (V_{j,p}^{(k)} W_{dm}) \in \mathbb{R}^{d_m}, \qquad
P_{ij}^{(k)} = \psi_{\text{pair}}\!\bigl([A_{ij}^{(k)};\, f_{ij}^{\text{pub}}]\bigr).
\]
With $H^{(0)} = P^{(1)}$ (and $H_{\text{in}}^{(k)} = H^{(k-1)} + \tfrac{1}{2} P^{(k)}$ for $k \ge 2$) and matrix message bank $B_M^{(0)} = 0$:
\[
\widetilde{H}^{(k)} = H_{\text{in}}^{(k)} + G_{\text{mat}}\!\bigl(H_{\text{in}}^{(k)}, \mathcal{I}_{\text{mode}}(B_M^{(k-1)})\bigr).
\]
Structured cross-frontier propagation combines **row-wise forward-to-backward attention**, **column-wise backward-to-forward attention**, and **planar tree-fringe $3 \times 3$ grouped spatial convolution**:
\[
C_{\text{row}, ij}^{(k)} = \sum_{j'=1}^{m_1} \alpha_{i, jj'}^{(k)} (\widetilde{H}_{ij'}^{(k)} W_{V,r}), \quad
C_{\text{col}, ij}^{(k)} = \sum_{i'=1}^{m_0} \beta_{ii', j}^{(k)} (\widetilde{H}_{i'j}^{(k)} W_{V,c}), \quad
S_{ij}^{(k)} = \operatorname{GELU}\!\bigl(\operatorname{Conv}_{2\text{d}, 3\times 3}(\widetilde{H}^{(k)})\bigr)_{ij},
\]
\[
R_{ij}^{(k)} = \tfrac{1}{2}\bigl(C_{\text{row}, ij}^{(k)} + C_{\text{col}, ij}^{(k)}\bigr) + S_{ij}^{(k)}, \qquad
H^{(k)} = \widetilde{H}^{(k)} + \operatorname{FFN}_m\!\bigl(\widetilde{H}^{(k)} + R_{ij}^{(k)}\bigr), \qquad
B_M^{(k)} = \operatorname{LayerNorm}\!\bigl(R_{ij}^{(k)} + \tfrac{1}{2}\mathcal{I}_{\text{mode}}(B_M^{(k-1)})\bigr).
\]

#### (c) Soft-Conductance Marginal Readout
Let $M_{ij}^{(t, k)} = w_{\text{pot}}^\top H_{ij}^{(k)} + b_{\text{pot}}$, and apply the exact public admissibility mask $\widehat{M}_{ij}^{(t, k)} = M_{ij}^{(t, k)}$ if $g_0(u_i) + g_1(v_j) + \max(1, \underline{d}(u_i, v_j)) \le D$ else $-12$. With learnable temperature $\tau = \operatorname{softplus}(\eta) + 0.1$, marginalize over opposite frontiers via normalized log-sum-exp (free-energy conductance) plus public node cost/degree readout $\nu_\theta$:
\[
s_{0, i}^{(t, k)} = \tau \log \frac{1}{m_1} \sum_{j=1}^{m_1} \exp\!\Bigl(\frac{\widehat{M}_{ij}^{(t, k)}}{\tau}\Bigr) + \nu_\theta\!\bigl(\bar{U}_i^{(k)}, q_0(u_i)\bigr), \qquad
s_{1, j}^{(t, k)} = \tau \log \frac{1}{m_0} \sum_{i=1}^{m_0} \exp\!\Bigl(\frac{\widehat{M}_{ij}^{(t, k)}}{\tau}\Bigr) + \nu_\theta\!\bigl(\bar{V}_j^{(k)}, q_1(v_j)\bigr),
\]
where $q_s(w)$ uses only public quantities (tree depth $g_s(w)/D$, public rule-substring match count $\operatorname{matches}(w, R)/24$ computed without generating successors, $\log(1+m_s)$, $\log(1+m_{1-s})$, $b^{(t)}/B$, and endpoint Hamming fraction). The policy selects $\arg\max_{a \in \mathcal{A}(S^{(t)})} s_a^{(t, K)}$ with deterministic tie-break (smaller frontier size, side $0$, lower fringe index).

---

## 4. Fresco Glyph Semantics and Quantitative Observables

The meeting-potential fresco projects predecision continuous tensors into both (i) canonical `lbf.observation.v1` / `lbf.encoded.v1` Braille-glyph decision records (preserving 100% compatibility with `lbf_gym.py`) and (ii) illuminated 2D matrix manuscript plates (`fresco_matrix.svg`):

| Fresco property | Mathematical definition | Glyph / manuscript encoding | Hypothesis tested |
|---|---|---|---|
| **Brightness** | $a_{ij}^{(t, k)} = \sigma(M_{ij}^{(t, k)}) \in [0, 1]$, quantized to $q \in \{0,1,2,3\}$ | Lite-Brite LED opacity; `value`/`closure` tiles | High brightness identifies multi-step rewrite bridges $(u_i, v_j)$ |
| **Changing brightness** | $\Delta_{\text{pass}} a_{ij}^{(t)} = a_{ij}^{(t, K)} - a_{ij}^{(t, 1)}$ | Pass-1 vs Pass-$K$ delta; `gate`/`memory` tiles | Positive brightening reflects multi-hop bridges invisible at Pass 1 |
| **Polarity** | $p_{ij}^{(t, k)} = \operatorname{sign}_\epsilon(M_{ij}^{(t, k)} - M_{ij}^{(t, 1)}) \in \{-1,0,+1\}$ ($\epsilon = 0.05$) | Braille dots 7/8 & emerald/teal/terracotta palette | $+1$ marks constructive recurrent amplification; $-1$ marks decoy suppression |
| **Spatial adjacency** | 2D rook-contiguity Moran's $I$ on planar fringe matrix $\sigma(M^{(t, k)})$ | $3\times 3$ planar cousin grid; tested via `scramble` | Tree-fringe cousin clusters form coherent luminous bridge patches |
| **Persistence** | Survival mask $\mathbb{I}[(u_i, v_j) \in \mathcal{P}^{(t-1)}]$ & centered $\tilde{a}_{ij}^{(t-1)}$ | Illuminated manuscript border; tested via `no_persist` | Persistent cross-step potentials stabilize search across outer expansions |

For any matrix $X \in \mathbb{R}^{m_0 \times m_1}$ with mean $\bar{x}$ and variance $s^2 > 0$ ($m_0 m_1 \ge 4$), 4-neighbor rook Moran's $I$ is
\[
I(X) = \frac{m_0 m_1}{\sum_{(i,j)\sim(i',j')} 1} \cdot \frac{\sum_{(i,j)\sim(i',j')} (X_{ij} - \bar{x})(X_{i'j'} - \bar{x})}{\sum_{i,j} (X_{ij} - \bar{x})^2} \in [-1, +1].
\]
Under the `scramble` control, row/column permutations destroy rook adjacency ($I \to 0$) while preserving exact multiset statistics of $X$.
