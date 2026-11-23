# LBF-15 Registration: Geometric Meeting-Potential Fresco

Registered 2026-11-24 **before** generating the LBF-15 benchmark suite, fitting any LBF-15 checkpoint, or scoring any LBF-15 root. Stacked on merged LBF-12/13/14 `main` (`1adc699efd0fa8b343a73115d5aa934633f3fda2`) and LBF-14 mechanistic diagnostics (`73d650bb`). Formal derivation in [`LBF_15_MATHEMATICAL_SPECIFICATION.md`](LBF_15_MATHEMATICAL_SPECIFICATION.md). Pinned rewrite simulator `cd9ac784c5804ec2fe595bb6a01d2697fc016610` / blob `204a1d012a4ebe8e80d8bd722590742b7e7f02ba`.

## 1. Fresh Lineage-Disjoint Suite and Custody Contract
- **Generator seed:** `20261215` (untouched prior to registration).
- **Shard counts:** `64` train, `48` calibration, `16` ordinary (sealed), `16` OOD (sealed) = `144` unique canonical rule-system lineages. Enforce zero canonical lineage overlap across shards and zero overlap with prior seeds `20261204` (LSA-02K), `20261205` (LBF-12), `20261207` (LBF-13), and `20261209` (LBF-14).
- **Sealed cohorts:** `ordinary.jsonl` and `ood.jsonl` are hash-recorded at freeze time and never deserialized by training or calibration evaluators. Calibration witnesses (`calibration_witnesses.jsonl`) are opened only after all frozen policy episodes finish, strictly for standalone Lean reference certification.
- **Proof and cost accounting:** Primary budget `WORK = 128` examined legal successors, path length cap `DEPTH = 16`. Every legal move in `rg.neighbors(case, word)` inspected during node expansion costs `1` work unit including duplicates; within-node move enumeration uses canonical sorted `Move` order identically across all policies. Public pairwise evaluations, public substring-match counts, neural forward calls, inner recurrent passes, and standalone Lean compilation are reported separately. Secondary budget-curve evaluation runs frozen checkpoints without refitting at `B ∈ {32, 64, 96, 128}`.
- **Native Lean certification:** Every training reference (`64`), audited counterfactual continuation witness (`24`: `12` equality, `12` dependent transport), successful calibration policy witness, and calibration reference witness (`48`) must replay from public endpoints and compile standalone and axiom-free in Lean `4.19.0` (`6caaee842e94`).

## 2. Train-Only Supervision Protocol
- Sample open-frontier bidirectional states exclusively from `64` training roots along `geom_mm` and exploratory bidirectional trajectories (`m_0 + m_1 > 2`, `m_0, m_1 <= 32`), populating persistent prior-step pair potentials $\tilde{p}_{ij}^{(t-1)}$ online from the rollout's previous step so persistence inputs match test-time semantics.
- Compute on training roots only:
  1. **Bounded pairwise bridge target $Y_{ij}^{\text{mat}}$:** intersect radius-3 rewrite balls around $u_i \in \mathcal{U}^{(t)}$ and $v_j \in \mathcal{V}^{(t)}$; set $Y_{ij}^{\text{mat}} = \exp(-0.45(d_R^*(u_i, v_j)-1))$ when $g_0(u_i) + d_R^*(u_i, v_j) + g_1(v_j) \le 16$ and $0$ otherwise.
  2. **Fixed-policy continuation cost $C_{\pi_0}(S^{(t)}, a; b_c)$ and counterfactual deletion marginal utility $\Delta U_{\pi_0}^{\text{marg}}(S^{(t)}, a; b_c)$** ($b_c = \min(b^{(t)}, 64)$, $\pi_0 = \texttt{geom\_mm}$), recording ties, mixed closures, and marginal-utility sign distribution.
- **Optimization:** Three independent seeds `107`, `109`, `113`; `480` AdamW updates per checkpoint (`lr = 0.002`, `weight_decay = 1e-4`, gradient clip `1.0`, pass-weighted objective $w_k = k$ combining positive-weighted matrix BCE $w_+ = 4$ and node softmax cross-entropy + marginal-utility MSE). For each seed, all four neural controller classes share the **exact same training-frame draw schedule**, and `recurrent_standard` (`gate_bias = -6.0`) vs `recurrent_opened` (`gate_bias = -2.0`) share **exact initial tensors** outside `word_gate.gate.bias` and `fresco_gate.gate.bias`.

## 3. Registered Conditions (`48` calibration roots)
1. **Symbolic & Explicit Geometric/Graph Controls (0 parameters):**
   - `astar`: pinned classical unidirectional A*.
   - `bidir`: pinned smaller-layer bidirectional BFS (exact equivalence verified on every root).
   - `geom_mm`: parameter-free explicit geometric Front-to-Front MM heuristic ($\min_v [g_0(u)+g_1(v)+\max(1,\lceil d_H(u,v)/3\rceil)]$, tie-breaking by $d_H$, mismatch span, smaller frontier).
   - `graph_bridge`: parameter-free explicit graph-theoretic bipartite $k$-mer rule-graph shortest-path conductance heuristic normalized by public rule-match degree volume.
   - `uniform`: planar fringe random selection across five deterministic streams (`17, 31, 47, 61, 83`).
2. **Matched Learned Controllers (~55.9K parameters, $\le 57,475$ budget, seeds `107, 109, 113`):**
   - `nonrecurrent` (`55,941` params, $K=1$ pass, structured fresco + untied FFN).
   - `unstructured_recurrent` (`55,865` params, $K=3$ passes, opened gates $-2.0$, pointwise MLP without cross-frontier axial attention or $3\times 3$ spatial convolution).
   - `recurrent_standard` (`55,859` params, $K=3$ passes, standard gates $-6.0$, structured axial + $3\times 3$ spatial fresco block).
   - `opened_normal` (`55,859` params, $K=3$ passes, opened gates $-2.0$, structured axial + $3\times 3$ spatial fresco block).
3. **Causal Memory & Spatial Fresco Interventions on Frozen `recurrent_opened` Checkpoints:**
   - `opened_zero`: zeroed word & matrix message banks ($k \ge 2$), preserving un-gated residual recurrence.
   - `opened_roll`: norm-preserving cyclic shift of word & matrix message banks ($k \ge 2$).
   - `opened_scramble`: deterministic spatial permutation of planar fringe rows/columns before $3\times 3$ spatial convolution and character positions before `Conv1d`, inverted immediately after.
   - `opened_no_persist`: zeroing cross-step persistence features from $t-1 \to t$.

## 4. Pre-Declared Decision Screens (`B = 128`, `48` calibration roots)
- **Primary Screen 1A (vs Strong Symbolic Bidirectional BFS):** `opened_normal` exceeds `bidir` by $\ge +10.0\text{ pp}$ certified closure AND reduces mean penalized work by $\ge 20\%$, with all three seeds strictly exceeding `bidir` closure.
- **Primary Screen 1B (vs Strongest Non-Neural Control $\max(\texttt{bidir}, \texttt{astar}, \texttt{geom\_mm}, \texttt{graph\_bridge}, \texttt{uniform})$):** `opened_normal` achieves $\ge +4.0\text{ pp}$ certified closure above the strongest non-neural control (with $\ge 50\%$ overall closure), OR ties/exceeds its closure while reducing mean penalized work by $\ge 10\%$, with non-negative mean closure contrast across seeds.
- **Secondary Architecture & Recurrence Screen:** `opened_normal` exceeds `nonrecurrent` AND `unstructured_recurrent` on closure (or ties closure with $\ge 5\%$ lower mean penalized work) and exceeds `recurrent_standard`.
- **Secondary Causal Memory & Spatial Coherence Screen:** `opened_normal` exceeds `opened_zero`, `opened_roll`, and `opened_scramble` on closure (or ties closure with lower mean penalized work), supported by continuous fresco diagnostics (gate strengths, slot/matrix effective rank, inter-pass brightening $\Delta_{\text{pass}} a_{ij}$, Moran's $I$ spatial coherence, and marginal-utility correlation).
- **Fail-closed engineering gates:** Any lineage collision, public-source blob mismatch, uncharged successor, non-equivalent `bidir` control, mutated frozen weight, first-pass sham discrepancy, roll norm drift, or rejected Lean certificate fails the campaign immediately.
