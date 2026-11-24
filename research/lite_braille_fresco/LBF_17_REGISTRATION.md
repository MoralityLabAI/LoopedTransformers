# LBF-17 Prospective Registration: Relational Abstract Interpretation and Provenance-Preserving Equivariant Frescoes

Registered 2026-11-24 **before** freezing or scoring the LBF-17 suite (`SEED = 20261227`), fitting any LBF-17 checkpoint, or inspecting any `20261227` instance or witness. Stacked on merged `main` commit `3230f1c50d65fdfa47c30fab789bb730c2f8c0f5` (frozen LBF-16 baseline [lbf16_equivariant_fresco.py](lbf16_equivariant_fresco.py), `53,047` parameters). Pinned rewrite simulator `cd9ac784c5804ec2fe595bb6a01d2697fc016610` / blob `204a1d012a4ebe8e80d8bd722590742b7e7f02ba`.

Mathematical specification & proofs: [LBF_17_RELATIONAL_ABSTRACT_INTERPRETATION.md](LBF_17_RELATIONAL_ABSTRACT_INTERPRETATION.md) · Standalone zero-axiom Lean 4.19.0 formalization: [lean/LBF17AbstractInterp.lean](lean/LBF17AbstractInterp.lean).

---

## 1. Lineage Custody and Sealed Cohorts

- **Fresh untouched generator seed:** `20261227` with split counts `{"train": 64, "calibration": 48, "ordinary": 16, "ood": 16}` (`144` unique canonical rule-system lineages, cryptographically verified to have `0` lineage collisions against all `720` historical lineages from seeds `20261204`, `20261205`, `20261207`, `20261209`, `20261215`, and `20261221`).
- **Sealed-cohort preservation:** Both LBF-16's existing sealed `ordinary.jsonl` and `ood.jsonl` (`SEED = 20261221`) and LBF-17's own `ordinary.jsonl` and `ood.jsonl` (`SEED = 20261227`) remain strictly unopened (`sealed_outcomes_opened = False`) throughout training, calibration evaluation, and mechanistic analysis, preserved for independent confirmation.
- **Frozen baseline:** `lbf16_equivariant_fresco.py` remains untouched on disk; its `EquivariantHorizonFrescoNet("recurrent_opened")` (`53,047` params) is trained and evaluated side-by-side on LBF-17's `20261227` suite under the exact same frame draw schedule as `lbf16_pos1_opened`.

---

## 2. Registered Controller Hierarchy and Matched Budgets

### 2.1 Symbolic, Heuristic, and Charged Exact-Lookahead Controls (`0` learned parameters)
1. `astar`: canonical unilateral A* search.
2. `bidir`: canonical smaller-frontier bidirectional BFS (verified identical to `lbf14.run(..., "bidir")`).
3. `geom_mm`: parameter-free front-to-front geometric meeting heuristic.
4. `graph_bridge`: parameter-free $k$-mer rule-graph shortest-path conductance heuristic.
5. `uniform`: 5 independent pseudorandom fringe streams (`17, 31, 47, 61, 83`).
6. `exact_lookahead`: bounded exact reachable-state set lookahead controller that probes 1-step concrete successor intersections across frontier candidates while **charging every examined legal successor (`s.work += 1`, including duplicates) directly against budget $B$**.

### 2.2 Parameter-Matched (`~53,047` params) and Compute-Matched Learned Controllers
All 8 learned architectures satisfy `53,041 <= params <= 53,089` (`<= 0.08%` spread around LBF-16's `53,047` budget) and share identical mini-batch training frame draw schedules per seed (`SEEDS = (107, 109, 113)`, `240` AdamW updates, `batch_size = 6`, cosine LR `3e-3 -> 2e-4`):
1. **`rel_prov_opened` (`53,074` params, $K=3$, `gate_bias = -1.5`) — Primary LBF-17 controller:** $F_p = 4$ spatial-congruence provenance fibers $\otimes$ trellis-filtered $6 \times 6$ adjacent-$2$-gram envelope + positionwise residual + horizon-separated subterm ribbon + `FactoredAxialFrescoBlock`.
2. `rel_prov_standard` (`53,074` params, $K=3$, `gate_bias = -6.0`): identical non-gate initialization to `rel_prov_opened` with closed initial gates.
3. `adj2_noprov_opened` (`53,074` params, $K=3$, `use_provenance = False`): merged single-fiber trellis-filtered adjacent-$2$-gram envelope (isolating provenance fiber separation).
4. `lowrank_opened` (`53,041` params, $K=3$): learned $\mathfrak{S}_6$-equivariant low-rank bilinear adjacent-symbol coupling without explicit $6 \times 6$ bigram trellis or provenance fibers.
5. `lbf16_pos1_opened` (`53,047` params, $K=3$): frozen LBF-16 baseline architecture (`EquivariantHorizonFrescoNet`, positionwise $1$-gram multiplicative envelope).
6. `param_nonrecurrent` (`53,089` params, $K=1$): single-pass parameter-matched nonrecurrent controller ($1$-hop rewrite horizon only).
7. `compute_nonrecurrent` (`53,066` params, $K=1$): single-pass **both parameter-matched and compute-matched** nonrecurrent controller ($\ge 1.04\times$ the 3-pass neural FLOPs of `rel_prov_opened` via 24 cross/self 1-hop fiber-ribbon views along $L$, with zero multi-hop envelope recurrence).
8. `sym_prov_readout` (`53,078` params, $K=1$ readout over $K=3$ exact symbolic `prov4_adj2` propagation): isolates exact symbolic relational-provenance envelope propagation + learned axial readout.

### 2.3 Five Orthogonal Factored Causal Interventions (on `rel_prov_opened`)
Evaluated across budgets without refitting:
1. `scramble_window` (rewrite-window coordinate perturbation only),
2. `scramble_ribbon` (1D `ribbon_conv` coordinate scramble only),
3. `ablate_axial` (self-only diagonal axial values & zero partner centering only),
4. `env_zero` and `env_roll` (envelope recurrent banks $(B_w, B_{w,\text{bi}})$ intervened only; $B_m$ intact),
5. `mat_zero` and `mat_roll` (cross-frontier matrix bank $B_m$ intervened only; $(B_w, B_{w,\text{bi}})$ intact).

---

## 3. Registered Budgets, Abstraction Diagnostics, and Decision Screens

- **Matched successor budgets:** `B ∈ {16, 32, 64, 96, 128}` (`WORK = 128` primary cap, `DEPTH = 16`), recording certified closures, penalized work, wall-clock latency (ms), exact `FlopCounterMode` neural FLOPs, rule-window evaluations, pairwise frontier evaluations, and peak tensor memory bytes.
- **Abstraction-error & glyph-calibration diagnostics (separately reported on `train` and `calibration`):**
  Compare concrete bounded $k=3$ reachability against $\mathcal{A}_1$ (`pos1`), $\mathcal{A}_{\text{prov1}}$ (`prov_pos1`), $\mathcal{A}_2$ (`adj2`), $\mathcal{A}_3$ (`sub3`), $\mathcal{A}_{\text{prov4}\otimes\text{adj2}}$ (`prov4_adj2`), and neural `lbf16_pos1_opened` vs `rel_prov_opened`: support precision, recall, false-positive count, false cross-frontier bridge rate, bank effective rank, oracle action-ranking regret, proof closure stratified by low vs high root abstraction error, and AUROC of fresco glyph brightness distinguishing certified concrete-reachable subterms from spurious $\mathcal{A}_1$-plausible subterms.
- **Standalone Lean 4.19.0 certification:** Every closed calibration episode at `B = 128`, all `64` training references, `24` counterfactual continuations, all `48` calibration references, and all `15` abstract-interpretation theorems in `lean/LBF17AbstractInterp.lean` must compile standalone with zero axioms (`#print axioms`).

### Pre-Declared Decision Screens
1. **Screen 1A (vs Symbolic `bidir` & Charged `exact_lookahead` at `B = 128` and Budget AUC):** `rel_prov_opened` exceeds `bidir` by $\ge +15.0\text{ pp}$ closure with $\ge 30\%$ lower penalized work at `B = 128` (all 3 seeds $>$ `bidir`), and exceeds charged `exact_lookahead` on both `B = 128` closure and 5-budget closure AUC (`B ∈ {16, 32, 64, 96, 128}`).
2. **Screen 1B (vs Strongest Explicit Heuristic $\max(\texttt{geom\_mm}, \texttt{graph\_bridge})$):** `rel_prov_opened` exceeds the strongest explicit heuristic by $\ge +5.0\text{ pp}$ closure and $\ge 20\%$ lower penalized work at `B = 128` (all 3 seeds strictly above), and exceeds it at every budget $B \in \{16, 32, 64, 96, 128\}$.
3. **Screen 2A (Multi-Pass Recurrence vs Parameter- & Compute-Matched Nonrecurrent & Closed Gates):** `rel_prov_opened` achieves higher 5-budget closure AUC (`B ∈ {16, 32, 64, 96, 128}`) and lower or equal `B = 128` penalized work than `param_nonrecurrent`, `compute_nonrecurrent`, and `rel_prov_standard`.
4. **Screen 2B (Relational Abstraction Precision & Calibrated Glyph Brightness):** Neural `rel_prov_opened` maintains $100\%$ concrete $3$-step symbol recall (`fn = 0`) while cutting calibration symbol false-positive rate by $\ge 90\%$ relative to `lbf16_pos1_opened` (`precision >= 95%`), and achieves glyph-brightness AUROC $\ge 0.90$ distinguishing certified reachable subterms from spurious $\mathcal{A}_1$-plausible subterms.
5. **Screen 2C (Factored Causal Mechanism Validity):** None of the 5 factored corruptions (`scramble_window`, `scramble_ribbon`, `ablate_axial`, `env_zero`/`env_roll`, `mat_zero`/`mat_roll`) systematically outperforms intact `rel_prov_opened` on 5-budget closure AUC (specifically `AUC(rel_prov_opened) >= AUC(sham) - 0.005` for all shams, with strictly positive causal advantage over `scramble_window`, `ablate_axial`, and `env_roll`).
