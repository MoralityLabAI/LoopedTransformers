# Paper outline for the Lite Braille Fresco programme

Companion to [`REFEREE_REPORT.md`](REFEREE_REPORT.md). It proposes the paper the evidence can carry, and marks what must exist before submission. Section references such as M1 or CE-3 point into the report.

---

## 1. The strongest defensible central theorem

Two statements, proved separately and used together.

**Setting.** Finite alphabet Σ, word length L ≥ 2, a finite set R of reversible length-preserving rules with window length in [2, ℓ_max]. `F_R` is the reflexive one-step operator on 𝒫(Σ^L). `A₂` is the lattice of position-indexed adjacent-pair tables with `(α₂, γ₂)`. For a modulus m, the fibered domain is `A₂^{1+m}` with concretisation `γ(D) = ⋃ᵢ γ₂(Dᵢ)`: component 0 holds `α₂({w₀})` and stays fixed; a Pass-1 redex at position p is routed to fiber `p mod m`.

**Guarded transfer.** At Pass 2, the firing of `x → y` at position p in fiber f is suppressed iff `p ≡ f (mod m)`, `y = w₀[p : p+|y|]`, and no Pass-1 redex at p is longer than `|y|`. No suppression at later passes. Otherwise each fiber evolves by the best `A₂` transformer `α₂ ∘ F_R ∘ γ₂`, computed locally on trellis-reduced tables.

> **Theorem A (soundness of the guarded congruence-fibered domain).** If `m ≥ ℓ_max`, then for every R, every `w₀` and every `k ≥ 0`, `F_R^k({w₀}) ⊆ γ(D^{(k)})`. The guard cannot be dropped: without it `R* = {abc↔def, ab↔de, bf↔cc}` loses a word at Pass 2 and a symbol at Pass 3.

*Proof sketch.* Invariant: every concrete word within k steps is `w₀` or lies in some fiber. A suppressed firing at `(p, ℓ)` in fiber f can apply to a concrete Pass-1 word created at `(p′, ℓ′)` with `p′ ≡ p`. If `p′ ≠ p` the windows are disjoint because `m ≥ ℓ_max`, the word agrees with `w₀` on `[p, p+ℓ)`, and the firing would need `x = y`. If `p′ = p` the guard gives `ℓ′ ≤ ℓ`, so the firing restores `w₀`, which component 0 holds. Unsuppressed firings are covered by step-soundness of the best transformer, and fibers are monotone.

> **Theorem B (architecture-determined support).** Let `N_θ` be the K-pass envelope cell assembled from non-negative sums, products, softplus-weighted linear maps, sigmoid gates, `tanh`, clamps to `[0, c]` and rational-saturation masks. Over the reals, for every parameter vector θ and every `k ≤ K`, the strict-positivity support of the bank `B_θ^{(k)}` equals `D^{(k)}`, cell for cell and fiber for fiber.

*Proof sketch.* The support map from `(ℝ_{≥0}, +, ×)` to the Booleans is a semiring homomorphism, and each listed primitive is zero-preserving and strictly positive on positive inputs, so it commutes with support. Induct on the computation graph.

> **Corollary (weight-independent soundness).** For every θ, a zero in every fiber of the bank at Pass k, together with a zero in the root encoding, certifies that the corresponding adjacent pair occurs in no word reachable within k steps.

**Finite-precision statement, as a measured proposition, not a theorem.** In float32 the equality holds while the smallest positive activation stays above underflow. On the archived `rel_prov_opened` checkpoints that minimum is 0.34 at K = 3 and decays by one to two orders of magnitude per further pass (CE-3). State the depth range for which the margin was verified, and ship a run-time check that fails closed when the margin drops below a stated bound.

**Why this is the right centre.** It is true after the repair, it is the part nobody else has, and it explains the one property that makes the architecture interesting: training cannot break soundness. Theorem A is elementary and should be proved in Lean in full generality, replacing the three instance checks as the formal core. Theorem B can stay on paper, or be formalised over the reals with Mathlib if the zero-dependency constraint is relaxed.

**Supporting lemmas with prior art, to be cited, not claimed:** the three Galois connections and their refinement chain; trellis reduction as chain arc consistency; the local form of the best `A₂` transformer; the last-exit decomposition.

**A proposition the paper needs and does not have.** Pass k of the fibered domain is not "k concrete steps". State an explicit bound of the form `γ(D^{(k)}) ⊆ F_R^{g(k)}`-closure-plus-chimeras, or give the horizon-overshoot measurements (CE-5) as the honest substitute.

---

## 2. The strongest defensible empirical contribution

> On 48 held-out random rewrite systems (L ∈ {10, 11}, twelve rules), a learned readout over a sound abstract envelope that generates no concrete successors closes 48 of 48 problems within 128 charged successor checks, using 29–30 on average. The best registered zero-parameter front-to-front heuristic closes 39 of 48 at 59; bidirectional BFS closes 30 of 48 at 84.5. A zero-parameter controller given exact three-step lookahead closes 48 of 48 at 31.4 after generating about 3,400 uncharged successors per episode. The envelope and readout therefore reproduce the decisions of exact three-step lookahead without concrete expansion.

Stated with its conditions, all of which belong in the abstract or the first page:

- Calibration cohort, n = 48 roots, one-sided 95% lower bound on closure 94.0%; confirmation on the sealed cohorts pending.
- "Charged successor checks" excludes the controller's own computation. Counting each word's successors once, the controller enumerates 125 against 148 for BFS run to closure (p = 0.15).
- In this domain a successor check costs microseconds and the controller costs about 280 ms per episode. The result matters only where successor checks cost milliseconds or more.
- A symbolic envelope with the same learned readout performs the same (30.1). The paper does not show that neural recurrence adds anything.

**Second contribution: reproducibility.** Bit-exact replication of 1,680 episodes on another OS and torch version; 2,496 Lean declarations rechecked; deterministic suites; matched training draws. Few papers in this area can say this, and it should be said with the audit's numbers.

**Third contribution: the negative results.** LBF-14 and LBF-15 missed their screens and say so; LBF-16's support theorem was refuted by LBF-17; this review refutes LBF-17's suppression lemma and supplies the repair. A paper that presents the sequence of corrections is more credible than one that presents the endpoint.

---

## 3. What the paper must not claim

| Current wording | Replace with |
|---|---|
| "53K-parameter recurrent neural-symbolic architecture implementing … abstract interpretation" as a learned capability | The abstract interpreter is written into the architecture; about 53K parameters learn a readout and magnitudes on a fixed support |
| "100% sound for all word lengths" | Theorem A, with the guard |
| "Cell-for-cell identical Boolean support" | Theorem B over the reals; measured in float32 at K = 3 on three checkpoints |
| "Formally verified" | "Generic soundness schema and Theorem A proved in Lean; Theorem B on paper" |
| "−65.5% work versus bidirectional search" | The charged figure next to the enumerated-once figure and time to closure |
| "Beats exact lookahead" | "Matches uncharged exact three-step lookahead without generating successors" |
| "Multi-hop recurrence beats compute-matched single pass" | The interval, and the tie with the symbolic-envelope control |
| "80.1% / 19.9%" | The 2 × 2 table with its interaction and interval; "dependence", not "contribution" |
| "Calibrated glyph brightness, AUROC 0.9985" | Support alone gives 0.9897; report the increment and its sample |
| "Provenance" | "Congruence trace partition" or "first-step fibers" |
| "OOD" | "Length shift to L = 13" |
| "Dependent type transport" | Drop, or state that it is the same equality proof transported |
| "False positives", "chimeras" for all excess support | Separate horizon overshoot from never-reachable cells |
| "Peak tensor memory", "exact FLOPs", wall clock from the registered runner | Measured peak allocation; FLOP coverage stated; uninstrumented timing |

---

## 4. Outline

**Working titles.**
- *Sound by construction: an equivariant network whose support is an abstract interpreter*
- *Congruence-fibered relational abstraction inside a small recurrent network*
- *What a learned readout adds to a sound abstract envelope in bidirectional rewrite search*

**Abstract (shape).** One sentence on the setting. One on Theorem A and B. One on the empirical result with n, the interval and the accounting caveat. One on what is not shown (recurrence beyond symbolic propagation; wall-clock benefit). One on reproducibility.

**1. Introduction.** Bounded bidirectional search in length-preserving equational theories as a testbed where a successor check stands for an expensive verifier call. The idea: put a sound abstract interpreter in the forward pass so that learning ranks and cannot unsoundly prune. Three contributions (§1, §2 above). A short paragraph of non-claims.

**2. Setting and accounting.** Rewrite systems, the fringe state, charged work, certificates. A subsection that defines what the controller computes without charge, with three reported quantities throughout: charged checks, successors enumerated once, time to closure. Break-even successor cost.

**3. Abstract domains for fixed-length words.** `A₁`, `A₂`, `A₃`; Galois connections; trellis reduction; local best transformer. Counterexamples 1–3 from the specification. Positioned against regular model checking, chain consistency and abstraction heuristics in planning.

**4. Congruence fibers.** Trace partitioning by first-redex residue; why m = 4 for ℓ_max = 3; `R*` and the guard; Theorem A. Collision statistics and the fiber-count sweep. Horizon semantics: the single-rule witness and the distance histogram.

**5. Architecture.** The positivity-preserving cell; Theorem B; the float32 margin and the fail-closed check; `S₆` and frontier-set equivariance, with the trajectory-level test; the readout (ribbon, axial block, heads). One table listing which parameters can and cannot affect support.

**6. Formalisation.** What is machine-checked (schema, Galois connections, refinements, Theorem A, instances) and what is not (Theorem B, equivariance, anything in float32). Certificates: what a path certificate states.

**7. Experiments.**
- 7.1 Design: registration, cohorts, deviations table (parameter counts, added controls), statistics (root as unit, sign-flip tests, family-wise adjustment, crossed root × seed model with at least ten seeds).
- 7.2 Controls: `bidir`, `geom_mm`, `graph_bridge`; `absint_mm`, `exact_mm`; a Hamming-only learned readout; `sym_prov_readout` as primary comparator; single-pass arms.
- 7.3 Main table: closures, charged work, enumerated-once successors, uninstrumented time, for every arm.
- 7.4 Where the gain comes from: the staircase from `bidir` to the primary arm.
- 7.5 Recurrence: the contrasts with intervals; the 2 × 2 with the joint cell; the retraining 2 × 2.
- 7.6 Soundness in practice: false negatives on every frontier word, before and after the repair; nested-pair cohort.
- 7.7 Scaling: L up to 24 and budgets at which BFS needs 10⁴–10⁶ nodes; K up to 8 with margins.

**8. Confirmation.** Sealed-cohort results under the frozen protocol, then the beacon-seeded fresh cohort.

**9. When abstraction pays.** Synthetic successor latency from 0 to 100 ms; native Lean `rw` execution in the loop; break-even curve against `bidir`, `geom_mm` and `exact_mm`. Incremental caching belongs here as an engineering subsection.

**10. Related work.** Abstract interpretation and trace partitioning; regular and tree-automata model checking; abstraction heuristics and bidirectional heuristic search; semiring provenance and differentiable reasoning; neural guidance of provers and solvers; equivariant set architectures.

**11. Limitations.** Fixed length, bounded windows, enumerable rules; depth limited by margin decay; no compression at the scale tested; precision confounded with horizon; one generator.

**Appendices.** Registrations verbatim with timestamps; deviations; full contrast tables; reproducibility checklist and audit scripts; Lean listing.

**Figures.** (1) The domain hierarchy on Counterexample 1. (2) `R*`: fibers at Passes 1–3 with and without the guard. (3) The staircase of §7.4. (4) Break-even curve of §9. (5) Margin decay against pass index. One fresco plate as an illustration, captioned as a rendering of symbolic support.

---

## 5. Gate before submission

| # | Item | Status at the reviewed commit |
|---:|---|---|
| 1 | Guarded suppression in `symbolic_prov4_adj2` and `FastProvenanceBigramCell`; Theorem A in Lean | Missing; counterexample and repair in `audit/` |
| 2 | Controls of §7.2 registered and run | `sym_prov_readout` exists; the others are post hoc in `audit_06` |
| 3 | Enumerated-once successors and uninstrumented time in every table | Missing; measured in `audit_08`, `audit_09` |
| 4 | Sealed-cohort confirmation, then fresh cohort | Not run; protocol in the report, §11 |
| 5 | At least a synthetic latency sweep | Missing |
| 6 | Ten or more training seeds for the four key arms | Three |
| 7 | Per-episode records at every budget archived; certificate sources archived; CI green on the campaign's tests | Missing |
| 8 | Hashes over canonical LF bytes or `-text` attributes for attested files | Missing |

Items 1–4 are required for any venue. With them the paper is a sound contribution to neural-symbolic proof search at workshop scale. Items 5–6 decide whether it supports a claim about efficient recurrent reasoning.
