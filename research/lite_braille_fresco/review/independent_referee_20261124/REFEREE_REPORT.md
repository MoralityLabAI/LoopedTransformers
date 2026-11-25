# Independent referee report: Lite Braille Fresco, LBF-14 to LBF-17

**Object of review.** `MoralityLabAI/LoopedTransformers`, `research/lite_braille_fresco/`, commit `3d8243818858c19c5031c36e2bf7e328f831cf25`: the LBF-14–17 specifications, registrations, implementations, `lean/LBF17AbstractInterp.lean`, results documents and the `results/lbf14…lbf17` archives.

**Method.** Everything below was checked against the repository, not taken from the results documents. The registered Lean file and all proof certificates were recompiled with Lean 4.19.0 (`6caaee842e94`). Every registered B = 128 calibration episode was re-run from the archived checkpoints. Counterexamples were run against the registered code and checkpoints, and one is kernel-checked. Scripts and outputs are in [`audit/`](audit/README.md).

**Custody.** No checkpoint was modified, nothing was trained, nothing under `results/` was written. The sealed `ordinary` and `ood` files were hashed as byte streams and never parsed. Added analyses use the open calibration cohort only and are labelled post hoc.

---

## 1. Verdict

**Major revision. Not publishable as submitted.** One stated soundness theorem is false, the efficiency claim depends on an accounting convention that does not charge the controller for the successor enumeration it performs, the recurrence claim is not separated from a symbolic control, and the controls that decide the interpretation are missing. The work is reproducible to an unusual degree and contains a clean architectural idea. A reframed paper with a corrected theorem is publishable; the outline is in [`PAPER_OUTLINE.md`](PAPER_OUTLINE.md).

| # | Claim in LBF-17 | Finding |
|---|---|---|
| 1 | 144/144 closures at B = 128 | Reproduced record for record. It is 48/48 independent roots: one-sided 95% lower bound 94.0%. Nine arms reach 100%, including both single-pass controls. Bidirectional BFS closes all 48 roots within 653 successor checks. |
| 2 | Provenance envelope is a sound over-approximation, "100% recall", for all L | **False as stated.** Counterexample `R*` (§4, CE-1), on the symbolic interpreter, on all six archived `rel_prov_*` checkpoints, and kernel-checked against the Lean executable model. Repairable. |
| 3 | Neural support equals symbolic support | True at the registered K = 3 for the three `rel_prov_opened` checkpoints (0 of 68,066 cells lost). For the registered `rel_prov_standard` checkpoints, 2.1–3.6% of supported cells fall under the diagnostics' own 1e-6 threshold at K = 3, and exact float32 zeros appear at K = 4. |
| 4 | 15 zero-axiom Lean theorems | Reproduced; compiler log matches the receipt byte for byte after line-ending conversion. They are generic lattice lemmas and three finite instances. None concerns the PyTorch cell, the trellis filter, the mod-4 routing, the suppression rule or equivariance. |
| 5 | 2,584 Lean declarations | 2,496 rechecked (15 + 2,481 calibration certificates regenerated from the archived paths; source hash equals the receipt). The 88 training-side certificates were not rechecked. |
| 6 | −65.5% successor work versus `bidir` | Reproduced under the registered accounting. With the controller's implicit one-hop enumeration counted and BFS allowed to finish: 125 versus 148 successors, p = 0.15. |
| 7 | Beats exact lookahead | Only a charged variant that is weaker than `geom_mm`. An uncharged zero-parameter exact control closes 48/48 at 31.4 work. |
| 8 | Recurrence beats compute-matched single pass | Work −2.54 (p = 0.017; 0.12 after Holm over the 20 reported contrasts). AUC +1.11 pp, 95% interval [−0.69, +3.06]. |
| 9 | Neural recurrent envelope versus symbolic envelope with learned readout | No detectable difference: −0.99 work, p = 0.17; AUC +1.53 pp [−0.56, +3.89]. The neural arm costs 2.8–3.6× the time. |
| 10 | 80% / 20% envelope / matrix attribution | The joint intervention, run here, shows no interaction (−0.42 [−1.48, +0.54]). The share is 0.82 with interval [0.68, 0.95]. It measures dependence of a trained model on its banks, not the contribution of recurrence. |
| 11 | Glyph AUROC 0.9985 / 0.9899 | Numbers reproduced from the archive. Boolean support alone gives 0.9897. |
| 12 | 1,713 ms, 150 MFLOPs, 4.1 MB | Timed region contains profiling: ~5× inflation. The FLOP counter covers ~40% of CPU time. "Peak tensor" is a formula; measured peak is ~7× higher and reverses the ordering against `sym_prov_readout`. |
| 13 | Sealed cohorts preserved | File hashes verify. The seal is procedural: the cohorts are a deterministic function of a public seed, stored as plain gzip, parsed for lineage by the campaign code, and attested by hard-coded flags. |

---

## 2. Reproducibility audit

What reproduces, stated first because it is the strongest part of the submission.

| Check | Result | Script |
|---|---|---|
| Lean theory file, 15 theorems | Compiles standalone, zero axioms; source SHA-256 matches; log hash matches under CRLF | `audit_02` |
| Calibration certificates | `Proofs.lean` regenerated from archived paths: SHA-256 equals receipt (CRLF). All 2,481 declarations axiom-free on Linux. Compiler-log hash not reproduced | `audit_05` |
| B = 128 outcomes | 1,680 of 1,680 records identical in `closed`, `work` and `path`: 5 symbolic controls × 48, 8 learned arms × 144, 2 archived shams × 144, under a different OS and torch 2.15.0 | `audit_03`, `audit_09` |
| Budget curves | Identical for all seven re-run arms at all five budgets | `audit_09` |
| Manifest arithmetic | Every count, mean and all 20 registered bootstrap intervals recomputed exactly from `episodes.jsonl` | `audit_04` |
| Archive hashes | 24 checkpoints, draws, traces, teacher summary, sealed and open suite files: all match | `audit_05` |
| Suite determinism | `train` and `calibration` regenerate bit-exactly from seed 20261227 with the pinned generator | `audit_05` |
| Matched training | Draw schedules identical across the 8 arms per seed; `standard`/`opened` share initialisation | `audit_05` |
| Trajectory-level S6 equivariance | 144/144 relabelled episodes identical for each of three arms | `audit_12` |
| LBF-14, 15, 16 archives | Closure counts and mean penalised work agree with their manifests | `audit_13` |

What does not reproduce or is not archived:

- **CI.** The only run of `lbf17-relational-fresco.yml` (run 45374638265, on the merged head) failed: 29 of 30 tests passed and the Lean test errored on a toolchain path. PR #343 was opened and merged within four seconds, before that run started. The workflow covers unit tests only; the campaign ran on a local Windows machine. All five LBF-17 tests pass locally with the corrected path.
- **"30/30 tests"** counts every test under `tests/`. Five are LBF-17 tests.
- **Line endings.** Registered hashes were computed over CRLF bytes; `.gitattributes` normalises text to LF. `suite/manifest.json`, `abstraction_diagnostics.json` and both SVGs fail verification as checked out and pass after LF→CRLF conversion.
- **Fingerprints.** `initial`/`final` are SHA-256 of `torch.save` byte streams and do not reproduce across torch versions. Checkpoint file hashes do.
- **Not archived:** certificate sources and compiler logs (receipts only), witness files, per-episode outcomes at B < 128 (9,600 episodes summarised as means), and the LBF-16 suite directory.
- **Simulator.** `rewrite_gym.py` is absent from the tree at the reviewed commit and must be extracted from `cd9ac784`.
- **Record counts.** LBF-15 and LBF-16 documents state 1,560 episode records; each archive holds 1,584.

---

## 3. Major objections

### M1. The provenance soundness theorem is false as stated, and the formalisation does not cover it

Theorem 2.2(3) asserts that suppressing Pass-2 rewrites at a fiber's origin window whose right-hand side equals the root's slice is sound for all word lengths. The implementation (`symbolic_prov4_adj2`, and `FastProvenanceBigramCell` at `pass_idx == 1`) suppresses any rule at any position `p ≡ f (mod 4)` whose right-hand side equals `w0[p:p+ℓ]`. A fiber merges every Pass-1 redex at that residue, including a length-3 and a length-2 redex at the same position. Restoring a length-2 slice of a word created by a length-3 redex does not return the root.

`R* = {abc ↔ def, ab ↔ de, bf ↔ cc}`, `w0 = abc·a⁷`. Concretely `abc → def → abf → acc`. The rewrite `de → ab` at `p = 0` is suppressed at Pass 2, so `abf…` is in no fiber's concretisation, and at K = 3 the cell (position 1, `c`) is missing although it is reachable in three steps. On all six `rel_prov_opened` / `rel_prov_standard` checkpoints the activation there is exactly 0.0. `adj2_noprov_opened`, `symbolic_adj2` and `symbolic_prov_pos1` are sound on the same instance, so the defect is the suppression rule.

Scope of the defect, measured:

- It requires a rule pair `(x₀x₁x₂ ↔ y₀y₁y₂, x₀x₁ ↔ y₀y₁)` with `x₂ ≠ y₂`. The generator produces one in 4.1% of systems (245 of 6,000 drawn from audit-only seeds; analytic 4.1%). Five of the 112 open LBF-17 systems have one.
- Given the pair, 26 of 245 systems (10.6%) showed a symbol-level false negative on at least one of about eight frontier-like words; 148 of 245 did on a word containing the triple. In a fully checked sample of 400 systems (3,398 words), no system without the pair showed one.
- Inside the registered open cohort, planted words in `train_001_506860d0ab2e` (6 of 18) and `train_062_69d38459ac19` (8 of 16) exhibit it. No word within radius 5 of an endpoint in the five affected open systems did.
- The registered diagnostic scored at most four words per root on 80 roots. At the measured rate it had roughly a two-in-three chance of observing no false negative. "fn = 0" is not evidence of soundness.

The Lean file does not help. `nonbacktracking_provenance_soundness` is a last-exit decomposition of concrete walks; it justifies removing the *word* `w0` from fibers. The file's executable model `stepPos1NonBacktrack` instead suppresses rewrites whose right-hand side equals the root slice, at every position and every pass, and is checked only on three instances. [`audit/LBF17AuditAppendix.lean`](audit/LBF17AuditAppendix.lean) proves by `rfl`, against the registered definitions and with zero axioms, that on `R*` this model does not support `c` at position 1 at passes 3, 4 and 8 and does not contain `abf` at passes 2, 4 and 8, while the file's other model, `runNonBacktrackProvAdj2Words`, which filters words, is sound. Two different semantics carry the name "non-backtracking provenance".

**Repair.** Suppress a rewrite at `(p, ℓ)` only if no Pass-1 redex longer than `ℓ` was routed to that fiber at `p`. With modulus `m ≥ ℓ_max`, same-fiber windows at different positions are disjoint and the guard is sufficient (proof sketch in `PAPER_OUTLINE.md`, Theorem A); `R*` shows it cannot be dropped. On 6,341 words from the 245 affected systems the repaired domain had zero false negatives; on the 2,080 frontier-like words among them it had 2.1% more false positives (581 versus 569). A zero in the provenance bank cannot be used as a certificate of unreachability until the repair is adopted and the general statement is proved.

### M2. The efficiency claim rests on what is and is not charged

"Work" counts successors generated by the simulator. The envelope is computed from the rule system on every frontier word at every decision at no charge. Pass 1 of the fibered bank is exactly `α₂` of each word's complete one-step successor set (verified cell for cell, 19,114 cells). LBF-14's registration required that "No model can examine successors earlier than controls". LBF-15 propagates rule compatibility "without generating successor strings", and from LBF-16 the envelope computes the successors' exact abstraction.

| Quantity per episode, `rel_prov_opened`, B = 128 | Value |
|---|---:|
| Charged successor checks | 29.2 |
| Successors of all distinct words presented to the network (each enumerated once, explicitly or implicitly) | 125.4 |
| The same as implemented (re-enumerated at every decision) | 398.4 |
| Bidirectional BFS, successor checks to closure with no cap (mean / median / max) | 147.5 / 77.5 / 653 |
| Paired difference, enumerated-once minus BFS | −22.2 [−52.8, +4.8], p = 0.15 |
| Roots on which BFS enumerates fewer | 25 of 48 |
| Rule-window comparisons, model side plus simulator side | 191,192 |
| Rule-window comparisons, `bidir` to its 128 budget | 3,715 |

The results table reports rule-window checks of 0 for `bidir`, `astar`, `geom_mm` and `exact_lookahead`. Their `neighbors` calls perform 16(L−1) + 8(L−2) such comparisons per expanded word. In the paper's own primitive the controller does 51 times the matching of `bidir`.

Uninstrumented wall clock on one core: 275–278 ms per episode for `rel_prov_opened`, 76–98 ms for `sym_prov_readout`, about 20 ms for the zero-parameter exact control, 1.3 ms for `bidir` (archived). A successor check in this domain costs microseconds. The controller wins on time only if a successor check costs more than about 5 ms against `bidir` or 9 ms against `geom_mm`. That regime exists (native tactic execution) and the submission contains no measurement in it.

Reported efficiency figures:

- **Wall clock.** `run_lbf17_episode` times a loop that also runs `cached_decision_flops`, an extra forward pass under `FlopCounterMode` for each unseen `(arch, m0, m1, L)`, and, for three seed-107 arms, `probe_lbf17_decision`, which runs eight forward passes with tensor capture. Re-running the registered runner as archived gives 1,333 ms against 275 ms uninstrumented for `rel_prov_opened` (4.8×) and 418 against 76 for `sym_prov_readout` (5.5×). Archived-over-clean ratios range from 3.0 to 8.3 across arms, so the archived table mis-orders them: `rel_prov_mat_zero` (993 ms) and `rel_prov_opened` (1,713 ms) execute the same computation, and `compute_nonrecurrent` (1,515 ms) is reported faster than `rel_prov_opened` although it is 1.4–1.8× slower when both are timed cleanly. The ratio to `sym_prov_readout` survives: 2.46 reported, 2.8–3.6 uninstrumented.
- **FLOPs.** `FlopCounterMode` counts matmul and convolution operators. 60% of `rel_prov_opened`'s CPU self-time is in operators it does not count; for `compute_nonrecurrent` the figure is 25%. Nanoseconds per counted FLOP range from 0.14 to 0.42 across arms. The FLOP cache key omits `mode`.
- **Memory.** `estimate_peak_tensor_bytes` is a closed-form expression. Measured peak allocation on identical states is 9.96 MB against an estimate of 1.35 MB for `rel_prov_opened` (7.4×), and 0.97 MB against 1.35 MB for `sym_prov_readout`. The table shows `rel_prov_opened` using less than `sym_prov_readout`; measured, it uses ten times more. "84.0% less than `compute_nonrecurrent`" becomes 33–60%.
- **"42.7% fewer FLOPs than compute-matched."** The control was padded with redundant views to exceed the registered 1.04× floor. On identical states it carries 1.45× the counted FLOPs and 1.43× the time. The comparison is conservative for the recurrence claim and says nothing about efficiency.

### M3. The decisive controls are missing

No registered symbolic control receives the uncharged rule-conditioned lookahead that every learned arm receives. `exact_lookahead` is charged for its probes and scores with a Hamming bound; it closes fewer roots than `geom_mm` (77.1% versus 81.3%), so it shows that the baseline is poorly designed, not that abstraction beats exact sets.

Controls added here, zero learned parameters, defined before being run, calibration cohort, B = 128:

| Controller | Lookahead given free | Closures | Penalised work | 5-budget AUC |
|---|---|---:|---:|---:|
| `bidir` | none | 30/48 | 84.5 | 34.2% |
| `geom_mm` | none | 39/48 | 59.1 | 52.9% |
| `absint1_mm`: `geom_mm` with the bound from 1-pass envelopes | abstract, 1 hop | 43/48 | 50.8 | 60.8% |
| `absint_mm`: bound from the registered 3-pass `prov4×adj2` envelopes | abstract, 3 hops | 45/48 | 44.9 | 64.2% |
| `exact_mm`: bound from exact radius-3 balls | exact, 3 hops | 48/48 | 31.4 | 73.3% |
| `param_nonrecurrent` (learned, 1 pass) | abstract, 1 hop | 144/144 | 31.8 | 72.9% |
| `sym_prov_readout` (symbolic envelope, learned readout) | abstract, 3 hops | 144/144 | 30.1 | 73.6% |
| `rel_prov_opened` (primary) | abstract, 3 hops | 144/144 | 29.2 | 75.1% |

Three readings follow. First, the sound envelope alone, with no learning, moves `geom_mm` from 59.1 to 44.9 work and from 39 to 45 closures. Second, the learned readout over that envelope reaches the level of exact three-step lookahead; the training target is built from exact radius-3 balls (`bounded_balls`) and fixed-policy continuation costs, so the readout is trained to imitate that oracle. Third, the zero-parameter exact control is within 2.3 successor checks of the primary arm and runs in about 20 ms; its 3,389 uncharged successor generations per episode are what the envelope replaces. That replacement is the defensible contribution, and it matters only when concrete successors are expensive.

A learned readout over Hamming-only features, with no envelope, is still missing. Without it the gain cannot be divided between "learning" and "envelope information".

### M4. Recurrent computation is not established

The envelope's support is fixed by the architecture: every nonlinearity is zero-preserving and strictly positive on positive inputs, so over the reals the support is independent of the weights. This is the paper's best idea. It also means the multi-hop propagation is symbolic computation executed in tensors; training changes magnitudes on a fixed support. The relevant control is therefore `sym_prov_readout`, and against it there is no detectable effect (claim 9 above). It is a registered arm but appears in no registered screen.

For the registered Screen 2A comparators:

| Comparator | Δ work | sign-flip p | Holm, 3 comparisons | Holm, 20 comparisons | Δ AUC (pp), 95% |
|---|---:|---:|---:|---:|---:|
| `rel_prov_standard` | −3.15 | 0.0024 | 0.007 | 0.022 | +1.67 [+0.14, +3.47] |
| `param_nonrecurrent` | −2.68 | 0.0061 | 0.012 | 0.049 | +2.22 [+0.42, +4.58] |
| `compute_nonrecurrent` | −2.54 | 0.0166 | 0.017 | 0.116 | +1.11 [−0.69, +3.06] |

The work contrasts hold within the registered family of three and weaken across the twenty contrasts the paper reports. The AUC criterion is passed on point estimates of 12, 16 and 8 episodes out of 720, in table order; per-root budget outcomes were not archived, so these intervals did not exist before this audit. Seed-level means for one architecture span up to 3.1 work units (`compute_nonrecurrent`: 33.5, 30.4, 31.2), the size of the contrasts, and three checkpoints do not support seed-level inference.

Statements in §5.1 of the results that do not survive adjustment within the seven-sham family: `ablate_axial` (p = 0.049, adjusted 0.146), `scramble_ribbon` (0.052, 0.146), `mat_zero` (0.032, 0.129), `mat_roll` (0.091). The provenance-fiber contrast against `adj2_noprov_opened` (−1.06) has p = 0.090. Robust: `scramble_window`, `env_zero`, `env_roll`, and the contrasts against `lbf16_pos1_opened` and `lowrank_opened`.

LBF-16 has the same shape. Its Screen 2A is recorded as passed with a closure contrast of +1.39 pp [−1.39, +4.86] and a work contrast of −0.88 [−2.45, +0.44] against the single-pass control.

### M5. The 80 / 20 attribution

The joint cell was run (`env_mat_zero`, both banks zeroed at k ≥ 2, on a patched copy of the module verified output-identical on registered modes).

| Penalised work, B = 128 | Matrix bank intact | Matrix bank zeroed |
|---|---:|---:|
| Envelope banks intact | 29.15 | 30.38 |
| Envelope banks zeroed | 34.07 | 34.87 |

Interaction −0.42 [−1.48, +0.54], p = 0.44; additive prediction for the joint cell 35.29. The Shapley share of the envelope is 0.82 [0.68, 0.95]; in AUC, 0.82 [0.63, 0.98]. The additive reading is not contradicted. Three corrections remain. "80.1% / 19.9%" carries precision the data lack. The matrix-bank effect is not robust (M4): zeroing the matrix bank reduces work on 7 of 48 roots, and zeroing the envelope banks does so on 7. Most important, the joint-zero arm (34.87) is worse than the retrained single-pass arms (31.7, 31.8) and than `sym_prov_readout` (30.1). Ablating a bank in a model trained with it measures dependence. The contribution of recurrence, by retraining, is 2.5–2.7 work, not the 6.1 that the ratio divides.

### M6. Strength of the generalisation evidence

- **n = 48.** Seeds share roots. 48/48 gives a two-sided 95% interval of [92.6%, 100%].
- **Ceiling.** The largest work in any `rel_prov_opened` episode is 73 of 128. Nine arms tie at 100%. Closure at B = 128 does not discriminate among learned arms; the paper's distinctions rest on work differences of 1–3 units with root-level standard deviations of 4–9.
- **Difficulty.** Median BFS work to closure is 78; the budget sits just above it. Certified proofs have length 2 to 10, median 6. The exact 3-ball around an endpoint holds 50 words on average.
- **Distribution.** Calibration is an i.i.d. draw from the training generator, with L ∈ {10, 11}, eight length-2 and four length-3 rules. Lineage disjointness is near-certain a priori for random 12-rule systems and is not evidence of shift. The controller is rule-conditioned and alphabet-equivariant, so transfer across rule systems is largely built in. The only shift available is the sealed L = 13 cohort.
- **Development.** Six registered studies ran on one day against one generator, with B, L and K fixed throughout. The specification committed with the registration reports neural precision and recall figures for `FastProvenanceBigramCell` (99.37% against 68.16%) that match neither the final train nor calibration values, so the method was developed and measured before registration on unnamed systems. "Prospective" describes the cohort, not the design.

### M7. What the formalisation covers

"Formally verified abstract interpretation" over-describes the Lean file. See §5. The identity between `SparseMultiplicativeRewriteCell` / `FastProvenanceBigramCell` and any Boolean transfer function, trellis exactness, the local form of the best `A₂` transformer, pass-1 exactness, the suppression rule, the mod-4 routing and all three equivariance statements are unformalised. Soundness of the positionwise domain is proved for a Boolean operator; it transfers to the network only over the reals. The certificates establish that each returned path is a valid equational proof and nothing about accounting, budgets or abstraction.

### M8. Registration and custody

In favour: the registration, specification and Lean file were committed at 11:56:51 (+08:00) in `f52f5bc` and never changed; implementation commits run from 12:12 to 15:30; results at 18:24. The last code change before scoring, by its message and its position in the code, fixes a check that precedes any calibration episode. Fit time totals 3.8 CPU-hours. The history is consistent with one scoring pass. The screens as coded match the screens as registered. LBF-14 and LBF-15 report their failed screens plainly.

Against:

- Timestamps are author-side. The PR was opened and merged in four seconds after results existed; no external record orders registration before data.
- The screens have no inferential criterion. 2A passes on any positive AUC difference. 2B is a property of the symbolic domain measured before registration, and its AUROC threshold is met by Boolean support alone (0.9897 against 0.90). 2C allows a sham to beat the intact model by 0.5 pp.
- Architectures changed after registration: `cal_w` adds 4 parameters (20 for `lowrank_opened`). `param_nonrecurrent` has 53,093 parameters, outside the registered window [53,041, 53,089]. The results state a new window without noting the deviation.
- `sealed_outcomes_opened`, `lbf16_sealed_cohorts_preserved_unopened` and `native_online_tactic_calls` are literals in the manifest writer.
- `verify_historical_disjointness` parses `ordinary.jsonl` and `ood.jsonl` and regenerates every historical suite including its sealed splits. "Strictly unopened" is not literally true, and nothing prevents silent evaluation.
- `--resume` exists and no run log is archived.

---

## 4. Mathematical counterexamples

| ID | Claim refuted or bounded | Witness | Reproduce |
|---|---|---|---|
| CE-1 | Slice-based root suppression is sound for all L (Thm 2.2(3), §2.3) | `R* = {abc↔def, ab↔de, bf↔cc}`, `w0 = abc·a^{L−3}`, L ∈ {3, 4, 10, 11, 13}: (1, `c`) reachable in 3 steps, unsupported at K = 3; `abf…` unsupported at Pass 2. Exact zero activation on six checkpoints. Kernel-checked for the Lean executable model | `audit_01`, `audit_02` |
| CE-2 | Characterisation | A concrete word is lost only when a Pass-1 redex of length 3 and a suppressed rewrite of length 2 share a position and the third symbols differ. No false negative in any system without the nested pair (388 of 400 fully checked systems); none after the repair (6,341 words) | `audit_01` |
| CE-3 | Support equality "for all k ≥ 1" in the implemented arithmetic | Archived `rel_prov_standard`: at K = 3, 1,440–2,465 of 68,066 supported cells ≤ 1e-6 (minimum 8e-32 to 3e-30); at K = 4, 82–114 exact zeros; at K = 6, 8,238–11,176. Archived `rel_prov_opened`: minimum positive activation 0.34 at K = 3, 0.02–0.07 at K = 4, 5e-7 to 2e-5 at K = 6, 3e-30 to 2e-12 at K = 12; first cells under 1e-6 at K = 6. `σ(−110) = 0` in float32 | `audit_07` |
| CE-4 | "Rational saturation avoids exponential attenuation" | `ρ_ε(x) ≈ x/ε` for `x ≪ ε`; nine edges at `x = 1e-9` give a mask of 1.4e-45, the float32 subnormal floor. The mask is computed under `torch.no_grad()`, so the filter is a detached mask, not a differentiable operator | `audit_07`, code |
| CE-5 | Pass k means "≤ k concrete steps"; residual false positives are chimeras | `R = {ab↔cd}`, `w0 = abeeab`: both redexes fall in fiber 0 and the Pass-1 fiber concretises to `{abeeab, abeecd, cdeeab, cdeecd}`, including a distance-2 word. On the 96 calibration endpoints, 27 of 28 K = 3 "false positives" of `prov4×adj2` are first reachable at distance 4 or 5; for `adj2`, 118 of 124; for `sub3`, 8 of 8; for `pos1`, 937 of 1,377 within 8 | `audit_10` |
| CE-6 | Fibers separate incompatible branches | Mean 5.3 Pass-1 redexes per word against 4 fibers: 426 of 467 words have two or more redexes in one fiber. With a single fiber the suppression rule yields 2 false-negative cells on registered calibration endpoints. Per-position fibers reduce K = 3 false positives from 28 to 17 | `audit_07` |

Consequences for the text. Precision and "false-positive reduction" against the exact 3-ball mostly measure horizon alignment for the relational domains, not chimera removal. The "false bridge rate" inherits the same ambiguity, since every source and target are connected by construction. The row "active bigram support, Pass 1 → Pass 3" compares `rel_prov_standard` (16.72%) with `rel_prov_opened` (17.21%) at a 1e-6 threshold. Their symbolic supports are identical on identical words; the gap is consistent with threshold loss and cannot be read as a difference in support.

---

## 5. Lean audit, theorem by theorem

All fifteen compile standalone with zero axioms. No `sorry`, `axiom` or `native_decide`.

| Theorem | Statement proved | Report claim it supports | Gap |
|---|---|---|---|
| `abstract_iter_soundness` | Monotone `F`, step-sound `Fs` ⇒ `Fᵏ(S₀) ⊆ γ(Fsᵏ(a₀))` | k-step soundness schema | None. Textbook |
| `best_transformer_soundness` | Instance for `Fs = α∘F∘γ` | Same | None |
| `galois_pos1` | `α₁(S) ≤ A ⇔ S ⊆ γ₁(A)` on `Fin L → Sym` | Thm 1.1 | Cartesian-closure remark unformalised (trivial) |
| `galois_adj2`, `galois_sub3` | Adjunctions for 2- and 3-windows | Thm 2.1(1) | None |
| `adj2_refines_pos1`, `sub3_refines_adj2` | `γ₃α₃ ⊆ γ₂α₂ ⊆ γ₁α₁` | Thm 2.1(1) | Strictness shown by instances only |
| `provenance_fibers_refine_merged` | `⋃_b γα(S_b) ⊆ γα(⋃_b S_b)` | Thm 2.2(1) | States that fibering is no less precise. Says nothing of a fibered transfer, of `p mod 4`, or of soundness |
| `repaired_lbf16_horizon_support_soundness` | `F_Rᵏ({w₀}) ⊆ γ₁(Tᵏ(α₁{w₀}))` for the Boolean multiplicative transfer `T` | Thm 1.2(1) | No link from the float32 cell to `T`; `T = α₁∘F∘γ₁` unproved; pass-1 exactness unproved in general |
| `nonbacktracking_provenance_soundness` | Last-exit decomposition of concrete walks from `w₀` | Thm 2.2(2) | Justifies removing the word `w₀`. Does not justify slice-based suppression. See CE-1 |
| `cex1_concrete_component_exact` | `rfl`: component `{abc, dec, afa}` | Counterexample 1 | None |
| `cex1_pos1_pass1_exact_pass2_spurious` | `rfl` | Thm 1.2(3) | None |
| `cex1_relational_and_provenance_exact` | `rfl` at passes 3 and 4 | Thm 2.2(3) on one instance | "and beyond" is a comment |
| `cex2_adj2_spurious_sub3_and_prov_exact` | `rfl` | Counterexample 2 | None |
| `cex3_sub3_loses_global_consistency_prov_exact` | `rfl` | Counterexample 3 | Uses the word-filter model for bigram provenance and the slice-suppression model for positionwise provenance |

Unformalised statements of the specification: Thm 1.2(2); the best-transformer identity of §1.2; Thm 2.1(2) and 2.1(3); every property of `𝒯_ε`; Thm 2.2(3); all of §2.3; anything about the neural cells. "Zero axioms" is correct and expected for finite combinatorics closed by `rfl`.

The "dependent transport" family is the same search problem. Its certificate is `Eq.mp (congrArg P proof) witness` around the identical equality proof. It is evidence about equational rewriting, not about dependent types.

---

## 6. Related work and novelty

| Contribution | Assessment | Closest prior work |
|---|---|---|
| Galois connections `(α₁, γ₁)`, `(α_n, γ_n)`; refinement chain | Standard. `α₁` is the Cartesian (independent-attribute) abstraction; `α_n` is the position-indexed n-window abstraction | Cousot & Cousot 1977, 1979; locally testable languages, McNaughton & Papert 1971 |
| Trellis exactness, `α_nγ_n` in O(L·6ⁿ) operations | Standard. Arc consistency on a chain is backtrack-free; equivalently forward–backward on a layered automaton | Montanari 1974; Freuder 1982 |
| Local form of the best `A₂` transformer on reduced states | Correct, short, unproved in Lean. A routine lemma | Regular model checking with abstraction by bounded-length languages: Bouajjani et al. 2000; Bouajjani, Habermehl & Vojnar 2004. For terms: tree-automata completion, Genet 1998; Feuillade, Genet & Viet Triem Tong 2004 |
| Abstraction-derived admissible bounds for bidirectional search | Not claimed, but it is what the envelope provides, and it is an old idea | Pattern databases, Culberson & Schaeffer 1998; Cartesian abstraction refinement, Seipp & Helmert 2018; merge-and-shrink, Helmert et al. 2014; front-to-front bidirectional search, de Champeaux & Sint 1977; MM, Holte et al. 2017 |
| Multiplicative rewrite envelope with zero-preserving support | Architectural adaptation of a semiring fact: the support map from a positive semiring to the Booleans is a homomorphism | Supports of weighted automata, Kirsten 2011; Droste, Kuich & Vogler 2009. Provenance semirings, Green et al. 2007, and their differentiable use in Scallop, Li, Huang & Naik 2023. Product conjunctions in differentiable provers: Rocktäschel & Riedel 2017; Cohen 2016; Evans & Grefenstette 2018 |
| Non-backtracking decomposition | Elementary last-exit argument | — |
| Spatial-congruence fibers | **The one mathematically new device.** A bounded trace partition keyed by the residue of the first redex position, chosen so that the key is permutation-equivariant and same-key windows are disjoint. I did not find a precedent for this key | Trace partitioning, Mauborgne & Rival 2005; Rival & Mauborgne 2007; disjunctive completion, Cousot & Cousot 1979 |
| `S₆` and frontier-set equivariance | Standard constructions, correctly applied; verified here at trajectory level | Zaheer et al. 2017; Hartford et al. 2018. Symbol-equivariant recurrent models guiding symbolic solvers: Bertram et al. 2026 |
| A network whose state is by construction a sound abstract iterate, feeding a learned meeting policy | **Novel as an architecture, to my knowledge.** Abstraction-guided search with learned components exists in synthesis; abstract interpretation *of* networks is the reverse direction | Wang, Dillig & Singh 2018; Guria, Foster & Van Horn 2023; neural guidance of equational provers, Piepenbrock et al. 2022; Loos et al. 2017; learned heuristics for combinatorial search, Agostinelli et al. 2019. Reverse direction: Gehr et al. 2018; Mirman et al. 2018 |

Terminology. "Provenance" here is a first-step class label, not provenance in the semiring sense; "trace partition" is the accurate name. "Relational abstract interpretation" is fair for `A₂` and `A₃`. The paper should cite the planning literature on abstraction heuristics and position the envelope against it; `absint_mm` in M3 is such a heuristic.

Bertram et al. report that neural guidance pays when the search space is large and the solver can override hints. Both conditions bear on M2 and M6: the benchmark's search space is small.

---

## 7. Mechanistic interpretation

Established by the factored interventions:

- The controller depends on correct rewrite-window alignment (`scramble_window`: +20.0 work, robust).
- It depends on its envelope banks at k ≥ 2 (`env_zero` +4.9, `env_roll` +3.1, robust).

Not established:

- That useful *recurrent* computation occurs beyond symbolic propagation. The envelope's support is weight-independent; the symbolic-envelope control ties.
- Independent contributions of the axial block, ribbon and matrix bank (not robust to multiplicity).
- That provenance fibers help search (−1.06 work, p = 0.09), as distinct from tightening support.

Alternative explanations, with the evidence from M3:

| Explanation | Evidence | Share of the gap from `bidir` (84.5) to the primary arm (29.2) |
|---|---|---|
| Node-level front-to-front selection | `geom_mm` 59.1 | 25.4 of 55.4 |
| Explicit relational structure given free (sound envelopes, no learning) | `absint_mm` 44.9 | a further 14.2 |
| Training supervision from exact radius-3 balls, distilled into a readout | `sym_prov_readout` 30.1; `exact_mm` 31.4 | a further 14.7 |
| Neural recurrence in the envelope | `rel_prov_opened` 29.2 | 1.0, not significant |

The decomposition is post hoc and path-dependent; it is offered as the null model the paper must beat. The continuous probes in §5.2 of the results (gate means, effective rank, Moran's I) are descriptive and are not tied to outcomes.

The glyphs. Four-level quantised brightness separates certified from spurious cells at AUROC 0.9899; Boolean support gives 0.9897. The interpretable content of a glyph is the symbolic support, which is computable without the network. The within-support AUROC of 0.93 rests on 56 negative cells from one checkpoint; by CE-5 most such cells are probably reachable one step later.

---

## 8. Earlier studies, LBF-14 to LBF-16

- **LBF-14.** Both screens missed and reported as missed. The frontier-geometry analysis is the most informative document in the series: it shows a node-level oracle closes 31 of 32 roots where `bidir` closes 23, which is the headroom every later study draws on.
- **LBF-15.** Screens 1B, 2A and 2B missed or inverted and reported as such. The diagnosis (sibling blurring, non-equivariant embeddings) is post hoc but was tested on a fresh seed in LBF-16.
- **LBF-16.** The registered Screen 2A reads "AUC and/or closure", a fork. Recorded as passed with intervals that include zero (M4). `opened_no_persist` ties the primary arm. Its Property 2 ("exact horizon support") is false, as LBF-17 shows; the correction is a merit of the programme.

Across the series, the gain over `bidir` arrives in LBF-15 with node-level selection, which the zero-parameter heuristics share (88.2% against 87.5%). The gain over those heuristics arrives in LBF-16, with the multiplicative envelope, when exact one-step rule matching first enters the model at no charge. The accounting question in M2 starts there.

---

## 9. Computational efficiency and the LBF-18 direction

**No LBF-18 proposal exists in the repository** at the reviewed commit, on `main` (`31256ab`), in PRs #344–#346 or in the branch list. The following reviews incremental caching as a design, with measurements from the archived model.

| Measurement, `rel_prov_opened`, B = 128 | Value |
|---|---:|
| Decisions per episode | 6.3 |
| Word-envelope evaluations / distinct words | 10,041 / 3,329: 66.8% repeated |
| Pair evaluations / distinct pairs | 35,533 / 15,207: 57.2% recurring |
| Decision time in word-level cells and gates | 64% |
| In pair ribbon projection and convolution | 11% |
| In pairwise overlap construction and glue | 20% |
| In frontier-level layers (pair input, axial block, heads) | 5% |

Assessment:

1. **Exactness.** Word envelopes depend only on (word, rules, K, mode) and are exactly cacheable at inference. Pair features are cacheable up to `rib_pool`. From `pair_in` onward the computation depends on the whole frontier, the remaining budget and `prev_pot`; an incremental axial update would change the function computed.
2. **Ceiling.** A word cache removes at most 0.64 × 0.67 = 43% of decision time (1.75×). Adding a pair cache on ribbon and overlaps removes a further 0.31 × 0.57 = 18% (2.5× in total), before cache overhead.
3. **Consequence.** 2.5× brings the primary arm to about 110 ms per episode, the uncached cost of `sym_prov_readout`, which benefits from the same cache. The ordering against the symbolic controls does not change. Caching does not address M2.
4. **A larger win is available.** Support is symbolic and 17% dense. Compute it with bitsets and evaluate learned magnitudes only on the support.
5. **If registered,** LBF-18 should fix: bitwise equality of cached and uncached outputs on every calibration decision; no instrumentation in the timed region; the same cache for every baseline; time to closure including model computation as the endpoint; and a synthetic successor latency swept over 0, 1, 10 and 100 ms to chart the break-even curve.

Suitability of the symbolic baselines. `bidir` and `astar` are adequate floors. `geom_mm` and `graph_bridge` are reasonable but uninformed about rules beyond degree. `exact_lookahead` is handicapped by design. `sym_prov_readout` is the correct primary comparator and should be promoted to a screen. `exact_mm` and `absint_mm` should be registered.

---

## 10. Broader research potential

The method needs a finite alphabet, a fixed-length positional encoding, length-preserving rules with bounded windows, and cheap enumeration of rule instances. Its value proposition is sound multi-step pruning without concrete successor generation.

| Target | Fit | What stands in the way |
|---|---|---|
| Native Lean proof search | Plausible only where a step is expensive and the state is first-order. This is the one regime in which M2 reverses | Real goals have binders, unification and variable size. The right abstract domain is tree automata (completion), not positional n-grams. No experiment here executes a Lean tactic during search |
| Dependent type transport | No evidence | The transport family is a wrapper around the same equality proof (§5) |
| Program synthesis | Abstraction-guided pruning is established | The contribution would be a learned ranking over a sound abstract state. Pruning cost grows with abstraction granularity, which is M2 again |
| Constraint satisfaction | The trellis is arc consistency on a chain | General CSPs need k-consistency on non-chain graphs; symbol-equivariant recurrent solvers already exist |
| Planning | The envelope is an abstraction heuristic | Must be compared with pattern databases and Cartesian abstraction heuristics on standard domains |
| Compact recurrent reasoning | The most promising direction | In LBF-17 the recurrent transfer function is written by hand. The open question is whether a small recurrent cell can *learn* a sound transfer function, and whether a zero-preserving prior helps it extrapolate in K and L |

**The fresco as a general representation.** Three measurements bound the claim. The fiber bank holds 1,368 Boolean cells per word per pass; the exact 3-ball it summarises holds 50 words, 1,349 bits. There is no compression at this scale. The margin that carries support decays by one to two orders of magnitude per pass beyond K = 3 (CE-3), so depth is limited without renormalisation. Precision against a fixed concrete horizon is confounded with overshoot (CE-5). The representation becomes interesting where reach sets grow exponentially and the envelope does not. The submission does not test that regime. As an interpretability device the glyph is sound in one direction only: dark certifies unreachability for the positionwise and merged-bigram domains over the reals, and for the provenance bank only after the repair in M1.

---

## 11. Independent confirmation protocol

**Material.** LBF-17 `ordinary` (16) and `ood` (16, L = 13), hashes verified. LBF-16 `ordinary` (16) and `ood` (16), regenerable from seed 20261221; not archived. All are lineage-disjoint from the seed-20261227 training set of the frozen checkpoints. Total 32 ordinary, 32 OOD.

**What 32 or 64 roots can decide.**

| Question | Calibration effect, root SD | Roots for 80% power | With the sealed cohorts |
|---|---|---:|---|
| Work versus `graph_bridge` / `geom_mm` | −30, 30 | 9 | Confirmable |
| Work versus `bidir` | −55, 31 | 3 | Confirmable |
| Closure > 90% | — | 32 with no failure | Confirmable at one-sided 0.05 only if 32/32 close |
| Closure > 95% | — | 59 with no failure | Only if all 64 close, pooling L = 13 |
| Work versus `lbf16_pos1_opened` | −6.5, 14.8 | 41 | Marginal |
| Work versus `compute_nonrecurrent` | −2.5, 7.7 | 72 | Not confirmable; detectable effect at n = 32 is 3.8 |
| Work versus `sym_prov_readout` | −1.0, 4.9 | 190 | Not confirmable; detectable effect at n = 32 is 2.4 |

**Procedure.**

1. *Freeze before unsealing,* in a signed tag: evaluation code hash; the 24 checkpoint hashes; the analysis script; the arm list; the decision rules below. Executed once by someone other than the author. Raw per-episode outputs published. A rerun is permitted only for a hash mismatch or a crash before any outcome is displayed.
2. *Arms.* `rel_prov_opened` (primary, as registered and with the M1 repair as a separate arm), `sym_prov_readout`, `compute_nonrecurrent`, `param_nonrecurrent`, `lbf16_pos1_opened`, `bidir`, `geom_mm`, `graph_bridge`, plus `exact_mm`, `absint_mm` and uncapped `bidir`.
3. *Confirmatory hierarchy,* fixed sequence, each at one-sided 0.025, root-paired sign-flip test, seeds averaged within root:
   - H1: penalised work at B = 128 lower than the better of `geom_mm` and `graph_bridge`, ordinary cohort (n = 32).
   - H2: the same on the L = 13 cohort (n = 32).
   - H3: closure above 88% on the ordinary cohort, exact binomial. At this level it requires 32 of 32 (0.88³² = 0.017).
   Testing stops at the first failure.
4. *Estimation only,* with intervals and no verdict: contrasts against `sym_prov_readout` (equivalence margin ±2 work, declared in advance), the two single-pass controls, and `exact_mm`; successors enumerated once against uncapped BFS; uninstrumented time to closure.
5. *Soundness endpoint.* Symbol-level false negatives of the envelope at strict positivity on every frontier word of every episode, against exact BFS. On the 64 sealed roots expect two or three systems with the nested rule pair, so a zero there is uninformative. On the enriched cohort below, this review predicts false negatives in at least one system in ten under the registered rule and none under the repair.
6. *Inference is conditional on three checkpoints.* Claims about the architecture need at least ten fresh training seeds and are outside the confirmation.

**A fresh cohort is required for the scientific contrasts.** Generate 200 ordinary and 200 L = 13 roots, plus a cohort enriched to 100 systems carrying the nested rule pair, from a seed fixed by a public randomness beacon value published after the freeze tag. This also replaces the procedural seal with a commit–reveal one. Exclude the audit seeds listed in `audit/README.md` by lineage. Cost at uninstrumented speed: a few CPU-hours.

---

## 12. Minor objections

1. The results table reports rule-window checks of 0 for controls that call `neighbors` (M2).
2. `evaluate_abstraction_and_glyphs` uses seed 107 only, at most four words per side per root, and a 1e-6 threshold; the text says "cell-for-cell identical".
3. Closures at B < 128 are replay-checked in Python, not kernel-certified, and their records are not archived.
4. The bootstrap uses 2,000 resamples with one fixed seed and percentile intervals at n = 48; several intervals are degenerate at [0, 0].
5. "Statistically significant (95% CI strictly excluding 0)" is used for families of 7 to 20 unadjusted contrasts.
6. `trellis_arc_filter_soft` is described as differentiable and smooth; its mask is detached.
7. Teacher coefficients (0.70, 0.25, 0.90, 0.04, temperature 4.0, cost cap 48) do not appear in the LBF-17 registration and are carried over unchanged from the LBF-16 code.
8. `SCHEMA` strings and certificate namespaces still read `lbf12`/`lsa02k`; harmless, confusing.
9. The LBF-16 and LBF-17 results are dated the same day as their registrations; state times, not dates.
10. "OOD" is a change of length from 10–11 to 13 with the same rule distribution. Call it a length shift.
11. Parameter matching to 0.09% is reported prominently; time matching, which matters for M2, is not attempted.
12. The compute-matched control is registered as ≥ 1.04× the three-pass FLOPs, tested as ≥ 1.0×, and measures 1.45× on identical states (1.75× along its own trajectories). Report the measured factor.

---

## 13. Ranked follow-up experiments

| Rank | Experiment | Why | Cost |
|---:|---|---|---|
| 1 | Adopt the suppression repair; state and prove the general soundness theorem in Lean; add a nested-pair cohort to the diagnostics | Removes a false theorem; makes the central claim true | Days |
| 2 | Register `sym_prov_readout` as primary comparator; add `exact_mm`, `absint_mm` and a Hamming-only learned readout; report enumerated-once successors and uncapped BFS | Decides between "recurrent computation" and "distilled lookahead" | Hours |
| 3 | One-shot sealed-cohort confirmation and a beacon-seeded fresh cohort (§11) | Converts calibration results into evidence | Hours |
| 4 | Expensive-successor regime: synthetic latency sweep, then native Lean `rw` execution inside the loop | The only regime in which the efficiency claim can be true | Days |
| 5 | Scale until BFS fails: L ∈ {16, 20, 24}, more rules, budgets where `bidir` needs 10⁴–10⁶ nodes; K ∈ {3…8} with margin monitoring | Tests the representation where it could matter; exposes CE-3 and CE-5 | Days |
| 6 | Ten or more training seeds for the four key arms; crossed root × seed model | Makes architecture-level claims possible | CPU-days |
| 7 | Learned transfer function: train a recurrent cell to match `α∘F∘γ` on random systems; test extrapolation in K and L with and without the zero-preserving prior | The question relevant to compact recurrent reasoning | Days |
| 8 | 2 × 2 with retraining (with and without each bank) | Separates dependence from contribution (M5) | CPU-day |
| 9 | Term-level generalisation by tree-automata completion | Prerequisite for real Lean goals | Weeks |
| 10 | LBF-18 caching under the conditions in §9 | Engineering; does not change conclusions | Days |

---

## 14. Limits of this review

- Training was not re-run. That the archived checkpoints result from the registered procedure is supported by matched draw hashes and timing, not by replication.
- The added controls and the joint intervention were designed after the calibration outcomes were known, and use those roots. They are evidence about alternative explanations only.
- Timings come from a shared two-core machine; ratios within a script are reliable, absolute values indicative.
- The sealed cohorts were not opened, so nothing here bears on out-of-sample performance.
- The Monte Carlo rates for CE-1 come from the pinned generator with audit-only seeds and a specific choice of frontier-like words.
- `AGENTS.md` asks that the EvalMath mechanistic-power skill and the FSI-01 gate be consulted. Neither could be opened from this session, and the `spiral_ln` package is not vendored. The analysis applies the rules stated in `AGENTS.md` (root as unit, cluster-aware inference, controls and raw paired outcomes retained), and per-case receipts are exported in the bridge's format under `audit/outputs/fsi_receipts/`.
- The novelty assessment reflects a targeted search, not a systematic one.

---

## References

- Agostinelli, McAleer, Shmakov, Baldi (2019). Solving the Rubik's cube with deep reinforcement learning and search. *Nature Machine Intelligence* 1.
- Bertram, Bhavnani, Freinschlag, Kobler, Mayr, Klambauer (2026). G-RRM: Guiding symbolic solvers with recurrent reasoning models. arXiv:2607.02491.
- Bouajjani, Habermehl, Vojnar (2004). Abstract regular model checking. CAV, LNCS 3114.
- Bouajjani, Jonsson, Nilsson, Touili (2000). Regular model checking. CAV, LNCS 1855.
- de Champeaux, Sint (1977). An improved bidirectional heuristic search algorithm. *JACM* 24(2).
- Cohen (2016). TensorLog: a differentiable deductive database. arXiv:1605.06523.
- Cousot, Cousot (1977). Abstract interpretation: a unified lattice model for static analysis of programs by construction or approximation of fixpoints. POPL.
- Cousot, Cousot (1979). Systematic design of program analysis frameworks. POPL.
- Culberson, Schaeffer (1998). Pattern databases. *Computational Intelligence* 14(3).
- Droste, Kuich, Vogler, eds. (2009). *Handbook of Weighted Automata.* Springer.
- Evans, Grefenstette (2018). Learning explanatory rules from noisy data. *JAIR* 61.
- Feuillade, Genet, Viet Triem Tong (2004). Reachability analysis over term rewriting systems. *Journal of Automated Reasoning* 33.
- Freuder (1982). A sufficient condition for backtrack-free search. *JACM* 29(1).
- Gehr, Mirman, Drachsler-Cohen, Tsankov, Chaudhuri, Vechev (2018). AI²: safety and robustness certification of neural networks with abstract interpretation. IEEE S&P.
- Genet (1998). Decidable approximations of sets of descendants and sets of normal forms. RTA.
- Green, Karvounarakis, Tannen (2007). Provenance semirings. PODS.
- Guria, Foster, Van Horn (2023). Absynthe: abstract interpretation-guided synthesis. *PACMPL* 7 (PLDI).
- Hartford, Graham, Leyton-Brown, Ravanbakhsh (2018). Deep models of interactions across sets. ICML.
- Helmert, Haslum, Hoffmann, Nissim (2014). Merge-and-shrink abstraction. *JACM* 61(3).
- Holte, Felner, Sharon, Sturtevant, Chen (2017). MM: a bidirectional search algorithm that is guaranteed to meet in the middle. *Artificial Intelligence* 252.
- Kirsten (2011). The support of a recognizable series over a zero-sum free, commutative semiring is recognizable. *Acta Cybernetica* 20(2).
- Li, Huang, Naik (2023). Scallop: a language for neurosymbolic programming. *PACMPL* 7 (PLDI).
- Loos, Irving, Szegedy, Kaliszyk (2017). Deep network guided proof search. LPAR-21.
- Mauborgne, Rival (2005). Trace partitioning in abstract interpretation based static analyzers. ESOP.
- McNaughton, Papert (1971). *Counter-Free Automata.* MIT Press.
- Mirman, Gehr, Vechev (2018). Differentiable abstract interpretation for provably robust neural networks. ICML.
- Montanari (1974). Networks of constraints: fundamental properties and applications to picture processing. *Information Sciences* 7.
- Piepenbrock, Heskes, Janota, Urban (2022). Guiding an automated theorem prover with neural rewriting. IJCAR.
- Rival, Mauborgne (2007). The trace partitioning abstract domain. *ACM TOPLAS* 29(5).
- Rocktäschel, Riedel (2017). End-to-end differentiable proving. NeurIPS.
- Seipp, Helmert (2018). Counterexample-guided Cartesian abstraction refinement for classical planning. *JAIR* 62.
- Wang, Dillig, Singh (2018). Program synthesis using abstraction refinement. *PACMPL* 2 (POPL).
- Zaheer, Kottur, Ravanbakhsh, Póczos, Salakhutdinov, Smola (2017). Deep sets. NeurIPS.
