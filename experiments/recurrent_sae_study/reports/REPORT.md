# Do recurrent loops build interpretable, reusable hidden-state features?

SAE / CKA / intervention study on LaCo-compressed, loop-recovered and HRM-controlled transformers

Status: complete for the small-model path (11 runs). The Qwen3-0.6B path is implemented and smoke-tested but not run on real checkpoints, because those were still being trained by the sibling sessions (§1).
All numbers are on **held-out** examples (disjoint RNG streams from SAE training, feature selection and controller training). Full tables: [tables.md](tables.md). Figures: [../figs](../figs). SAE configs and weights: [../out/sae](../out/sae). Raw result JSON: [../out/res](../out/res).

---

## 0. Bottom line: which measurements earned their keep?

| Measurement | Predictive value | Causal value | Verdict |
|---|---|---|---|
| **SAE sparse code as a linear probe** (1024 latents, k=16) | Best single predictor of "answer will be lost / retained / gained by more loops" in-distribution: ΔAUROC over a confidence+loop-index baseline **+0.125 (loss), +0.070 (retained), +0.076 (gain)**, CI above 0 in 11/11 runs. Beats the same probe on the raw 128-d state (+0.068 / +0.033 / +0.046). **Does not transfer reliably to longer strings** (loss +0.081, 9/11 runs positive; gain +0.011, 6 positive / 4 negative). | Not tested by itself | **Real, but in-distribution only.** |
| **Individual SAE features chosen for "answer-correct" / "about-to-lose"** | Replicate on held-out in-distribution data (sign agreement 0.88–1.0; effect 5–15× the matched-null 95th percentile). Replicate only partially on longer strings (sign agreement 0.43–1.0). | **No.** Ablating the top-8 event-selected features (own model) changes accuracy by ≤0.7 pp at early loops and does *not* prevent answer loss (rescue rate 0–5% vs random edits 0–14%). | Correlational markers, not levers. |
| **SAE features that write the predicted class** (direct-logit weight) | – | **Yes.** Ablating the 8 strongest active ones moves the readout at the edited loop in 37/45 (run, loop) cells (typically −25 to −55 pp) vs 5/45 for count-matched random features and 5/45 for energy-matched random directions. Adding them flips the prediction to the target class in 14–100% of examples that were wrong at that loop, rising with strength (random direction 1–27%). | **Causal for the readout at the edited loop; persistence depends on regime and timing** (§4.5). |
| **SAE scalar signals** (reconstruction error, active-set Jaccard with previous loop, code cosine, carried energy, novel-feature mass) | Small gains over baseline: mean ΔAUROC +0.016 (gain), +0.026 (retained), +0.038 (loss) in-distribution, +0.048 (loss) on longer strings; 9–10 of 11 runs positive except 'gain' on longer strings. **Not clearly better than two non-SAE state-dynamics scalars** (relative state change, cosine to previous state: +0.014 to +0.036). Combining them adds little. | – | Useful but replaceable. |
| **SAE scalars as inputs to the HRM-style Q-controller** | Frontier gain vs the controller without them: **+0.007 to +0.037** (in-distribution, 3/3 runs, CI above 0); longer strings **+0.011, +0.018, −0.004 (n.s.)** in the same three runs. Shuffling the same scalars across examples drops the gain to ≤0, and adding the non-SAE dynamics scalars *hurts* (−0.003 to −0.033). So the gain is example-specific information, not extra input capacity. | – | **Modest, real, small sample (3 runs, 1 controller recipe).** |
| **Code-space similarity to the teacher** (SAE trained on the pruned span) | In the mini-LaCo replica it tracks function: peaks at the trained loop count (T=5) with code cosine 0.74/0.71 (seeds 0/1) vs 0.03 for LaCo-only or untrained loops and 0.35–0.39 for the label-matched null (teacher's adjacent layer: 0.87–0.89). | Ablating the "span-written" features changes accuracy by ≤1.4 pp in both teacher and student → **cannot discriminate** | Descriptive; functional equivalence not established at feature level. |
| **Linear CKA / subspace overlap** | Tracks function while inputs stay in-distribution, **dissociates out of distribution** (CKA 0.87 with 22% accuracy; LaCo-only CKA 0.51 at chance accuracy). | – | Representational similarity ≠ functional equivalence. |
| **SAE reconstruction error alone** | Weak: single-signal AUROC for loss 0.69–0.72 (own model). Rises past the trained loop depth for an SAE trained only on loops ≤ K_train (own dfa8 model: 1.4% → 5%). | – | Weak OOD-depth alarm. |

What the data say about the research questions, in one line each:

1. **Stable features across loops?** Yes. The active-feature set changes slowly and example-specifically (Jaccard between consecutive loops 0.52–0.89 at the trained depth vs 0.25–0.50 for a label-matched null), while the activation *coordinates* drift (own model: CKA between loop 1 and loop 32 ≈ 0.05; top-8 PCA subspace overlap stays 0.8–0.95).
2. **Do pruned layers contain features recurrent adapters rebuild?** Partly. In the mini-LaCo replica (both seeds), loop-recovered states reproduce the teacher's feature code far above the null at the trained depth (0.74/0.71 vs null 0.35/0.39) but below the ceiling of a neighbouring teacher layer (0.87/0.89), only at the trained loop count, and not out of distribution.
3. **Why do correct answers disappear?** Not because of reconstruction error and mostly not because of unfamiliar features. The exact SAE decomposition of the margin loss (SAE residual share ≈ 0) splits into **supporting features switching off (49–78%)** and **rival-class features switching on (31–57%)**; "novel" features (rare within training depth) supply 1–5% of the rival gain in DFA models but 9–42% in cyclic models. Models trained to answer only at the last loop lose mainly by switching supporting features off.
4. **Can feature-level measurements predict when more computation helps or harms?** Yes in-distribution (sparse code probe), weakly and replaceably out of distribution (scalars).
5. **Does it help the controller?** Slightly; see the table above.

---

## 1. What was analysed (and what was not)

The three sibling sessions (LaCo reproduction, "Recurrent recovery from transformer pruning", "Recurrent-recovery with adaptive controller") had only just started when this work began, and their Qwen3 checkpoints did not exist yet. Frozen snapshots of their code are in [../vendor](../vendor) (provenance and hashes in `PROVENANCE.*`).

| Source | Used for | Status |
|---|---|---|
| crrbench looped-transformer grid (weight-tied block, K_train=8, evaluated to 32 loops; `results/full/models`) | RQ1, RQ3, RQ4, interventions, reuse | Analysed: dfa (5-state non-group DFA) and cyclic (Z5), regimes `final` and `random_exit`, seeds 0–1 (+ partial controller runs) |
| Own looped model (dfa 8-state, K_train=16) | Development model; also steering and event-feature ablation | Analysed |
| **Mini-LaCo replica** (12-layer, d=128 causal Qwen3-style teacher → LaCo collapse of layers 2–5 into layer 1 → looped site with loop-specific LoRA, loop norms and gates, distilled from the teacher; `looprec` code from the sibling) | RQ2 (compressed vs recovered), cross-model interventions | Analysed, seeds 0 and 1 |
| crrbench HRM-style Q-controller (CONTINUE / PRESERVE / HALT, price-conditioned) | Stopping-policy test | Analysed on 3 models |
| **Qwen3-0.6B-Base LaCo + recovery** | Everything | **Not analysed.** Pipeline written and smoke-tested end-to-end on the sibling's tiny Qwen3 test checkpoint: `run_qwen.sh`. The sibling's teacher tuning was at step ~350/1200 when I checked. |

The mini replica is my own reproduction at a scale that trains on CPU; its "LaCo" step reuses the sibling's merge arithmetic, but it is **not** the sibling's Qwen run. The Qwen3 results the question asks about therefore remain open; everything below is evidence from small models.

## 2. Method summary

- **Activations.** Residual state at the readout position after each loop (looped models: last token; mini/Qwen: last prompt token), fp16. Mini replica also stores the teacher's states at every layer of the pruned span. Collection sizes: ≤12k sequences per split × ≤33 states × 128 dims (≈85 MB/split).
- **Splits (disjoint RNG streams).** `sae_train` (fits SAEs), `sae_val`, `ana_train` (feature selection, probes, controllers), `ana_val` (operating points), `test_id` (trained string lengths), `test_ood` (longer strings: the trained model's depth/length are exceeded). All reported results use `test_*`.
- **SAEs.** TopK SAE (Gao et al. 2024), 1024 latents, k=16, unit-norm decoder, dead-latent auxiliary loss, 2000–3000 Adam steps. Two per run: one over all loops (`all`) and one over loops ≤ K_train (`trainK`). Held-out NMSE 0.4–1.7% (looped), 2.4–13% (mini recovered student). Configs: `out/sae/*.json`.
- **Stability.** Row-wise Jaccard of active sets, code cosine, carried energy between consecutive loops against a null that pairs each example with another of the same label and length.
- **Events.** For (example, loop t): *stay* (correct at t…t+4), *harm* (correct at t, wrong at t+4), *gain* (wrong at t, correct at t+4). Features are ranked by loop-binned AUROC on `ana_train`, then re-evaluated on `test_*` against random features.
- **Predictors.** L2 logistic regression on: baseline (max-prob, margin, entropy, loop index); + SAE scalars; + non-SAE dynamics; + SAE sparse code; + raw state. 95% CIs by bootstrap over examples (paired for differences).
- **Attribution.** The readout is exactly linear in the last-token state (loop RMS-norm fixes its scale), so the change in (correct − rival) margin between the last correct loop and loop 32 splits exactly into per-feature terms plus a residual.
- **Interventions.** Edit the last-token state right after loop t0, re-normalise to its original norm, keep looping. Arms: target features; count-matched random active features; energy-matched random direction. Paired on identical inputs.
- **Similarity.** Linear CKA, top-8 PCA subspace overlap, principal angles, SAE-code cosine, per-feature correlation; functional measures (accuracy, top-1 agreement with the teacher).
- **Controller.** The sibling's Q-learning controller with one added input block at a time; metric = mean advantage over a budget-matched fixed-depth baseline across the price grid (the sibling's yardstick), FLOP-charged including the SAE encoder.

## 3. Figures

- [figs/all__x1_overview.png](../figs/all__x1_overview.png) accuracy / feature-overlap / SAE-error by loop for every run.
- [figs/all__x2_predictive_gain.png](../figs/all__x2_predictive_gain.png) ΔAUROC forest plot per probe, target and split.
- [figs/all__x3_loss_attribution.png](../figs/all__x3_loss_attribution.png) answer-loss attribution.
- Per run: `f1_behaviour_recon`, `f2_stability`, `f3_prediction`, `f7_atlas` (feature trajectories by outcome type, with the class each feature writes), `f8_loop_cka`, `f9_poscontrol`, `f10_reuse`, `f5_controller`, `f6_hrm_sae`; mini replica: `mini_s*__m2_compare`, `m3_cross_ablation`.

## 4. Results

### 4.1 Loops: behaviour, fidelity, stability (RQ1)

Trained depth is 8 loops for the crrbench grid and 5 for the mini replica; models are evaluated to 4× and 2× that depth. `final`-regime models answer correctly only near the trained loop (cyclic: 0.96–0.97 at loop 8, 0.11–0.25 at loop 32), `random_exit` models retain better but still lose 18–44% of in-distribution answers by loop 32 (Table 1). Over the recovered mini student, accuracy falls from 0.84 (T=5) to 0.16 (T=10).

The shared SAE reconstructs states at every loop, including past the trained depth, with NMSE below 2% for the looped models (Table 1). The active feature set is *specific* to the example and sequence: consecutive-loop Jaccard rises from ≈0.3 (loops 1→2) to 0.52–0.89 at the trained depth, versus 0.25–0.50 for the label-matched null. Answer-lost trajectories show lower overlap after the trained depth than stable ones (own model: 0.75 vs 0.9). Coordinates drift while the dominant subspace stays (own model: CKA loop 1 vs 32 ≈ 0.05; PCA-subspace overlap 0.8–0.95 after loop 8): `figs/dfa_random_exit_K16_s0__f8_loop_cka.png`.

### 4.2 Why answers disappear (RQ3)

Table 4. Over 16 (run, split) cells with ≥350 lost examples each: SAE residual share −0.00 to 0.02 (the decomposition is exact and complete); supporting features switched off 0.46–0.78; rival features switched on 0.31–0.66 (partly offsetting terms make shares sum to 1). Regimes differ: `final`-regime and cyclic models lose mostly through supporting features switching off (0.65–0.78), `random_exit` DFA models are ~50/50. Share of the rival gain carried by features rare inside the training depth (<1% activation): 0.01–0.05 (dfa) but 0.09–0.42 (cyclic), so "the model enters unfamiliar feature territory" is supported only for cyclic. At the SAE level the OOD-depth signature is a rise in reconstruction error for the `trainK` SAE (e.g. 1.4% → 5%).

### 4.3 Prediction (RQ4) — Tables 2, 2b

See §0. The sparse-code probe's in-distribution advantage over the raw state probe suggests the SAE basis is a better-conditioned feature space for linear readout of outcome, not merely a re-encoding. Its collapse out of distribution (loss +0.081 vs raw +0.040 is still better, but gain and stay flip sign in 4 and 1 runs) indicates feature associations tied to the trained length range.

### 4.4 Compressed vs recovered (RQ2) — Tables M, M2

Mini-LaCo replica, `test_id`, seeds 0/1: teacher accuracy 0.98; LaCo-only 0.13/0.18; untrained loops 0.13/0.13; recovered student at T=5: **0.84/0.80** (top-1 agreement with teacher 0.85). SAE_T was trained on the teacher states across the pruned span only.

| | code cosine to teacher output | per-feature correlation | CKA |
|---|---|---|---|
| LaCo-only | 0.03 / 0.04 | 0.01 / 0.01 | 0.15 / 0.13 |
| untrained loops (T=5) | 0.04 / 0.02 | 0.00 / 0.00 | 0.08 / 0.05 |
| **recovered (T=5)** | **0.74 / 0.71** | **0.43 / 0.51** | **0.81 / 0.79** |
| label-matched null | 0.35 / 0.39 | | |
| teacher's adjacent layer | 0.87 / 0.89 | 0.75 / 0.85 | |

(Seed 0 / seed 1.) Code similarity across loops peaks exactly at the trained loop count (seed 0: 0.38, 0.45, 0.57, 0.64, **0.74**, 0.42, 0.18 at loops 1–7; seed 1: 0.48, 0.60, 0.66, 0.70, **0.71**, 0.43, 0.26) and tracks accuracy. Features that change across the pruned span are reproduced no better than other features in seed 1 (0.53 vs 0.54; seed 0: 0.60 vs 0.44), so I do not claim span-specific recovery. Out of distribution the recovered student reproduces little (code cosine 0.32/0.50 vs null 0.27/0.22) and CKA dissociates from function: CKA 0.87 at loop 1 with 22% accuracy, and LaCo-only CKA 0.51 with chance accuracy.
Functional test: replacing the loop-T last-token state by its SAE_T reconstruction costs 0–3 pp (0.73 → 0.70); ablating the top "span-written" features changes accuracy by −1.4 to +0.4 pp in the teacher and in the student alike (Table M2). The ablation is too weak to say anything about feature-level equivalence.

### 4.5 Interventions (RQ5, RQ6) — Tables 5, 5b, and `f9_poscontrol`

- *Class-writing features (positive control).* Ablating the top 8 active features that write the predicted class: Δaccuracy at the edited loop significantly negative in 37/45 cells (−5 to −57 pp), random features 5/45, random direction 5/45. Persistence depends on regime and timing: for `random_exit` models the damage persists at loop 32 (−7 to −33 pp; larger for later edits); for dfa `final` models it largely heals (−0.5 to −5 pp); cyclic `final` is mixed in sign. (At loop 32, random-feature edits are themselves significant in 19/45 in-distribution cells, of either sign and typically 1–4 pp, so final-readout effects below that size should not be read as causal.) A few `final`-regime cells show *positive* Δ at the edited loop: where the predicted class was already wrong, removing it lets another class win. That is an artefact of the metric, not a benefit.
- *Steering.* Adding the class-writing features flips examples that were wrong at the edited loop to the target class in 14–100% (α=0.5–2, rising with α) vs 1–27% for a random direction (own model, both splits). By loop 32 the effect has washed out (62% vs 64% correct for t0=4; late edits 6–27% vs 6–19%), except for longer-string edits at α=2 (0.27 vs 0.06 at t0=24).
- *Event-selected features.* Ablating the top-8 "answer-correct" or "about-to-lose" features (own model, t0 ∈ {4,16,24}): ≤0.7 pp at early loops, −3 to −4.6 pp at final readout for late edits in-distribution (≈0 on longer strings) vs ±0.5 pp for random; rescue of examples that would have lost the answer: 0–5% vs 0–14% for random edits. The features that *predict* loss are not the ones that *cause* it.

### 4.6 Reuse — Table 6

Native resume of a frozen state plus four new symbols (5 runs, seed 0 only): in-distribution resume accuracy 0.20–0.87, below the from-scratch run with the same total loops in 3 of 5 (cyclic random_exit, dfa random_exit, own dfa8) and above it in 2 (dfa final, cyclic final at chance). The SAE code predicts which frozen states resume successfully better than the baseline in 3 of 5 in-distribution runs (+0.026, +0.081, +0.102 AUROC; the raw state gives about the same), and per-feature associations selected on `ana_train` replicate above the matched-null p95 in 5/5 in-distribution runs and, at much smaller size, in 4/5 on longer strings. Cyclic models are at chance on resume (≈0.2) and give no usable signal.

### 4.7 Stopping policies — Tables 7, 8

Controller (sibling's Q-network) with added SAE scalars: advantage over budget-matched fixed depth improves in-distribution on all three models; the shuffled-SAE control gives −0.001 to −0.137, i.e. information is example-specific, and non-SAE dynamics features reduce the metric. Out of distribution two of three improve. These gains are ≈1–4 percentage points of accuracy-at-equal-compute and, for dfa, none of the controllers reaches the accuracy of the best *fixed* depth (≈0.99 at loop 8) — the controller's value is robustness to not knowing the depth, not beating an oracle depth. My simpler hazard-logistic and GRU policies show mixed results (Table 7), with the sparse-code probe's strong in-distribution AUROC not converting into a better stopping policy for `random_exit` models, whose fixed-depth curve is already near ceiling.

## 5. Limitations

- **Scale.** All SAE/CKA/intervention evidence is from d=128 models trained on automaton word problems. Whether the Qwen3-0.6B recovery behaves similarly is untested.
- **Replication.** Two seeds per looped task × regime (own dfa8: one), two mini replicas, three controller runs. CIs are over examples, not over seeds; tallies across runs are not corrected for multiple comparisons.
- **Selection of SAE hyper-parameters.** One setting (1024 latents, k=16) throughout; no sparsity or dictionary-size sweep. With k=16 of 128 dimensions the SAE is a rich basis, which probably helps the sparse-code probe.
- **OOD.** "Out of distribution" means longer strings than trained on, where the teacher model itself is often near chance; conclusions about OOD transfer are mostly "does not hold up".
- **Mini replica.** The teacher is trained from scratch on a single task; the LaCo-merged span is layers 1–5 (early layers), chosen by the LaCo similarity search, which is not typical of Qwen.
- **Controller.** Only the sibling's recipe was tested; my 3 controller runs were trained with 2 seeds and 200–300 iterations each.
- **Interventions** edit one position's state; they do not test distributed, multi-position features.
- Sibling sessions were still running; the snapshot may differ from their final code.

## 6. How to reproduce / extend

```
./run_toy.sh <name>                         # own looped checkpoint: collect → SAE → analyses → controllers
./run_sib.sh <substring> [threads] [stages]  # crrbench grid checkpoints (collect,sae,t3,t4b,t7,t8[,t5,t6])
./run_mini.sh <seed>                        # mini-LaCo replica: teacher → LaCo → recovery → SAE_T / compare / interventions
./run_qwen.sh <sib_root> <run> <tag> cuda   # Qwen3 path for a finished recovery run (re-snapshot vendor/looprec2 first)
python scripts/r1_aggregate.py              # tables.md + cross-run figures
```

Windows note: the session folder's absolute path nearly exhausts MAX_PATH, so heavy artefacts live in `C:\Users\patri\.cache\recsae_data` and libraries in `C:\Users\patri\.cache\recsae_pylibs`.

Code map: `src/recsae` (SAE, statistics, analysis, models), `scripts/t*` (looped models), `scripts/m*` (mini replica), `scripts/q1_collect.py` (Qwen3), `scripts/s1_train_sae.py`, `scripts/r1_aggregate.py`.
